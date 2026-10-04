"""F1 Racer - drive laps around F1-style circuits with the arrow keys.

Controls:  UP = throttle   DOWN = brake / reverse   LEFT / RIGHT = steer
           1-9 = choose circuit   TAB = next circuit   R = restart   ESC = quit

Track: road (dark) and kerbs are fast, sand run-off is slow, red/white barriers
stop you. The HUD shows steering angle, heading, lateral G, and for each corner
its angle, the speed you carried through it and a suggested target speed.

Setup:  pip install pygame
Run:    python f1_racer.py
"""
import math
import sys

import pygame

W, H = 1100, 700
ROAD_W = 70
STEP = 6            # distance between track sample points (px)
A_GRIP = 0.45       # max lateral acceleration (px/frame^2) = 4.0 G
VMAX = 8.0          # px/frame  -> 320 km/h
KMH = 40            # px/frame to km/h

GRASS, ROAD, SAND = (40, 118, 62), (58, 60, 68), (214, 190, 130)
KERBS = [(236, 236, 236), (228, 70, 30)]
WALL_W, WALL_R = (255, 255, 255), (150, 20, 30)

# Circuit layouts: closed loops of control points, smoothed into curves.
# Point 0 is on the start/finish straight. Edit these to design your own!
TRACKS = [
    ("Fast circuit (Monza style)",
     [(300, 610), (560, 610), (820, 610), (930, 590), (985, 540), (1020, 470),
      (1045, 380), (1040, 280), (990, 190), (900, 130), (780, 110), (600, 108),
      (450, 120), (360, 160), (385, 205), (310, 245), (200, 270), (120, 340),
      (100, 450), (140, 550), (210, 600)]),
    ("Street circuit (Monaco style)",
     [(300, 100), (560, 95), (820, 105), (940, 150), (990, 240), (960, 330),
      (880, 380), (780, 370), (700, 300), (610, 250), (500, 300), (480, 400),
      (560, 480), (700, 500), (820, 540), (900, 600), (850, 650), (650, 645),
      (450, 650), (300, 640), (190, 590), (120, 500), (140, 390), (210, 330),
      (190, 250), (160, 170), (220, 110)]),
    ("Flowing GP circuit",
     [(250, 600), (520, 612), (780, 605), (930, 560), (990, 470), (1000, 360),
      (950, 250), (860, 170), (730, 125), (590, 130), (480, 190), (430, 280),
      (340, 300), (240, 260), (170, 190), (110, 300), (105, 420), (150, 520),
      (195, 575)]),
]

# extra circuits (Silverstone, Spa, Suzuka, ...) live in f1_tracks.py next to this file
try:
    from f1_tracks import EXTRA_TRACKS
    TRACKS += EXTRA_TRACKS
except ImportError:
    pass


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def build(name, ctrl):
    """Turn control points into a smooth track: samples, corners and a drawn surface."""
    m = len(ctrl)
    dense = []
    for i in range(m):  # Catmull-Rom spline through the control points
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[(i + 1) % m], ctrl[(i + 2) % m]
        for k in range(40):
            t = k / 40
            t2, t3 = t * t, t * t * t
            dense.append(tuple(
                0.5 * (2 * p1[a] + (-p0[a] + p2[a]) * t
                       + (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * t2
                       + (-p0[a] + 3 * p1[a] - 3 * p2[a] + p3[a]) * t3)
                for a in (0, 1)))
    cum = [0.0]
    for i in range(1, len(dense) + 1):
        a, b = dense[i - 1], dense[i % len(dense)]
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    n = int(total / STEP)
    step = total / n
    pts, j = [], 0
    for k in range(n):  # even spacing along the curve
        s = k * step
        while cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        f = (s - cum[j]) / seg if seg else 0
        a, b = dense[j], dense[(j + 1) % len(dense)]
        pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))

    head = [math.atan2(pts[(i + 1) % n][1] - pts[i][1], pts[(i + 1) % n][0] - pts[i][0]) for i in range(n)]
    dh = [wrap(head[(i + 1) % n] - head[i]) for i in range(n)]

    # --- find corners from how fast the heading changes ---
    w = 6
    flags = [abs(sum(dh[(i + k) % n] for k in range(w))) > 0.2 for i in range(n)]
    runs, i = [], 0
    while i < n:
        if flags[i]:
            s = i
            while i < n and flags[i]:
                i += 1
            runs.append([s, i - 1])
        else:
            i += 1

    def turn(s, e):
        return sum(dh[k % n] for k in range(s, e + w))

    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 4 and turn(*r) * turn(*merged[-1]) > 0:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    corners = []
    for s, e in merged:
        t = turn(s, e)
        if abs(math.degrees(t)) < 20:
            continue
        radius = (e + w - s) * step / abs(t)
        corners.append({"s": s, "e": e + w, "n": len(corners) + 1,
                        "dir": "RIGHT" if t > 0 else "LEFT",
                        "ang": abs(math.degrees(t)),
                        "target": min(math.sqrt(A_GRIP * radius) * KMH, VMAX * KMH),
                        "last": None})

    # --- draw the circuit once: barrier, run-off, kerbs, road ---
    surf = pygame.Surface((W, H))
    surf.fill(GRASS)
    for col, wd in ((WALL_W, ROAD_W + 80), (WALL_R, ROAD_W + 74), (SAND, ROAD_W + 56)):
        for x, y in pts:
            pygame.draw.circle(surf, col, (int(x), int(y)), wd // 2)
    for k, (x, y) in enumerate(pts):
        pygame.draw.circle(surf, KERBS[(k // 3) % 2], (int(x), int(y)), (ROAD_W + 12) // 2)
    for x, y in pts:
        pygame.draw.circle(surf, ROAD, (int(x), int(y)), ROAD_W // 2)
    nx, ny = -math.sin(head[0]), math.cos(head[0])  # start line across the road
    for k in range(10):
        c = (250, 250, 250) if k % 2 else (20, 20, 20)
        o1, o2 = (k - 5) * 7, (k - 4) * 7
        pygame.draw.line(surf, c, (pts[0][0] + nx * o1, pts[0][1] + ny * o1),
                         (pts[0][0] + nx * o2, pts[0][1] + ny * o2), 8)
    return {"name": name, "pts": pts, "head": head, "n": n, "step": step,
            "corners": corners, "surf": surf, "best": None}


def surface_at(tr, x, y):
    xi, yi = int(x), int(y)
    if not (0 <= xi < W and 0 <= yi < H):
        return "wall"
    c = tr["surf"].get_at((xi, yi))[:3]
    if c == WALL_W or c == WALL_R:
        return "wall"
    return "road" if (c == ROAD or c in KERBS or c in ((20, 20, 20), (250, 250, 250))) else "sand"


def fmt(ms):
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}.{ms % 1000 // 10:02d}"


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("F1 Racer")
    clock = pygame.time.Clock()
    f_s, f_m, f_b = (pygame.font.SysFont(None, s) for s in (22, 30, 72))

    cache = {}

    def get_track(i):
        if i not in cache:
            cache[i] = build(*TRACKS[i])
        return cache[i]

    car_img = pygame.Surface((38, 18), pygame.SRCALPHA)
    pygame.draw.rect(car_img, (225, 40, 40), (4, 3, 30, 12), border_radius=4)
    pygame.draw.rect(car_img, (225, 40, 40), (0, 0, 6, 18))       # rear wing
    pygame.draw.rect(car_img, (30, 30, 40), (17, 6, 7, 6), border_radius=2)  # cockpit
    pygame.draw.rect(car_img, (225, 40, 40), (32, 1, 6, 16))      # front wing

    def reset(ti):
        tr = get_track(ti)
        for c in tr["corners"]:
            c["last"] = None
        i0 = tr["n"] - 12
        return {"ti": ti, "x": tr["pts"][i0][0], "y": tr["pts"][i0][1], "a": tr["head"][i0],
                "v": 0.0, "steer": 0.0, "omega": 0.0, "slide": False, "idx": i0, "cum": -12,
                "laps": 0, "t0": pygame.time.get_ticks(), "last": None,
                "cur": None, "cur_min": None, "flash": 0, "hits": 0}

    g = reset(0)

    def panel(rect):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((10, 12, 16, 195))
        screen.blit(s, rect.topleft)

    def text(t, pos, font=f_s, col=(235, 235, 235)):
        screen.blit(font.render(t, True, col), pos)

    def bar(x, y, w, h, frac, col, centered=False):
        pygame.draw.rect(screen, (45, 48, 56), (x, y, w, h))
        if centered:
            half = w // 2
            wd = int(abs(frac) * half)
            pygame.draw.rect(screen, col, (x + half - (wd if frac < 0 else 0), y, wd, h))
            pygame.draw.line(screen, (200, 200, 200), (x + half, y - 2), (x + half, y + h + 2))
        else:
            pygame.draw.rect(screen, col, (x, y, int(max(0, min(1, frac)) * w), h))

    def key(cx, cy, pressed, d):
        r = pygame.Rect(0, 0, 38, 38)
        r.center = (cx, cy)
        pygame.draw.rect(screen, (240, 200, 60) if pressed else (50, 54, 64), r, border_radius=6)
        c = (20, 20, 20) if pressed else (200, 200, 205)
        tri = {"u": [(cx, cy - 9), (cx - 9, cy + 7), (cx + 9, cy + 7)],
               "d": [(cx, cy + 9), (cx - 9, cy - 7), (cx + 9, cy - 7)],
               "l": [(cx - 9, cy), (cx + 7, cy - 9), (cx + 7, cy + 9)],
               "r": [(cx + 9, cy), (cx - 7, cy - 9), (cx - 7, cy + 9)]}[d]
        pygame.draw.polygon(screen, c, tri)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    g = reset(g["ti"])
                elif pygame.K_1 <= e.key <= pygame.K_9 and e.key - pygame.K_1 < len(TRACKS):
                    g = reset(e.key - pygame.K_1)
                elif e.key == pygame.K_TAB:          # next circuit
                    g = reset((g["ti"] + 1) % len(TRACKS))

        tr = get_track(g["ti"])
        pts, n, step, corners = tr["pts"], tr["n"], tr["step"], tr["corners"]
        k = pygame.key.get_pressed()
        up, down, left, right = k[pygame.K_UP], k[pygame.K_DOWN], k[pygame.K_LEFT], k[pygame.K_RIGHT]

        # ---------- physics ----------
        under = surface_at(tr, g["x"], g["y"])
        on_sand = under == "sand"
        if up:
            g["v"] += 0.12 * (1 - max(g["v"], 0) / (VMAX + 0.5))
        if down:
            g["v"] -= 0.28 if g["v"] > 0 else 0.08
        g["v"] *= 0.9975 if not up else 0.9985
        if on_sand:
            g["v"] = min(g["v"], 3.2) * 0.96
        g["v"] = max(-2.5, min(VMAX, g["v"]))

        target = (right - left) * math.radians(28)  # steering wheel angle input
        g["steer"] += max(-0.05, min(0.05, target - g["steer"]))
        demand = g["v"] * math.tan(g["steer"]) / 24          # turn rate the wheels ask for
        cap = (A_GRIP * (0.55 if on_sand else 1.0)) / max(abs(g["v"]), 0.5)  # what the tyres can give
        g["omega"] = max(-cap, min(cap, demand))
        g["slide"] = abs(demand) > cap * 1.03 and abs(g["v"]) > 2
        g["a"] += g["omega"]

        ox, oy = g["x"], g["y"]
        g["x"] += math.cos(g["a"]) * g["v"]
        g["y"] += math.sin(g["a"]) * g["v"]
        hit = any(surface_at(tr, g["x"] + math.cos(g["a"]) * d, g["y"] + math.sin(g["a"]) * d) == "wall"
                  for d in (-16, 0, 16))
        if hit:  # barrier: bounce back and lose speed
            g["x"], g["y"] = ox, oy
            g["v"] = -g["v"] * 0.3
            g["flash"], g["hits"] = 70, g["hits"] + 1
        g["flash"] = max(0, g["flash"] - 1)

        # ---------- progress, laps, corners ----------
        best_i, bd = g["idx"], 1e18
        for j in range(-45, 46):
            i = (g["idx"] + j) % n
            d = (pts[i][0] - g["x"]) ** 2 + (pts[i][1] - g["y"]) ** 2
            if d < bd:
                bd, best_i = d, i
        di = best_i - g["idx"]
        di = di - n if di > n / 2 else di + n if di < -n / 2 else di
        g["cum"] += di
        g["idx"] = best_i
        if g["cum"] >= n:
            now = pygame.time.get_ticks()
            g["last"] = now - g["t0"]
            tr["best"] = g["last"] if tr["best"] is None else min(tr["best"], g["last"])
            g["laps"] += 1
            g["t0"] = now
            g["cum"] -= n

        kmh = abs(g["v"]) * KMH
        cur = next((c for c in corners if c["s"] <= g["idx"] <= c["e"]), None)
        if cur is not g["cur"]:
            if g["cur"] is not None and g["cur_min"] is not None:
                g["cur"]["last"] = g["cur_min"]
            g["cur"], g["cur_min"] = cur, None
        if cur is not None:
            g["cur_min"] = kmh if g["cur_min"] is None else min(g["cur_min"], kmh)
        nxt = next((c for c in corners if c["s"] > g["idx"]), corners[0])
        dist = ((nxt["s"] - g["idx"]) % n) * step

        # ---------- draw world ----------
        screen.blit(tr["surf"], (0, 0))
        rot = pygame.transform.rotate(car_img, -math.degrees(g["a"]))
        screen.blit(rot, rot.get_rect(center=(g["x"], g["y"])))
        if g["flash"]:
            t = f_b.render("BARRIER HIT", True, (255, 90, 90))
            screen.blit(t, t.get_rect(center=(W // 2, H // 2)))

        # ---------- HUD: lap info ----------
        panel(pygame.Rect(14, 12, 330, 84))
        text(tr["name"], (24, 18))
        text(f"LAP {g['laps'] + 1}    {fmt(pygame.time.get_ticks() - g['t0'])}", (24, 40), f_m)
        text(f"Last {fmt(g['last']) if g['last'] else '-'}    Best {fmt(tr['best']) if tr['best'] else '-'}", (24, 70))

        # ---------- HUD: corner data ----------
        panel(pygame.Rect(W - 354, 12, 340, 112))
        text(f"NEXT  T{nxt['n']}  {nxt['dir']}  {nxt['ang']:.0f} deg", (W - 344, 18), f_m)
        text(f"in {dist:.0f} m     target {nxt['target']:.0f} km/h", (W - 344, 48))
        if g["cur"] is not None:
            over = kmh > g["cur"]["target"] * 1.05
            text(f"IN T{g['cur']['n']}   speed {kmh:.0f} km/h   (target {g['cur']['target']:.0f})",
                 (W - 344, 74), f_s, (255, 110, 110) if over else (120, 235, 140))
        else:
            prev = next((c for c in reversed(corners) if c["last"] is not None), None)
            text(f"Last corner T{prev['n']}: {prev['last']:.0f} km/h" if prev else "Last corner: -",
                 (W - 344, 74))
        text(f"Barrier hits: {g['hits']}", (W - 344, 98))

        # ---------- HUD: speed, steering, G ----------
        panel(pygame.Rect(14, H - 166, 340, 152))
        over_target = g["cur"] is not None and kmh > g["cur"]["target"] * 1.05
        text(f"{kmh:.0f}", (26, H - 160), f_b, (255, 110, 110) if over_target else (255, 255, 255))
        text("km/h", (140, H - 128))
        if g["slide"]:
            text("SLIDING - lift off!", (200, H - 152), f_s, (255, 110, 110))
        text(f"Steering  {math.degrees(g['steer']):+.0f} deg", (26, H - 88))
        bar(190, H - 84, 150, 10, g["steer"] / math.radians(28), (240, 200, 60), True)
        text(f"Heading   {(math.degrees(g['a']) + 90) % 360:03.0f} deg", (26, H - 62))
        gforce = abs(g["v"] * g["omega"]) / (A_GRIP / 4)
        text(f"Lateral G {gforce:.1f}", (26, H - 38))
        bar(190, H - 34, 150, 10, gforce / 4, (255, 110, 110) if gforce > 3.6 else (120, 235, 140))

        # ---------- HUD: on-screen controls ----------
        panel(pygame.Rect(W - 274, H - 166, 260, 152))
        cx, cy = W - 240, H - 138
        key(cx + 42, cy, up, "u")
        key(cx, cy + 44, left, "l")
        key(cx + 42, cy + 44, down, "d")
        key(cx + 84, cy + 44, right, "r")
        text("UP  throttle", (W - 120, H - 150))
        text("DOWN  brake", (W - 120, H - 126))
        text("L/R  steer", (W - 120, H - 102))
        text("1-9 / TAB  circuit", (W - 264, H - 70))
        text("R  restart   ESC  quit", (W - 264, H - 44))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()