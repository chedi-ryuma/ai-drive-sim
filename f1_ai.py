"""F1 AI Trainer - 10 self-driving cars learn to race a full 56-lap Grand Prix (neuroevolution).

Every generation 10 cars race at once. Each has a tiny neural network that reads distance
sensors, its speed and its tyre wear, and outputs steering / throttle / brake.

RACE MODE (default)
  * 56 laps, like an F1 race. The first car over the line wins; the rest get 10 s to finish.
  * Tyres wear with distance driven. Worn tyres lose grip and top speed, so lap times creep up.
  * Every car must make 2 pit stops (Medium > Hard > Soft). WHEN to pit is chosen by the
    neural network: its 4th output is the "box now" decision, and it sees its tyre wear, race
    progress and stops made. (Panel option: Random or Fixed laps instead.) A finisher who made
    fewer than 2 stops gets +20 s per missing stop and a reward penalty. In the pit lane the car
    is limited to 80 km/h and stands still for the tyre change; it leaves on fresh tyres.
SPRINT MODE: the old 50-second generations, no tyres, no pit stops (fast early training).

GUI
  Drop-downs: training method, obstacles, barrier look, circuit, mode.
  [Reward]  button: edit the reward function live and choose who decides the pit stops.
  [Weights] button: see the neural-network weights of every car as heat maps.
  Click the road to place a barrier, right-click one to remove it.

Keep this file in the same folder as f1_racer.py and f1_tracks.py.

Keys:  SPACE = speed (1x / 4x / 12x)   N = end this generation now
       1..9 = circuit (brain is kept)   R = start again from random brains
       ESC = close the open panel / quit.
       The best brain is auto-saved to best_brain.json (older brains are upgraded automatically).
       Reward / pit settings are saved to reward.json.

Setup:  pip install pygame
Run:    python f1_ai.py
"""
import json
import math
import os
import random
import sys

import pygame

import f1_racer as F

N_CARS = 10
RAYS = [-75, -45, -20, 0, 20, 45, 75]   # sensor angles (deg) relative to the car
RAY_LEN, RAY_STEP = 200, 10
# inputs: 7 sensors + speed + tyre wear + race progress + stops made; outputs: steer, throttle, brake, PIT
N_IN, N_HID, N_OUT = len(RAYS) + 4, 10, 4
NW = N_HID * (N_IN + 1) + N_OUT * (N_HID + 1)
OLD_SHAPES = {123: (8, 3), 133: (9, 3)}      # brain sizes saved by older versions -> (inputs, outputs)
SPRINT_FRAMES = 60 * 50   # sprint generation: at most 50 s
STALL = 150               # frames without progress before a car is retired
SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]   # mutation strength of the 9 children
SAVE = "best_brain.json"
SETTINGS_FILE = "reward.json"
METHODS = [
    ("Best parent + mutation", "Best car kept, 9 mutated copies of it"),
    ("Top 3 + crossover (GA)", "Mix genes of the 3 best, then mutate"),
    ("Tournament selection", "3-car contests pick parents; mix + mutate"),
    ("Evolution strategy (ES)", "Centre brain moves toward better noise"),
    ("Random search", "Best car kept, 9 new random brains"),
]
MODES = ["Race", "Sprint"]
OBSTACLES = [("No obstacles", 0), ("Few obstacles (6)", 6), ("Many obstacles (14)", 14)]
OBST_R = 10
BARRIERS = ["Mixed", "Tyre stack", "Concrete block", "Cone cluster"]
PALETTE = [(255, 210, 40), (80, 160, 255), (90, 220, 120), (230, 90, 220), (255, 140, 60),
           (80, 230, 230), (240, 240, 240), (170, 120, 255), (255, 120, 150), (170, 230, 60)]

# ---------------- reward (fitness) function: editable from the GUI ----------------
# score = progress*(samples driven) - sand*(frames on sand) + dist*(px driven)
#         + speed*(avg speed share)*lap - crash*lap (if crashed) + finish*lap*(time saved) , in samples
REWARD_DEFAULT = {"progress": 1.0, "sand": 0.3, "dist": 0.002, "speed": 0.0, "crash": 0.0, "finish": 1.0,
                  "stops": 1.0}
REWARD = dict(REWARD_DEFAULT)
PIT_MODES = ["ai", "random", "fixed"]
PIT_MODE_NAMES = {"ai": "AI chooses", "random": "Random", "fixed": "Fixed laps"}
SETTINGS = {"pit_mode": "ai", "team_call": True}
MANDATORY_STOPS = 2
PEN_PER_STOP_S = 20.0                 # race-time penalty for every mandatory stop not made
REWARD_ROWS = [   # key, label, help text, step, min, max, format
    ("progress", "Progress", "reward per track sample driven (1.0 = one lap is worth one lap)", 0.25, 0.0, 5.0, "{:.2f}"),
    ("sand", "Sand penalty", "penalty for every frame spent on sand / run-off", 0.1, 0.0, 5.0, "{:.1f}"),
    ("dist", "Distance bonus", "tiny bonus per pixel driven (any driving beats none)", 0.001, 0.0, 0.05, "{:.3f}"),
    ("speed", "Speed bonus", "laps-worth of reward for a high average speed", 0.5, 0.0, 10.0, "{:.1f}"),
    ("crash", "Crash penalty", "laps lost when a car hits a barrier", 0.25, 0.0, 5.0, "{:.2f}"),
    ("finish", "Finish bonus", "laps-worth of reward for finishing the race sooner", 0.5, 0.0, 10.0, "{:.1f}"),
    ("stops", "Missed pit stop", "laps lost per mandatory stop not made (finishers only)", 0.5, 0.0, 10.0, "{:.1f}"),
]


def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            d = json.load(f)
        for k, v in d.get("reward", {}).items():
            if k in REWARD and isinstance(v, (int, float)):
                REWARD[k] = float(v)
        if d.get("pit_mode") in PIT_MODES:
            SETTINGS["pit_mode"] = d["pit_mode"]
        elif "random_pits" in d:                       # file written by the previous version
            SETTINGS["pit_mode"] = "random" if d["random_pits"] else "fixed"
        SETTINGS["team_call"] = bool(d.get("team_call", True))
    except (OSError, ValueError, AttributeError):
        pass


def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"reward": REWARD, "pit_mode": SETTINGS["pit_mode"], "team_call": SETTINGS["team_call"]}, f, indent=1)
    except OSError:
        pass


def reward_parts(c, n):
    """Score broken down into its terms (in track samples; divide by n for laps)."""
    R = REWARD
    return {
        "progress": R["progress"] * (c["maxcum"] + 12),
        "sand": -R["sand"] * c["sand"],
        "dist": R["dist"] * c["dist"],
        "speed": R["speed"] * n * (c["dist"] / c["frames"] / F.VMAX) if c["frames"] else 0.0,
        "crash": -R["crash"] * n if c["why"] == "crashed" else 0.0,
        "finish": (R["finish"] * n * max(0.0, 1 - (c["finish_s"] if c["finish_s"] is not None else c["frames"] / 60)
                                          * 60 / RACE_MAX_FRAMES) if c["finished"] else 0.0),
        "stops": -R["stops"] * n * max(0, MANDATORY_STOPS - c["stops"]) if c["finished"] else 0.0,
    }


def reward(c, n):
    return sum(reward_parts(c, n).values())


# ---------------- race, tyres and pit stops ----------------
RACE_LAPS = 56
RACE_MAX_FRAMES = 60 * 60 * 12       # safety cap: 12 minutes of race time
FINISH_GRACE = 60 * 10               # after the winner finishes, the others get 10 s
TYRES = {                            # grip / pace = performance when new, life = laps until fully worn
    "S": {"name": "Soft",   "col": (235, 60, 60),   "grip": 1.05, "pace": 0.020, "life": 16},
    "M": {"name": "Medium", "col": (245, 205, 50),  "grip": 1.00, "pace": 0.000, "life": 24},
    "H": {"name": "Hard",   "col": (235, 235, 235), "grip": 0.96, "pace": -0.015, "life": 36},
}
STRATEGY = ["M", "H", "S"]           # 2 pit stops = 3 stints (F1 needs two different compounds)
_lives = [TYRES[t]["life"] for t in STRATEGY]
PIT_LAPS, _acc = [], 0.0             # fixed plan: stint lengths proportional to tyre life -> laps 18 and 44
for _l in _lives[:-1]:
    _acc += RACE_LAPS * _l / sum(_lives)
    PIT_LAPS.append(int(round(_acc)))
PIT_OFF = 52                         # pit lane distance from the track centre line (px)
PIT_ENTER, PIT_BOX, PIT_EXIT = 14, 18, 32   # in track samples: entry before the line, garage box, exit
PIT_SPEED = 2.0                      # pit-lane limit: 2 px/frame = 80 km/h
PIT_STOP_FRAMES = 50                 # ~0.8 s standing still for the tyre change


def random_pit_laps():
    """Random but sensible: no tyre is driven far beyond its life (M<=24, H<=36, S<=16 laps)."""
    p1 = random.randint(10, 24)
    lo, hi = max(p1 + 8, RACE_LAPS - 16), min(p1 + 36, RACE_LAPS - 8)
    return [p1, random.randint(lo, hi)]


def pit_text(c):
    """Short description of a car's stops for the HUD."""
    if SETTINGS["pit_mode"] == "ai":
        return "L" + "/".join(str(x) for x in c["stop_laps"]) if c["stop_laps"] else "AI"
    return "L" + "/".join(str(x) for x in c["pit_laps"])


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def pit_offset(u):
    return PIT_OFF * min(smooth(u / 8), 1 - smooth((u - (PIT_EXIT - 8)) / 8))


def pit_pos(tr, u, extra=0.0):
    """Position and heading on the pit-lane path; u = track samples since the pit entry."""
    n = tr["n"]
    p = n - PIT_ENTER + u
    i = int(p // 1)
    f = p - i
    a, b = tr["pts"][i % n], tr["pts"][(i + 1) % n]
    h = tr["head"][i % n]
    off = pit_offset(u) + extra
    return (a[0] + (b[0] - a[0]) * f - math.sin(h) * off,
            a[1] + (b[1] - a[1]) * f + math.cos(h) * off, h)


def tyre_factors(c, race):
    """(grip multiplier, top-speed multiplier) from compound and wear."""
    if not race:
        return 1.0, 1.0
    t, wear = TYRES[c["comp"]], min(c["wear"], 1.3)
    deg = 1 - 0.22 * wear ** 1.4          # worn tyres lose up to ~1/3 of their grip
    return t["grip"] * deg, (1 + t["pace"]) * (1 - 0.07 * wear)


def fmt_time(s):
    if s is None:
        return "--:--.-"
    return f"{int(s // 60)}:{s % 60:04.1f}"


def draw_barrier(world, x, y, kind):
    x, y = int(x), int(y)
    if kind == 0:        # tyre stack: black / white rings
        for r, col in ((OBST_R, (20, 20, 20)), (OBST_R - 3, (235, 235, 235)),
                       (OBST_R - 5, (20, 20, 20)), (3, (70, 70, 70))):
            pygame.draw.circle(world, col, (x, y), r)
    elif kind == 1:      # concrete block with hazard stripes
        pygame.draw.rect(world, (150, 150, 155), (x - 10, y - 10, 20, 20))
        for k in range(-10, 11, 7):
            pygame.draw.line(world, (240, 200, 30), (x + k, y - 10), (x + k + 7, y + 10), 3)
        pygame.draw.rect(world, (60, 60, 65), (x - 10, y - 10, 20, 20), 2)
    else:                # cone cluster
        for dx, dy in ((-5, -4), (5, -3), (0, 5)):
            pygame.draw.circle(world, (255, 120, 20), (x + dx, y + dy), 5)
            pygame.draw.circle(world, (245, 245, 245), (x + dx, y + dy), 2)


def draw_pit_lane(world, tr):
    """Pit lane next to the start/finish straight (decoration only - the cars follow it by script)."""
    path = [pit_pos(tr, u / 2)[:2] for u in range(0, 2 * PIT_EXIT + 1)]
    for x, y in path:
        pygame.draw.circle(world, (96, 98, 108), (int(x), int(y)), 12)
    for x, y in path:
        pygame.draw.circle(world, (66, 68, 78), (int(x), int(y)), 10)
    cx, cy, h = pit_pos(tr, PIT_BOX, 24)           # the garage / pit box
    tx, ty, nx, ny = math.cos(h), math.sin(h), -math.sin(h), math.cos(h)
    corners = [(cx + tx * a + nx * b, cy + ty * a + ny * b) for a, b in ((-12, -9), (12, -9), (12, 9), (-12, 9))]
    pygame.draw.polygon(world, (50, 54, 66), corners)
    pygame.draw.polygon(world, (245, 205, 50), corners, 2)


# ---------------- neural network ----------------
def forward(w, inp):
    hid = []
    for h in range(N_HID):
        o = h * (N_IN + 1)
        hid.append(math.tanh(sum(w[o + i] * inp[i] for i in range(N_IN)) + w[o + N_IN]))
    base = N_HID * (N_IN + 1)
    out = []
    for k in range(N_OUT):
        o = base + k * (N_HID + 1)
        out.append(math.tanh(sum(w[o + i] * hid[i] for i in range(N_HID)) + w[o + N_HID]))
    return out   # steer, throttle, brake


def mutate(w, sigma, rate=0.2):
    return [x + random.gauss(0, sigma) if random.random() < rate else x for x in w]


# ---------------- cars ----------------
def new_car(tr, w, k, tag="", elite=False, eps=None):
    i0 = tr["n"] - 12
    return {"id": k, "w": w, "elite": elite, "tag": tag, "eps": eps, "x": tr["pts"][i0][0], "y": tr["pts"][i0][1],
            "a": tr["head"][i0], "v": 0.0, "steer": 0.0, "idx": i0, "cum": -12, "maxcum": -12,
            "alive": True, "dist": 0.0, "frames": 0, "still": 0, "sand": 0, "fit": 0.0, "laps": 0,
            "lap_t": 0, "best_lap": None, "why": "",
            "wear": 0.0, "comp": STRATEGY[0], "stops": 0, "pit": None, "lts": [], "pit_marks": [],
            "pit_laps": list(PIT_LAPS), "pit_req": False, "stop_laps": [], "pen_s": 0.0,
            "finished": False, "finish_s": None}


def sense(tr, c):
    out = []
    for d in RAYS:
        ang = c["a"] + math.radians(d)
        dx, dy = math.cos(ang), math.sin(ang)
        dist = RAY_LEN
        for r in range(RAY_STEP, RAY_LEN + 1, RAY_STEP):
            if F.surface_at(tr, c["x"] + dx * r, c["y"] + dy * r) != "road":
                dist = r
                break
        out.append(dist / RAY_LEN)
    return out


def drive(tr, c, race):
    """Same physics as the human game, but the network drives. Tyres change grip and top speed."""
    grip, vm = tyre_factors(c, race)
    vmax = F.VMAX * vm
    wear_in = min(c["wear"], 1.0) if race else 0.0
    prog_in = min(c["laps"] / RACE_LAPS, 1.0) if race else 0.0
    stops_in = c["stops"] / MANDATORY_STOPS if race else 0.0
    out = forward(c["w"], sense(tr, c) + [c["v"] / F.VMAX, wear_in, prog_in, stops_in])
    if race and SETTINGS["pit_mode"] == "ai" and out[3] > 0.5 and c["stops"] < MANDATORY_STOPS:
        c["pit_req"] = True                       # the network says "box, box": pit at the next entry
    thr, brk = max(0.0, out[1]), max(0.0, out[2])
    v = c["v"]
    sand = F.surface_at(tr, c["x"], c["y"]) == "sand"
    if thr > 0:
        v += 0.12 * thr * (1 - max(v, 0) / (vmax + 0.5))
    if brk > 0:
        v -= (0.28 if v > 0 else 0.08) * brk
    v *= 0.9985 if thr > 0 else 0.9975
    if sand:
        v = min(v, 3.2) * 0.96
        c["sand"] += 1
    v = max(-2.5, min(vmax, v))
    c["v"] = v
    c["steer"] += max(-0.05, min(0.05, out[0] * math.radians(28) - c["steer"]))
    demand = v * math.tan(c["steer"]) / 24
    cap = F.A_GRIP * grip * (0.55 if sand else 1.0) / max(abs(v), 0.5)
    c["a"] += max(-cap, min(cap, demand))
    c["x"] += math.cos(c["a"]) * v
    c["y"] += math.sin(c["a"]) * v
    c["dist"] += abs(v)
    if race:   # tyres wear with distance: fully worn after `life` laps
        c["wear"] += abs(v) / (tr["n"] * tr["step"]) / TYRES[c["comp"]]["life"]
    if any(F.surface_at(tr, c["x"] + math.cos(c["a"]) * d, c["y"] + math.sin(c["a"]) * d) == "wall"
           for d in (-16, 0, 16)):
        c["alive"], c["why"] = False, "crashed"


def pit_drive(tr, c):
    """Scripted pit stop: slow to the pit limit, stop in the box, change tyres, leave."""
    p, v = c["pit"], c["v"]
    if p["ph"] == "in":
        v = max(PIT_SPEED, v - 0.35)
        p["u"] += v / tr["step"]
        if p["u"] >= PIT_BOX:
            p["u"], v, p["ph"], p["t"] = PIT_BOX, 0.0, "stop", 0
    elif p["ph"] == "stop":
        v = 0.0
        p["t"] += 1
        if p["t"] >= PIT_STOP_FRAMES:      # new tyres!
            c["stops"] += 1
            c["comp"] = STRATEGY[min(c["stops"], len(STRATEGY) - 1)]
            c["wear"], c["pit_req"] = 0.0, False       # fresh tyres, the request is done
            c["pit_marks"].append(len(c["lts"]))
            p["ph"] = "out"
    else:
        v = min(PIT_SPEED, v + 0.15)
        p["u"] += v / tr["step"]
        if p["u"] >= PIT_EXIT:
            c["pit"] = None
    x, y, h = pit_pos(tr, min(p["u"], PIT_EXIT))
    c["x"], c["y"], c["a"], c["v"], c["steer"] = x, y, h, v, 0.0


def progress(tr, c, race):
    pts, n = tr["pts"], tr["n"]
    best_i, bd = c["idx"], 1e18
    for j in range(-45, 46):
        i = (c["idx"] + j) % n
        d = (pts[i][0] - c["x"]) ** 2 + (pts[i][1] - c["y"]) ** 2
        if d < bd:
            bd, best_i = d, i
    di = best_i - c["idx"]
    di = di - n if di > n / 2 else di + n if di < -n / 2 else di
    c["idx"] = best_i
    c["cum"] += di
    if c["cum"] > c["maxcum"]:
        c["maxcum"], c["still"] = c["cum"], 0
    else:
        c["still"] += 1
    c["frames"] += 1
    laps = max(0, c["maxcum"] // n)
    if laps > c["laps"]:
        c["laps"] = laps
        t = (c["frames"] - c["lap_t"]) / 60
        c["best_lap"] = t if c["best_lap"] is None else min(c["best_lap"], t)
        c["lts"].append(t)
        c["lap_t"] = c["frames"]
        if race and laps >= RACE_LAPS and not c["finished"]:   # chequered flag
            c["finished"], c["alive"], c["why"] = True, False, "finished"
            c["pen_s"] = PEN_PER_STOP_S * max(0, MANDATORY_STOPS - c["stops"])   # F1-style time penalty
            c["finish_s"], c["v"] = c["frames"] / 60 + c["pen_s"], 0.0


def step(tr, c, race):
    if not c["alive"]:
        return
    n = tr["n"]
    mode = SETTINGS["pit_mode"]
    if (race and mode == "ai" and SETTINGS["team_call"] and c["pit"] is None and c["wear"] >= 1.0
            and c["stops"] < MANDATORY_STOPS):
        c["pit_req"] = True                       # the pit wall calls a car in whose tyres are dead
    if race and c["pit"] is None and c["stops"] < MANDATORY_STOPS and c["idx"] >= n - PIT_ENTER:
        if mode == "ai":
            go = c["pit_req"] and c["laps"] >= 1
        else:
            pl = c["pit_laps"]
            go = c["stops"] < len(pl) and c["laps"] == pl[c["stops"]] - 1
        if go:
            c["pit"] = {"u": float(c["idx"] - (n - PIT_ENTER)), "ph": "in", "t": 0}
            c["pit_req"] = False
            c["stop_laps"].append(c["laps"] + 1)
    if c["pit"] is not None:
        pit_drive(tr, c)
    else:
        drive(tr, c, race)
    progress(tr, c, race)
    if c["pit"] is not None:
        c["still"] = 0
    if c["alive"] and c["still"] > STALL:
        c["alive"], c["why"] = False, "stuck"


def rand_w():
    return [random.gauss(0, 0.8) for _ in range(NW)]


def cross(a, b):
    return [x if random.random() < 0.5 else y for x, y in zip(a, b)]


def make_obstacles(tr, count, ti):
    """Place obstacles [x, y, kind] along the track, alternating sides, never on the start grid / pit lane."""
    rng = random.Random(1000 + ti * 17 + count)
    n, out = tr["n"], []
    for k in range(count):
        i = int((k + 0.5) * n / count + rng.uniform(-8, 8)) % n
        if (i - (n - 16)) % n < 55:
            i = (i + 70) % n
        h, off = tr["head"][i], (1 if k % 2 == 0 else -1) * rng.uniform(13, 21)
        out.append([tr["pts"][i][0] - math.sin(h) * off, tr["pts"][i][1] + math.cos(h) * off, k % 3])
    return out


def reshape(w, old_in, old_out):
    """Upgrade a brain saved by an older version: new inputs / outputs start with weight 0."""
    out, stride = [], old_in + 1
    for h in range(N_HID):
        blk = w[h * stride:(h + 1) * stride]
        out += blk[:old_in] + [0.0] * (N_IN - old_in) + [blk[old_in]]
    base = N_HID * stride
    for k in range(N_OUT):
        out += w[base + k * (N_HID + 1):base + (k + 1) * (N_HID + 1)] if k < old_out else [0.0] * (N_HID + 1)
    return out


def load_brain():
    try:
        with open(SAVE) as f:
            w = json.load(f)
    except (OSError, ValueError):
        return None
    if len(w) == NW:
        return w
    if len(w) in OLD_SHAPES:
        try:
            name = f"best_brain_backup_{len(w)}.json"
            if not os.path.exists(name):               # keep a copy of the old brain
                with open(name, "w") as f:
                    json.dump(w, f)
        except OSError:
            pass
        return reshape(w, *OLD_SHAPES[len(w)])
    return None


def save_brain(w):
    try:
        with open(SAVE, "w") as f:
            json.dump(w, f)
    except OSError:
        pass


# ---------------- training state ----------------
class Trainer:
    def __init__(self, cache, ti=0, parent=None, gen=1, msg="", method=0, obst=0, manual=None, race=True):
        self.cache, self.ti, self.parent = cache, ti, parent
        self.gen, self.history, self.best_lap, self.msg = gen, [], None, msg
        self.method, self.obst, self.race = method, obst, race
        self.max_frames = RACE_MAX_FRAMES if race else SPRINT_FRAMES
        self.manual = manual if manual is not None else {}
        self.center, self.weak = parent, False
        if ti not in cache:
            cache[ti] = F.build(*F.TRACKS[ti])
        self.tr = cache[ti]
        self.auto = make_obstacles(self.tr, OBSTACLES[obst][1], ti)
        self.rebuild_view()
        if parent is None:
            self.plan = [(rand_w(), "random", False, None) for _ in range(N_CARS)]
        else:
            self.plan = [(list(parent), "parent", True, None)] + [
                (mutate(parent, SIGMAS[k - 1]), f"mut {SIGMAS[k - 1]:.2f}", False, None)
                for k in range(1, N_CARS)]
        self.start_gen()

    # obstacles are painted onto a copy of the track image, so sensors and
    # collisions treat them exactly like barriers
    def obstacles(self):
        return self.auto + self.manual.setdefault(self.ti, [])

    def rebuild_view(self):
        surf = self.tr["surf"].copy()
        for x, y, _ in self.obstacles():
            pygame.draw.circle(surf, F.WALL_W, (int(x), int(y)), OBST_R)
            pygame.draw.circle(surf, F.WALL_R, (int(x), int(y)), OBST_R - 3)
        self.view = dict(self.tr)
        self.view["surf"] = surf

    def start_gen(self):
        self.cars = [new_car(self.view, w, k, tag, el, eps) for k, (w, tag, el, eps) in enumerate(self.plan)]
        if self.race:
            for c in self.cars:
                if SETTINGS["pit_mode"] == "random":   # every car gets its own random pit laps
                    c["pit_laps"] = random_pit_laps()
                elif SETTINGS["pit_mode"] == "ai":     # the network decides, no plan
                    c["pit_laps"] = []
        for c in self.cars:
            c["fit"] = reward(c, self.tr["n"])
        self.t = 0
        self.grace = None

    def obstacles_changed(self):
        self.rebuild_view()
        self.start_gen()       # restart this generation with the new layout

    def set_obstacle_mode(self, mode):
        self.obst = mode
        self.auto = make_obstacles(self.tr, OBSTACLES[mode][1], self.ti)
        self.obstacles_changed()

    def click(self, pos, add, kind=0):
        lst = self.obstacles()
        if add:
            if F.surface_at(self.view, *pos) == "road" and math.hypot(
                    pos[0] - self.tr["pts"][0][0], pos[1] - self.tr["pts"][0][1]) > 90 \
                    and all(math.hypot(pos[0] - x, pos[1] - y) > 2 * OBST_R for x, y, _ in lst):
                self.manual[self.ti].append([float(pos[0]), float(pos[1]), kind])
                self.obstacles_changed()
        elif lst:
            o = min(lst, key=lambda p: math.hypot(pos[0] - p[0], pos[1] - p[1]))
            if math.hypot(pos[0] - o[0], pos[1] - o[1]) < 18:
                (self.auto if o in self.auto else self.manual[self.ti]).remove(o)
                self.obstacles_changed()

    # ----- the training methods: build the plan for the next generation -----
    def breed(self):
        ranked = sorted(self.cars, key=lambda c: -c["fit"])
        best, m, out = ranked[0], self.method, []
        keep = (list(best["w"]), "parent", True, None)
        if m == 0:      # best parent + mutation
            out = [keep] + [(mutate(best["w"], SIGMAS[k - 1]), f"mut {SIGMAS[k - 1]:.2f}", False, None)
                            for k in range(1, N_CARS)]
        elif m == 1:    # top 3 + crossover
            top = ranked[:3]
            out = [keep]
            for _ in range(1, N_CARS):
                a, b = random.sample(top, 2)
                out.append((mutate(cross(a["w"], b["w"]), random.choice([0.05, 0.1, 0.2, 0.3]), 0.1),
                            "cross", False, None))
        elif m == 2:    # tournament selection
            def pick():
                return max(random.sample(self.cars, 3), key=lambda c: c["fit"])
            out = [keep]
            for _ in range(1, N_CARS):
                out.append((mutate(cross(pick()["w"], pick()["w"]), random.choice([0.05, 0.1, 0.2]), 0.1),
                            "tourn.", False, None))
        elif m == 3:    # evolution strategy: move the centre along the good noise
            sigma = 0.25
            samples = sorted([c for c in self.cars if c["eps"] is not None], key=lambda c: c["fit"])
            if len(samples) >= 3 and self.center is not None:
                cnt, delta = len(samples), [0.0] * NW
                for r, c in enumerate(samples):
                    u = r / (cnt - 1) - 0.5
                    for j in range(NW):
                        delta[j] += u * c["eps"][j]
                self.center = [self.center[j] + sigma * 4 / cnt * delta[j] for j in range(NW)]
            else:
                self.center = list(best["w"])
            out = [(list(self.center), "centre", True, None)]
            for _ in range(1, N_CARS):
                eps = [random.gauss(0, 1) for _ in range(NW)]
                out.append(([self.center[j] + sigma * eps[j] for j in range(NW)], "noise", False, eps))
        else:           # random search
            out = [keep] + [(rand_w(), "random", False, None) for _ in range(1, N_CARS)]
        if self.weak and m != 4:   # parent barely drives: add newcomers to explore
            for k in range(N_CARS - 4, N_CARS):
                out[k] = (rand_w(), "random", False, None)
        return out

    def end_gen(self):
        n = self.tr["n"]
        for c in self.cars:
            c["fit"] = reward(c, n)
        best = max(self.cars, key=lambda c: c["fit"])
        self.parent = list(best["w"])
        self.weak = best["fit"] < 0.1 * n
        self.history.append(best["fit"] / n)
        for c in self.cars:
            if c["best_lap"] and (self.best_lap is None or c["best_lap"] < self.best_lap):
                self.best_lap = c["best_lap"]
        done = [c for c in self.cars if c["finished"]]
        if self.race and done:     # the winner is whoever crossed the line first (not necessarily `best`)
            win = min(done, key=lambda c: c["finish_s"])
            self.msg = (f"Gen {self.gen}: car #{win['id'] + 1} won the {RACE_LAPS}-lap race in "
                        f"{fmt_time(win['finish_s'])} ({win['stops']} stops, {len(done)} finished); "
                        f"best score: car #{best['id'] + 1} -> {METHODS[self.method][0]}")
        else:
            self.msg = (f"Gen {self.gen}: car #{best['id'] + 1} was best "
                        f"({best['fit'] / n:.2f} laps) -> {METHODS[self.method][0]}")
        save_brain(self.parent)
        self.plan = self.breed()
        self.gen += 1
        self.start_gen()

    def update(self):
        n = self.tr["n"]
        for c in self.cars:
            step(self.view, c, self.race)
            c["fit"] = reward(c, n)            # live: changing the reward in the GUI applies at once
        self.t += 1
        if self.race and self.grace is None and any(c["finished"] for c in self.cars):
            self.grace = FINISH_GRACE          # winner is home; the others have 10 s left
        if self.grace is not None:
            self.grace -= 1
        if (self.t >= self.max_frames or not any(c["alive"] for c in self.cars)
                or (self.grace is not None and self.grace <= 0)):
            self.end_gen()


class Dropdown:
    """Drop-down list. Long lists show `max_vis` rows and scroll with the mouse wheel."""

    def __init__(self, rect, options, sel=0, max_vis=12):
        self.rect, self.options, self.sel, self.open = pygame.Rect(rect), options, sel, False
        self.max_vis, self.top = max_vis, 0

    @property
    def vis(self):
        return min(len(self.options), self.max_vis)

    def item(self, i):
        return pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.h, self.rect.w, self.rect.h)

    def scroll(self, d):
        self.top = max(0, min(len(self.options) - self.vis, self.top + d))

    def show(self):
        self.open = True
        self.top = max(0, min(self.sel - self.vis // 2, len(self.options) - self.vis))

    def click(self, pos):
        """Returns (consumed, newly selected index or None)."""
        if self.open:
            for i in range(self.vis):
                if self.item(i).collidepoint(pos):
                    self.open = False
                    idx = self.top + i
                    changed = idx != self.sel
                    self.sel = idx
                    return True, (idx if changed else None)
            self.open = False
            return self.rect.collidepoint(pos), None
        if self.rect.collidepoint(pos):
            self.show()
            return True, None
        return False, None

    def draw(self, screen, font):
        def box(r, label, hot):
            pygame.draw.rect(screen, (60, 66, 82) if hot else (34, 38, 48), r, border_radius=5)
            pygame.draw.rect(screen, (150, 160, 180), r, 1, border_radius=5)
            screen.blit(font.render(label, True, (245, 245, 245)), (r.x + 8, r.y + 6))
        box(self.rect, self.options[self.sel], self.open)
        cx, cy = self.rect.right - 14, self.rect.centery
        pygame.draw.polygon(screen, (230, 230, 230), [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)])
        if self.open:
            for i in range(self.vis):
                box(self.item(i), self.options[self.top + i], self.top + i == self.sel)
            if len(self.options) > self.vis:   # scroll thumb
                track = pygame.Rect(self.rect.right - 5, self.rect.bottom, 4, self.vis * self.rect.h)
                th = max(16, track.h * self.vis // len(self.options))
                ty = track.y + (track.h - th) * self.top // (len(self.options) - self.vis)
                pygame.draw.rect(screen, (110, 118, 140), (track.x, ty, track.w, th), border_radius=2)


def track_name(i):
    """Display name of circuit i, taken from F.TRACKS when the entry starts with a string."""
    t = F.TRACKS[i]
    name = t[0] if isinstance(t, (tuple, list)) and t and isinstance(t[0], str) else f"Circuit {i + 1}"
    return name.split(" (")[0]


def weight_color(v):
    """Heat-map colour of one weight: red = positive, blue = negative, brighter = larger."""
    t = math.tanh(v / 1.2)
    if t >= 0:
        return (int(45 + 210 * t), int(45 + 30 * t), 45)
    return (45, int(45 - 60 * t), int(45 - 210 * t))


SCALE, WX, WY = 0.8, 0, 70       # the circuit is shown at 80% in its own area, HUD around it
VIEW_W, VIEW_H = int(F.W * SCALE), int(F.H * SCALE)
COL_X = VIEW_W + 10               # right-hand column
PANEL = pygame.Rect(WX + 16, WY + 12, VIEW_W - 32, VIEW_H - 24)   # overlay for Reward / Weights


def reward_layout():
    """Hit boxes of the Reward panel (computed from PANEL so drawing and clicking always agree)."""
    rows, y = [], PANEL.y + 68
    for r in REWARD_ROWS:
        rows.append({"key": r[0], "y": y,
                     "minus": pygame.Rect(PANEL.right - 170, y + 3, 40, 30),
                     "plus": pygame.Rect(PANEL.right - 60, y + 3, 40, 30)})
        y += 39
    return {"rows": rows,
            "pits": pygame.Rect(PANEL.right - 170, y + 3, 150, 30), "pits_y": y,
            "team": pygame.Rect(PANEL.right - 120, y + 42, 100, 30), "team_y": y + 39,
            "reset": pygame.Rect(PANEL.x + 20, PANEL.bottom - 44, 150, 32),
            "close": pygame.Rect(PANEL.right - 120, PANEL.bottom - 44, 100, 32)}


def main():
    pygame.init()
    screen = pygame.display.set_mode((F.W, F.H))
    pygame.display.set_caption("F1 AI Trainer - 56-lap race")
    clock = pygame.time.Clock()
    f_xs, f_s, f_m = (pygame.font.SysFont(None, s) for s in (20, 22, 30))
    world = pygame.Surface((F.W, F.H))
    load_settings()

    cache, imgs = {}, {}

    def car_img(color):
        if color not in imgs:
            s = pygame.Surface((38, 18), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (4, 3, 30, 12), border_radius=4)
            pygame.draw.rect(s, color, (0, 0, 6, 18))
            pygame.draw.rect(s, color, (32, 1, 6, 16))
            pygame.draw.rect(s, (25, 25, 30), (17, 6, 7, 6), border_radius=2)
            imgs[color] = s
        return imgs[color]

    loaded = load_brain()
    T = Trainer(cache, 0, loaded, 1,
                "Loaded saved brain from best_brain.json" if loaded else "Starting from random brains")
    speeds, si = (1, 4, 12), 0
    dd_method = Dropdown((10, 32, 230, 28), [m[0] for m in METHODS], 0)
    dd_obst = Dropdown((246, 32, 108, 28), ["None", "Few (6)", "Many (14)"], 0)
    dd_barrier = Dropdown((360, 32, 150, 28), BARRIERS, 0)
    dd_track = Dropdown((516, 32, 190, 28), [track_name(i) for i in range(len(F.TRACKS))], 0)
    dd_mode = Dropdown((712, 32, 80, 28), MODES, 0)
    dds = [dd_method, dd_obst, dd_barrier, dd_track, dd_mode]
    btn_reward, btn_weights = pygame.Rect(800, 8, 80, 24), pygame.Rect(800, 36, 80, 24)
    ui = {"panel": None}              # None / "reward" / "weights"
    BG, EDGE = (18, 20, 27), (70, 76, 92)

    def text(t, pos, font=f_s, col=(235, 235, 235)):
        screen.blit(font.render(t, True, col), pos)

    def button(rect, label, active=False, font=f_xs):
        pygame.draw.rect(screen, (70, 96, 150) if active else (44, 50, 66), rect, border_radius=5)
        pygame.draw.rect(screen, (150, 160, 180), rect, 1, border_radius=5)
        img = font.render(label, True, (245, 245, 245))
        screen.blit(img, img.get_rect(center=rect.center))

    def new_trainer(ti, parent, gen, msg):
        return Trainer(cache, ti, parent, gen, msg, dd_method.sel, dd_obst.sel, T.manual, dd_mode.sel == 0)

    def reward_click(pos):
        lay = reward_layout()
        for row, spec in zip(lay["rows"], REWARD_ROWS):
            key, label, _, stp, lo, hi, _f = spec
            for rect, sign in ((row["minus"], -1), (row["plus"], 1)):
                if rect.collidepoint(pos):
                    REWARD[key] = round(min(hi, max(lo, REWARD[key] + sign * stp)), 4)
                    save_settings()
                    T.msg = f"Reward: {label} = {REWARD[key]:g} (applies live)"
        if lay["pits"].collidepoint(pos):          # cycle: AI chooses -> random -> fixed laps
            SETTINGS["pit_mode"] = PIT_MODES[(PIT_MODES.index(SETTINGS["pit_mode"]) + 1) % len(PIT_MODES)]
            save_settings()
            T.msg = {"ai": "Pit stops: the neural network decides when to pit (4th output)",
                     "random": "Pit stops: random laps for every car, every generation",
                     "fixed": f"Pit stops: everyone pits on laps {PIT_LAPS[0]} and {PIT_LAPS[1]}"}[SETTINGS["pit_mode"]]
        elif lay["team"].collidepoint(pos):
            SETTINGS["team_call"] = not SETTINGS["team_call"]
            save_settings()
            T.msg = ("Team call ON: the pit wall calls a car in when its tyres are dead" if SETTINGS["team_call"]
                     else "Team call OFF: the AI must pit on its own or finish with dead tyres / a penalty")
        elif lay["reset"].collidepoint(pos):
            REWARD.update(REWARD_DEFAULT)
            save_settings()
            T.msg = "Reward reset to defaults"
        elif lay["close"].collidepoint(pos):
            ui["panel"] = None

    def draw_panel(lead):
        if ui["panel"] is None:
            return
        bg = pygame.Surface(PANEL.size, pygame.SRCALPHA)
        bg.fill((12, 14, 20, 240))
        screen.blit(bg, PANEL.topleft)
        pygame.draw.rect(screen, (150, 160, 180), PANEL, 1, border_radius=6)
        n = T.tr["n"]
        if ui["panel"] == "reward":
            text("REWARD FUNCTION  (changes apply live; saved to reward.json)", (PANEL.x + 20, PANEL.y + 12), f_m)
            text("score = progress x samples - sand x frames on sand + dist x px + speed x avg speed"
                 " - crash + finish - missed stops", (PANEL.x + 20, PANEL.y + 44), f_xs, (190, 200, 215))
            lay = reward_layout()
            for row, spec in zip(lay["rows"], REWARD_ROWS):
                key, label, helptext, _s, _lo, _hi, fmt = spec
                text(label, (PANEL.x + 20, row["y"] + 2), f_s)
                text(helptext, (PANEL.x + 20, row["y"] + 21), f_xs, (150, 160, 175))
                button(row["minus"], "-", font=f_m)
                button(row["plus"], "+", font=f_m)
                val = f_s.render(fmt.format(REWARD[key]), True, (255, 210, 40))
                screen.blit(val, val.get_rect(center=((row["minus"].right + row["plus"].x) // 2, row["minus"].centery)))
            text("Pit stop decision", (PANEL.x + 20, lay["pits_y"] + 2), f_s)
            text("who chooses when to pit: the neural network (4th output), random laps, or fixed laps "
                 f"{PIT_LAPS[0]}/{PIT_LAPS[1]}", (PANEL.x + 20, lay["pits_y"] + 21), f_xs, (150, 160, 175))
            button(lay["pits"], PIT_MODE_NAMES[SETTINGS["pit_mode"]], SETTINGS["pit_mode"] == "ai")
            text("Team call (AI mode)", (PANEL.x + 20, lay["team_y"] + 2), f_s)
            text("pit wall calls the car in when its tyres are dead (wear 100%), even if the AI did not ask",
                 (PANEL.x + 20, lay["team_y"] + 21), f_xs, (150, 160, 175))
            button(lay["team"], "ON" if SETTINGS["team_call"] else "OFF", SETTINGS["team_call"])
            parts = reward_parts(lead, n)           # live breakdown for the leading car
            line = "  ".join(f"{k} {v / n:+.2f}" for k, v in parts.items())
            text(f"Leader #{lead['id'] + 1} score = {sum(parts.values()) / n:.2f} laps   ({line})",
                 (PANEL.x + 20, PANEL.bottom - 76), f_xs, (120, 235, 140))
            button(lay["reset"], "Reset defaults")
            button(lay["close"], "Close")
        else:
            text("NEURAL-NETWORK WEIGHTS OF EVERY CAR", (PANEL.x + 20, PANEL.y + 10), f_m)
            text("red = positive, blue = negative, brighter = larger.", (PANEL.x + 20, PANEL.y + 38), f_xs, (190, 200, 215))
            text("Big grid: 10 hidden neurons (rows) x [7 sensors, speed, tyre wear, race progress, stops made, bias].",
                 (PANEL.x + 20, PANEL.y + 54), f_xs, (190, 200, 215))
            text("Small grid: [steer, throttle, brake, PIT] (rows) x [10 hidden neurons, bias]. "
                 "The bottom row is the pit-stop decision.", (PANEL.x + 20, PANEL.y + 70), f_xs, (190, 200, 215))
            cs, bw = 10, (PANEL.w - 32) // 5
            for c in T.cars:
                col, row = c["id"] % 5, c["id"] // 5
                bx, by = PANEL.x + 16 + col * bw, PANEL.y + 98 + row * 196
                on = c["alive"] or c["finished"]
                tcol = PALETTE[c["id"]] if on else (125, 125, 132)
                pygame.draw.rect(screen, PALETTE[c["id"]], (bx, by + 2, 11, 11))
                text(f"#{c['id'] + 1}  {c['fit'] / n:5.2f}  {c['comp']} {int(min(c['wear'], 1.3) * 100)}%",
                     (bx + 16, by), f_xs, tcol)
                w = c["w"]
                for h in range(N_HID):
                    for i in range(N_IN + 1):
                        pygame.draw.rect(screen, weight_color(w[h * (N_IN + 1) + i]),
                                         (bx + i * cs, by + 20 + h * cs, cs - 1, cs - 1))
                base = N_HID * (N_IN + 1)
                for k in range(N_OUT):
                    for j in range(N_HID + 1):
                        pygame.draw.rect(screen, weight_color(w[base + k * (N_HID + 1) + j]),
                                         (bx + j * cs, by + 128 + k * cs, cs - 1, cs - 1))
                pit_out = max(abs(x) for x in w[base + 3 * (N_HID + 1):])
                text("PIT", (bx + (N_HID + 1) * cs + 4, by + 128 + 3 * cs - 3), f_xs,
                     (80, 230, 230) if pit_out > 0.05 else (110, 110, 118))
                mean = sum(abs(x) for x in w) / len(w)
                text(f"mean |w| {mean:.2f}  max {max(abs(x) for x in w):.1f}", (bx, by + 172), f_xs, (150, 160, 175))
            text("Press the Weights button again to close.", (PANEL.x + 20, PANEL.bottom - 24), f_xs, (150, 160, 175))

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.MOUSEWHEEL:
                for d in dds:
                    if d.open:
                        d.scroll(-e.y)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if ui["panel"] is not None:
                        ui["panel"] = None
                    else:
                        pygame.quit()
                        sys.exit()
                elif e.key == pygame.K_SPACE:
                    si = (si + 1) % len(speeds)
                elif e.key == pygame.K_n:
                    T.end_gen()
                elif e.key == pygame.K_r:
                    T = new_trainer(T.ti, None, 1, "Restarted from random brains")
                elif pygame.K_1 <= e.key <= pygame.K_9 and e.key - pygame.K_1 < len(F.TRACKS):
                    ti = e.key - pygame.K_1
                    dd_track.sel = ti
                    T = new_trainer(ti, T.parent, T.gen, f"Circuit: {track_name(ti)} - brain kept")
            if e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                consumed, changed, which = False, None, None
                opened = [d for d in dds if d.open]
                if opened:
                    which = opened[0]
                    _, changed = which.click(e.pos)
                    consumed = True
                    for o in dds:   # clicking another header opens it directly
                        if o is not which and o.rect.collidepoint(e.pos):
                            o.show()
                elif e.button == 1:
                    for d in dds:
                        c, changed = d.click(e.pos)
                        if c:
                            consumed, which = True, d
                            break
                if which is dd_method and changed is not None:
                    T.method = changed
                    T.msg = f"Method changed to {METHODS[changed][0]} - used from the next generation"
                elif which is dd_obst and changed is not None:
                    T.set_obstacle_mode(changed)
                    T.msg = f"{OBSTACLES[changed][0]} - generation restarted"
                elif which is dd_barrier and changed is not None:
                    T.msg = f"Barrier style: {BARRIERS[changed]}"
                elif which is dd_track and changed is not None:
                    T = new_trainer(changed, T.parent, T.gen, f"Circuit: {track_name(changed)} - brain kept")
                elif which is dd_mode and changed is not None:
                    T = new_trainer(T.ti, T.parent, T.gen,
                                    f"{RACE_LAPS}-lap race with 2 pit stops" if changed == 0
                                    else "Sprint mode: 50 s generations, no tyres or pit stops")
                if not consumed and e.button == 1:
                    if btn_reward.collidepoint(e.pos):
                        ui["panel"] = None if ui["panel"] == "reward" else "reward"
                        consumed = True
                    elif btn_weights.collidepoint(e.pos):
                        ui["panel"] = None if ui["panel"] == "weights" else "weights"
                        consumed = True
                    elif ui["panel"] is not None and PANEL.collidepoint(e.pos):
                        if ui["panel"] == "reward":
                            reward_click(e.pos)
                        consumed = True
                if (not consumed and ui["panel"] is None
                        and WX <= e.pos[0] < WX + VIEW_W and WY <= e.pos[1] < WY + VIEW_H):
                    kind = dd_barrier.sel - 1 if dd_barrier.sel > 0 else random.randrange(3)
                    T.click(((e.pos[0] - WX) / SCALE, (e.pos[1] - WY) / SCALE), e.button == 1, kind)
                    T.msg = f"{len(T.obstacles())} obstacles on this circuit"

        for _ in range(speeds[si]):
            T.update()

        # ---------- circuit ----------
        tr = T.view
        world.blit(tr["surf"], (0, 0))
        if T.race:
            draw_pit_lane(world, tr)
        for x, y, kind in T.obstacles():   # realistic look; collision is the wall circle in the view surface
            draw_barrier(world, x, y, dd_barrier.sel - 1 if dd_barrier.sel > 0 else kind)
        alive = [c for c in T.cars if c["alive"]]
        leader = max(alive, key=lambda c: c["fit"]) if alive else None
        if leader and leader["pit"] is None:  # show what the leading car "sees"
            for d, val in zip(RAYS, sense(tr, leader)):
                ang = leader["a"] + math.radians(d)
                end = (leader["x"] + math.cos(ang) * val * RAY_LEN, leader["y"] + math.sin(ang) * val * RAY_LEN)
                pygame.draw.line(world, (255, 235, 90), (leader["x"], leader["y"]), end, 1)
        for c in sorted(T.cars, key=lambda c: c["alive"]):
            img = car_img(PALETTE[c["id"]]).copy()
            if not c["alive"]:
                img.set_alpha(150 if c["finished"] else 70)
            rot = pygame.transform.rotate(img, -math.degrees(c["a"]))
            world.blit(rot, rot.get_rect(center=(c["x"], c["y"])))
            if c["elite"] and c["alive"]:
                pygame.draw.circle(world, (255, 210, 40), (int(c["x"]), int(c["y"])), 17, 2)
            if T.race and (c["alive"] or c["finished"]):   # dot = tyre compound
                pygame.draw.circle(world, TYRES[c["comp"]]["col"], (int(c["x"]), int(c["y"])), 4)
                pygame.draw.circle(world, (20, 20, 20), (int(c["x"]), int(c["y"])), 4, 1)
            if c["pit"] is not None:
                pygame.draw.circle(world, (80, 230, 230), (int(c["x"]), int(c["y"])), 21, 2)
        screen.fill(BG)
        screen.blit(pygame.transform.smoothscale(world, (VIEW_W, VIEW_H)), (WX, WY))

        # ---------- top bar ----------
        text("Training method", (10, 10))
        text("Obstacles", (246, 10))
        text("Barrier", (360, 10))
        text("Circuit", (516, 10))
        text("Mode", (712, 10))
        button(btn_reward, "Reward", ui["panel"] == "reward")
        button(btn_weights, "Weights", ui["panel"] == "weights")

        # ---------- right column ----------
        n = tr["n"]
        lead = max(T.cars, key=lambda c: c["fit"])
        text(f"GENERATION {T.gen}", (COL_X, 8), f_m)
        text(track_name(T.ti), (COL_X, 38), f_xs)
        text(f"alive {len(alive)}/{N_CARS}   speed {speeds[si]}x", (COL_X, 58), f_xs)
        text(f"time {T.t / 60:5.0f} / {T.max_frames // 60} s", (COL_X, 78), f_xs)
        if T.race:
            text(f"LAP {min(max(c['laps'] for c in T.cars) + 1, RACE_LAPS)}/{RACE_LAPS}"
                 f"   obstacles {len(T.obstacles())}", (COL_X, 98), f_xs, (255, 210, 40))
        else:
            text(f"obstacles {len(T.obstacles())}", (COL_X, 98), f_xs)
        text(f"best score {lead['fit'] / n:.2f} laps", (COL_X, 118), f_xs)
        text(f"best lap {T.best_lap:.1f} s" if T.best_lap else "best lap -", (COL_X, 138), f_xs)

        pygame.draw.rect(screen, EDGE, (COL_X - 4, 164, F.W - COL_X - 2, 252), 1)
        text("Cars: score  tyre  pit", (COL_X, 170), f_xs)
        for r, c in enumerate(sorted(T.cars, key=lambda c: -c["fit"])):
            y = 192 + r * 22
            pygame.draw.rect(screen, PALETTE[c["id"]], (COL_X, y + 2, 11, 11))
            col = (235, 235, 235) if (c["alive"] or c["finished"]) else (125, 125, 132)
            if c["pit"]:
                label = "PIT"
            elif not c["alive"]:
                label = c["why"]
            elif T.race and c["pit_req"]:
                label = "BOX"
            elif T.race:
                label = pit_text(c) if SETTINGS["pit_mode"] != "ai" else f"{c['stops']}stp"
            else:
                label = c["tag"]
            tyre = f"{c['comp']}{int(min(c['wear'], 1.3) * 100):3d}%" if T.race else ""
            text(f"#{c['id'] + 1:<2}{c['fit'] / n:5.2f} {tyre} {label}", (COL_X + 16, y), f_xs, col)

        pygame.draw.rect(screen, EDGE, (COL_X - 4, 426, F.W - COL_X - 2, 150), 1)
        text("Best score per generation (laps)", (COL_X, 432), f_xs)
        h = T.history[-60:]
        if len(h) > 1:
            top = max(max(h), 1.0)
            cw = F.W - COL_X - 14
            pl = [(COL_X + i * cw / (len(h) - 1), 568 - (max(v, 0) / top) * 110) for i, v in enumerate(h)]
            pygame.draw.lines(screen, (255, 210, 40), False, pl, 2)
            text(f"{h[-1]:.2f}", (COL_X, 452), f_xs, (255, 210, 40))
        else:
            text("(needs 2 generations)", (COL_X, 490), f_xs, (150, 150, 155))

        # ---------- tyre / pit panel: leader's tyres and lap times ----------
        pygame.draw.rect(screen, EDGE, (COL_X - 4, 584, F.W - COL_X - 2, 112), 1)
        if T.race:
            grip, _ = tyre_factors(lead, True)
            pygame.draw.circle(screen, TYRES[lead["comp"]]["col"], (COL_X + 6, 598), 6)
            text(f"Leader #{lead['id'] + 1}  {TYRES[lead['comp']]['name']}  wear "
                 f"{int(min(lead['wear'], 1.3) * 100)}%", (COL_X + 16, 591), f_xs)
            text(f"grip {grip * 100:.0f}%  stop {lead['stops']}/{MANDATORY_STOPS}  {pit_text(lead)}",
                 (COL_X, 609), f_xs, (190, 200, 215))
            lts = lead["lts"]
            cx0, cy0, cw, ch = COL_X, 646, F.W - COL_X - 14, 42
            if len(lts) > 1:
                lo, hi = min(lts), max(lts)
                rng = max(hi - lo, 0.5)
                for m in lead["pit_marks"]:   # cyan tick = pit stop
                    px = cx0 + (m - 0.5) * cw / (len(lts) - 1)
                    pygame.draw.line(screen, (80, 230, 230), (px, cy0), (px, cy0 + ch), 1)
                pl = [(cx0 + i * cw / (len(lts) - 1), cy0 + ch - (v - lo) / rng * ch) for i, v in enumerate(lts)]
                pygame.draw.lines(screen, (255, 210, 40), False, pl, 2)
                text(f"lap times: last {lts[-1]:.1f}s  best {lo:.1f}s", (COL_X, 628), f_xs, (190, 200, 215))
            else:
                text("lap times (needs 2 laps)", (COL_X, 628), f_xs, (150, 150, 155))
        else:
            text("Sprint mode: no tyres,", (COL_X, 600), f_xs, (150, 150, 155))
            text("no pit stops. Switch the", (COL_X, 620), f_xs, (150, 150, 155))
            text("Mode drop-down to Race.", (COL_X, 640), f_xs, (150, 150, 155))

        # ---------- bottom bar ----------
        text("SPACE speed   N next generation   1-9 circuit   R restart", (10, WY + VIEW_H + 8), f_xs, (190, 200, 215))
        text(METHODS[dd_method.sel][1], (480, WY + VIEW_H + 8), f_xs, (190, 200, 215))
        text("Click the road to add a barrier, right-click one to remove it", (10, WY + VIEW_H + 30), f_xs, (190, 200, 215))
        if T.race:
            text(f"Race: {RACE_LAPS} laps, " + {"ai": "the AI chooses its pit stops",
                                                "random": "random pit stops",
                                                "fixed": f"pit on laps {PIT_LAPS[0]} and {PIT_LAPS[1]}"}[SETTINGS["pit_mode"]],
                 (480, WY + VIEW_H + 30), f_xs, (190, 200, 215))
        text(T.msg, (10, F.H - 20), f_xs, (255, 255, 255))
        draw_panel(lead)
        for d in dds:   # drop-downs last so open lists are on top
            d.draw(screen, f_s)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()