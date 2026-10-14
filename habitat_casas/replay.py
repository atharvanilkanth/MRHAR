"""Top-down replay video of one simulated day: residents, sensors, live event log.

    python -m habitat_casas.replay outputs/dataset/day_09 --speed 10

Left: apartment floor plan (navmesh; furniture shows as holes) with motion
zones that light up while ON (coloured by the resident who triggered them),
door/drawer contacts (squares, filled when OPEN), item pads (circles, filled
when ABSENT) and the residents with their current activity. Right: clock,
resident status and the CASAS event log scrolling as events happen.

The floor plan is computed once from the sim and cached in outputs/apt_map.npz.
"""
import argparse
import bisect
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from habitat_casas.env import REPO_ROOT
from habitat_casas.export import DEFAULT_START

MAP_CACHE = os.path.join(REPO_ROOT, "outputs", "apt_map.npz")
RES = 0.04  # m per map cell
W, H = 1280, 904  # multiple of 8 for the encoder
MAP_W = 520  # left panel width
COLORS = {"agent_0": (31, 119, 180), "agent_1": (255, 127, 14)}
MIXED = (148, 103, 189)  # several residents / unattributed
FONTS = ["/System/Library/Fonts/Menlo.ttc",  # macOS
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]  # Linux


def _font(size):
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ------------------------------------------------------------------ map
def build_map(path: str = MAP_CACHE) -> dict:
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k].item() if d[k].dtype == object else d[k] for k in d.files}
    import random

    from habitat_casas.env import make_multi_humanoid_env, rebuild_navmesh
    from habitat_casas.activities import find_link
    from habitat_casas.run_days import EPISODE_ID
    from habitat_casas.scenario_apt import CONTACTS, HOME, ITEMS

    random.seed(0)
    np.random.seed(0)
    env = make_multi_humanoid_env(2)
    env.episodes = [e for e in env.episodes if e.episode_id == EPISODE_ID]
    env.reset()
    sim = env.sim
    rebuild_navmesh(sim)
    lo, hi = sim.pathfinder.get_bounds()
    xs = np.arange(lo[0], hi[0], RES)
    zs = np.arange(lo[2], hi[2], RES)
    grid = np.array([[sim.pathfinder.is_navigable(np.array([x, 0.11, z]), 0.5) for x in xs]
                     for z in zs])
    contacts = {}
    for c in CONTACTS:
        ao, j, _ = find_link(sim, c.furniture, c.link)
        lid = [l for l in ao.get_link_ids() if ao.get_link_joint_pos_offset(l) == j][0]
        p = np.array(ao.get_link_scene_node(lid).absolute_translation)
        contacts[c.sensor_id] = (float(p[0]), float(p[2]))
    rom = sim.get_rigid_object_manager()
    items = {it.sensor_id: tuple(float(v) for v in np.array(rom.get_object_by_handle(it.obj).translation)[[0, 2]])
             for it in ITEMS}
    places = {n: (p[0], p[2]) for n, p in HOME.places.items() if "_top" not in n and not n.endswith("_2")}
    env.close()
    d = {"grid": grid, "x0": float(lo[0]), "z0": float(lo[2]),
         "contacts": contacts, "items": items, "places": places}
    np.savez(path, **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v) for k, v in d.items()})
    return d


# ------------------------------------------------------------------ data
def load_day(day_dir: str):
    traj = defaultdict(lambda: ([], [], [], []))  # t, x, z, activity
    with open(os.path.join(day_dir, "trajectories.csv")) as f:
        for r in csv.DictReader(f):
            T = traj[r["agent_id"]]
            T[0].append(float(r["t"]))
            T[1].append(float(r["x"]))
            T[2].append(float(r["z"]))
            T[3].append(r["activity"])
    with open(os.path.join(day_dir, "sensor_events.csv")) as f:
        events = [(float(r["t"]), r["sensor_id"], r["value"], r["agent_ids"], r["activities"])
                  for r in csv.DictReader(f)]
    with open(os.path.join(day_dir, "sensor_layout.csv")) as f:
        layout = list(csv.DictReader(f))
    summary = os.path.join(day_dir, "summary.json")  # run_days output
    if os.path.exists(summary):
        with open(summary) as f:
            start = datetime.strptime(json.load(f)["start"], "%Y-%m-%d %H:%M")
    else:  # run_routines output
        start = datetime.strptime(DEFAULT_START, "%Y-%m-%d %H:%M:%S")
    return traj, events, layout, start


# ------------------------------------------------------------------ render
class Renderer:
    def __init__(self, amap, layout):
        g = amap["grid"]
        self.x0, self.z0 = amap["x0"], amap["z0"]
        span_x, span_z = g.shape[1] * RES, g.shape[0] * RES
        self.scale = min((MAP_W - 30) / span_x, (H - 40) / span_z)  # px per m
        self.ox = (MAP_W - span_x * self.scale) / 2
        self.oy = (H - span_z * self.scale) / 2
        floor = np.where(g, 236, 196).astype(np.uint8)
        img = Image.fromarray(floor).resize(
            (int(span_x * self.scale), int(span_z * self.scale)), Image.NEAREST).convert("RGB")
        self.base = Image.new("RGB", (W, H), (250, 250, 248))
        self.base.paste(img, (int(self.ox), int(self.oy)))
        self.f_small = _font(12)
        self.f_log = _font(14)
        self.f_big = _font(26)
        self.f_mid = _font(16)
        self.zones = {r["sensor_id"]: (float(r["x_min"]), float(r["x_max"]), float(r["z_min"]), float(r["z_max"]), r["name"])
                      for r in layout if r["type"] == "motion"}
        self.contacts = amap["contacts"]
        self.items = amap["items"]
        d = ImageDraw.Draw(self.base)
        for name, (x, z) in amap["places"].items():
            px, py = self.px(x, z)
            d.text((px + 3, py + 3), name, fill=(150, 150, 150), font=self.f_small)
        d.line([(MAP_W, 0), (MAP_W, H)], fill=(210, 210, 210), width=2)

    def px(self, x, z):
        return (self.ox + (x - self.x0) * self.scale, self.oy + (z - self.z0) * self.scale)

    def frame(self, clock, speed, pos, acts, trails, state, log, flash_t, t):
        im = self.base.copy()
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        # motion zones: area sensors (M013+) drawn thin; spot sensors bolder
        for sid, (x0, x1, z0, z1, name) in self.zones.items():
            area = name.endswith("_area")
            a, b = self.px(x0, z0), self.px(x1, z1)
            on, who = state.get(sid, (False, ""))
            if on:
                c = COLORS.get(who, MIXED)
                d.rectangle([a, b], fill=c + ((40 if area else 90),))
            d.rectangle([a, b], outline=(120, 120, 120, 110 if area else 200), width=1)
            if not area:
                d.text((a[0] + 3, a[1] + 2), sid, fill=(90, 90, 90, 255), font=self.f_small)
        for sid, (x, z) in self.contacts.items():
            p = self.px(x, z)
            on, who = state.get(sid, (False, ""))
            fill = COLORS.get(who, MIXED) + (255,) if on else (255, 255, 255, 255)
            d.rectangle([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=fill, outline=(40, 120, 40, 255), width=2)
            d.text((p[0] + 8, p[1] - 7), sid, fill=(40, 120, 40, 255), font=self.f_small)
        for sid, (x, z) in self.items.items():
            p = self.px(x, z)
            on, who = state.get(sid, (False, ""))
            fill = (220, 50, 50, 255) if on else (255, 255, 255, 255)
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=fill, outline=(180, 40, 40, 255), width=2)
            d.text((p[0] - 30, p[1] - 6), sid, fill=(180, 40, 40, 255), font=self.f_small)
        # residents
        for a, (x, z) in pos.items():
            c = COLORS[a]
            pts = [self.px(*q) for q in trails[a]]
            if len(pts) > 1:
                d.line(pts, fill=c + (120,), width=3)
            p = self.px(x, z)
            d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], fill=c + (255,), outline=(255, 255, 255, 255), width=2)
            d.text((p[0] - 8, p[1] - 7), "R" + str(int(a[-1]) + 1), fill=(255, 255, 255, 255), font=self.f_small)
        im = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")

        # right panel
        d = ImageDraw.Draw(im)
        x0 = MAP_W + 24
        d.text((x0, 18), clock.strftime("%Y-%m-%d  %H:%M:%S"), fill=(20, 20, 20), font=self.f_big)
        d.text((x0 + 420, 26), f"{speed:g}x speed", fill=(120, 120, 120), font=self.f_mid)
        y = 64
        for a in sorted(pos):
            d.ellipse([x0, y + 3, x0 + 14, y + 17], fill=COLORS[a])
            d.text((x0 + 22, y), f"R{int(a[-1]) + 1}  {acts[a] or '-'}", fill=(30, 30, 30), font=self.f_mid)
            y += 26
        d.text((x0, y + 6), "zones fill while ON  ·  squares = door/drawer contacts (filled = OPEN)  ·  red dot = item ABSENT",
               fill=(120, 120, 120), font=self.f_small)
        y += 34
        d.line([(x0, y), (W - 20, y)], fill=(210, 210, 210), width=1)
        d.text((x0, y + 6), "CASAS event log  (time  sensor  value  | ground truth: resident  activity)",
               fill=(90, 90, 90), font=self.f_small)
        y += 28
        line_h = 19
        n = (H - y - 10) // line_h
        for (et, sid, val, who, act) in log[-n:]:
            ts = (clock - timedelta(seconds=t - et)).strftime("%H:%M:%S.%f")[:-3]
            res = ",".join("R" + str(int(w[-1]) + 1) for w in who.split(";")) if who != "none" else "-"
            fresh = t - et < flash_t
            col = COLORS.get(who, MIXED) if fresh else (60, 60, 60)
            txt = f"{ts}  {sid:<5}{val:<8}| {res:<6}{act.split(';')[0][:22]}"
            d.text((x0, y), txt, fill=col, font=self.f_log)
            y += line_h
        return im


def render(day_dir: str, out: str, speed: float = 10.0, fps: int = 30,
           t_from: float = 0.0, t_to: float = None) -> str:
    import imageio

    amap = build_map()
    traj, events, layout, start = load_day(day_dir)
    rnd = Renderer(amap, layout)
    t_end = max(T[0][-1] for T in traj.values())
    t_to = t_end if t_to is None else min(t_to, t_end)
    dt = speed / fps
    state, log, k = {}, [], 0
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    t = t_from
    while t <= t_to:
        while k < len(events) and events[k][0] <= t:
            et, sid, val, who, act = events[k]
            state[sid] = (val in ("ON", "OPEN", "ABSENT"), who)
            log.append(events[k])
            k += 1
        pos, acts, trails = {}, {}, {}
        for a, (ts, xs, zs, ac) in traj.items():
            i = max(bisect.bisect_right(ts, t) - 1, 0)
            pos[a] = (xs[i], zs[i])
            acts[a] = ac[i]
            j = max(bisect.bisect_right(ts, t - 8 * speed / 10) - 1, 0)  # short trail
            trails[a] = list(zip(xs[j:i + 1:4], zs[j:i + 1:4])) + [(xs[i], zs[i])]
        clock = start + timedelta(seconds=t)
        writer.append_data(np.asarray(rnd.frame(clock, speed, pos, acts, trails, state, log,
                                                flash_t=1.5 * speed, t=t)))
        t += dt
    writer.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day_dir")
    ap.add_argument("--speed", type=float, default=10.0, help="1 = real time")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--from-min", type=float, default=0.0)
    ap.add_argument("--to-min", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.day_dir, f"replay_{args.speed:g}x.mp4")
    render(args.day_dir, out, args.speed, args.fps, 60 * args.from_min,
           None if args.to_min is None else 60 * args.to_min)
    print("wrote", out)


if __name__ == "__main__":
    main()
