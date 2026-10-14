"""Run the scripted two-resident routine in the ReplicaCAD apartment.

    python -m habitat_casas.run_routines [--video]

Writes to outputs/routines/: trajectories.csv, activities.csv,
interactions.csv, sensor_events.csv, sensor_layout.csv, timeline.png,
topdown.png, sensor_events.png, CASAS exports (casas_cairo.txt,
casas_smrt.txt, kyoto_format.csv; see export.py) (+ agent_<i>.mp4 with --video).
"""
import argparse
import os
import random
import time

import numpy as np

from habitat_casas.activities import RoutineRunner
from habitat_casas.export import DEFAULT_START, export
from habitat_casas.env import REPO_ROOT, make_multi_humanoid_env, rebuild_navmesh
from habitat_casas.scenario_apt import CONTACTS, HOME, ITEMS, MOTION, ROUTINES
from habitat_casas.sensors import SensorSuite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--start", default=DEFAULT_START, help="wall-clock time of t=0")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "routines"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = make_multi_humanoid_env(num_agents=len(ROUTINES), render=args.video, seed=args.seed)
    env.reset()
    rebuild_navmesh(env.sim)
    runner = RoutineRunner(env, HOME, ROUTINES)
    sensors = SensorSuite(runner, MOTION, CONTACTS, ITEMS, seed=args.seed)

    frames = {r.agent: [] for r in ROUTINES}
    if args.video:
        def grab(r, obs):
            if r.n_steps % 4 == 0:  # 120 Hz -> 30 fps
                for i in frames:
                    frames[i].append({"third_rgb": obs[f"agent_{i}_third_rgb"]})
        runner.frame_cb = grab

    t0 = time.time()
    runner.run()
    wall = time.time() - t0
    print(f"sim time {runner.t / 60:.1f} min, {runner.n_steps} physics steps, "
          f"{wall:.1f}s wall ({runner.t / wall:.0f}x real time)")

    sensors.finish()
    runner.save(args.out)
    sensors.save(args.out)
    export(args.out, args.start, pair_id=f"sim{args.seed:02d}")
    for row in runner.activity_log:
        print(f"  {row[0]}  {row[1]:18s} {row[2]:8.1f}s -> {row[3]:8.1f}s")
    print(f"  {len(runner.interactions)} interactions, {len(sensors.events)} sensor events")
    plot_sensor_events(sensors, runner.activity_log, os.path.join(args.out, "sensor_events.png"))
    plot_timeline(runner.activity_log, os.path.join(args.out, "timeline.png"))
    plot_topdown(runner.trajectory, HOME, os.path.join(args.out, "topdown.png"))

    if args.video:
        from habitat_sim.utils import viz_utils as vut

        for i, fr in frames.items():
            vut.make_video(fr, "third_rgb", "color", os.path.join(args.out, f"agent_{i}"),
                           fps=30, open_vid=False)
    print("wrote", args.out)
    env.close()


def plot_timeline(activity_log, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = sorted({r[0] for r in activity_log})
    labels = list(dict.fromkeys(r[1] for r in activity_log))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.8 * len(agents)))
    for agent, label, start, end in activity_log:
        y = agents.index(agent)
        c = cmap(labels.index(label) % 20)
        ax.barh(y, (end - start) / 60, left=start / 60, color=c, edgecolor="white")
        if end - start > 45:
            ax.text((start + end) / 120, y, label, ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.invert_yaxis()
    ax.set_xlabel("minutes")
    ax.set_title("Ground-truth activities")
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_sensor_events(sensors, activity_log, path):
    """One row per sensor; marks coloured by the agent that caused the event."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [r[0] for r in sensors.layout()]
    names = {r[0]: r[2] for r in sensors.layout()}
    colors = {"agent_0": "tab:blue", "agent_1": "tab:orange"}
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(ids) + 1.5))
    on = {}
    for t, sid, val, agents, _ in sensors.events:
        y = ids.index(sid)
        c = colors.get(agents, "purple")  # purple = several agents at once
        if val in ("ON", "OPEN", "ABSENT"):
            on[sid] = (t, c)
        elif sid in on:
            t0, c0 = on.pop(sid)
            ax.plot([t0 / 60, max(t, t0 + 1) / 60], [y, y], color=c0, lw=6, solid_capstyle="butt")
    t_end = max(r[3] for r in activity_log)
    for sid, (t0, c0) in on.items():  # still open/absent at the end
        ax.plot([t0 / 60, t_end / 60], [ids.index(sid)] * 2, color=c0, lw=6, solid_capstyle="butt")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{i} {names[i][:22]}" for i in ids], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("minutes")
    handles = [plt.Line2D([], [], color=c, lw=6) for c in [*colors.values(), "purple"]]
    ax.legend(handles, [*colors, "both"], loc="lower right", fontsize=8)
    ax.set_title("Sensor activity (ON / OPEN / ABSENT periods), coloured by ground-truth agent")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_topdown(trajectory, home, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 8))
    for agent, color in [("agent_0", "tab:blue"), ("agent_1", "tab:orange")]:
        xz = np.array([[r[3], r[5]] for r in trajectory if r[2] == agent])
        ax.plot(xz[:, 0], xz[:, 1], color=color, label=agent, alpha=0.8)
        ax.scatter(*xz[0], color=color, s=50)
    for name, p in home.places.items():
        ax.scatter(p[0], p[2], color="grey", marker="x", s=20)
        ax.annotate(name, (p[0], p[2]), fontsize=7, color="dimgrey")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Paths (dot = start)")
    fig.savefig(path, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
