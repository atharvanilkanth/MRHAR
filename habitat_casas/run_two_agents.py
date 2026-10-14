"""Two humanoids walk to different objects in the same scene at the same time.

Writes a per-step trajectory CSV (ground truth for the sensor logger), a
top-down plot, and optionally a third-person video per agent.

    python -m habitat_casas.run_two_agents --video
"""
import argparse
import csv
import os
import random
import time

import numpy as np

from habitat_casas.env import (
    REPO_ROOT,
    MultiHumanoidDriver,
    make_multi_humanoid_env,
    nav_point_near,
    object_positions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--visits", type=int, default=3, help="objects per agent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "two_agents"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)  # agent spawning uses the global RNGs
    np.random.seed(args.seed)
    env = make_multi_humanoid_env(num_agents=2, render=args.video, seed=args.seed)
    env.reset()
    sim = env.sim
    driver = MultiHumanoidDriver(env)
    dt = 1.0 / sim.ctrl_freq

    # Give each agent its own list of objects to visit.
    rng = np.random.default_rng(args.seed)
    objs = {h: p for h, p in object_positions(sim).items()
            if nav_point_near(sim, p) is not None}
    handles = list(rng.permutation(sorted(objs)))
    plans = [handles[i::2][: args.visits] for i in range(driver.n)]
    for i, plan in enumerate(plans):
        print(f"agent_{i} starts at {np.round(driver.position(i), 2)}, plan: {plan}")
        driver.set_goal(i, objs[plan[0]])
    visit_idx = [0] * driver.n

    rows, frames = [], [[] for _ in range(driver.n)]
    t0 = time.time()
    for step in range(args.steps):
        obs = driver.step()
        for i in range(driver.n):
            p = driver.position(i)
            target = plans[i][visit_idx[i]] if visit_idx[i] < len(plans[i]) else ""
            rows.append([step, round(step * dt, 3), f"agent_{i}", *np.round(p, 3), target])
            if args.video and step % 4 == 0:  # 120 Hz -> 30 fps
                frames[i].append({"third_rgb": obs[f"agent_{i}_third_rgb"]})
            if driver.arrived(i) and visit_idx[i] < len(plans[i]):
                print(f"step {step:5d}  agent_{i} reached {plans[i][visit_idx[i]]}")
                visit_idx[i] += 1
                if visit_idx[i] < len(plans[i]):
                    driver.set_goal(i, objs[plans[i][visit_idx[i]]])
        if all(v >= len(p) for v, p in zip(visit_idx, plans)):
            break
    elapsed = time.time() - t0
    print(f"{step + 1} steps ({(step + 1) * dt:.1f}s sim time) in {elapsed:.1f}s wall "
          f"= {(step + 1) / elapsed:.0f} steps/s")

    traj_path = os.path.join(args.out, "trajectories.csv")
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "agent_id", "x", "y", "z", "target"])
        w.writerows(rows)
    print("wrote", traj_path)

    plot_topdown(rows, objs, plans, os.path.join(args.out, "topdown.png"))

    if args.video:
        from habitat_sim.utils import viz_utils as vut

        for i in range(driver.n):
            vut.make_video(frames[i], "third_rgb", "color",
                           os.path.join(args.out, f"agent_{i}"), fps=30, open_vid=False)
    env.close()


def plot_topdown(rows, objs, plans, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for k, (aid, color) in enumerate([("agent_0", "tab:blue"), ("agent_1", "tab:orange")]):
        xz = np.array([[r[3], r[5]] for r in rows if r[2] == aid])
        ax.plot(xz[:, 0], xz[:, 1], color=color, label=aid)
        ax.scatter(*xz[0], color=color, marker="o", s=60)
        for h in plans[k]:
            ax.scatter(objs[h].x, objs[h].z, color=color, marker="*", s=150)
            ax.annotate(h.split(":")[0], (objs[h].x, objs[h].z), fontsize=7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Two-agent trajectories (o = start, * = visited object)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
