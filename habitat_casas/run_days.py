"""Generate a multi-day, two-resident CASAS-style dataset.

    caffeinate -i python -m habitat_casas.run_days --days 10   # keep the Mac awake

Finished days are kept (day_kk/summary.json), so an interrupted run can be
restarted with the same command; --fresh regenerates everything.

Each day: reset the apartment, generate a random routine pair (days.py) from
seed = --seed + day, simulate it with sensors, and export it. Day k is dated
--date + k days, starting at a random time between 06:30 and 08:30, and is one
session (pair_id "day_kk") in the Kyoto-format file.

Output (default outputs/dataset/):
  day_kk/                 per-day files (same as run_routines)
  kyoto_format.csv        all days, Kyoto multi-resident columns
  casas_cairo.txt         all days, Cairo raw format with begin/end labels
  casas_smrt.txt          all days, smrt_impl parser format
  activities_gt.csv       all ground-truth activity intervals
  activity_ids.csv        fixed activity id table
  days.csv                per-day summary (plan, duration, events)
"""
import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timedelta

import numpy as np

from habitat_casas.activities import RoutineRunner
from habitat_casas.days import LABELS, generate_day
from habitat_casas.env import REPO_ROOT, make_multi_humanoid_env, rebuild_navmesh
from habitat_casas.export import export
from habitat_casas.scenario_apt import CONTACTS, HOME, ITEMS, MOTION
from habitat_casas.sensors import SensorSuite

EPISODE_ID = "10"  # v3_sc3_staging_16, the layout HOME/MOTION are measured on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="day k uses seed + k")
    ap.add_argument("--date", default="2026-10-08", help="date of day 0")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "dataset"))
    ap.add_argument("--fresh", action="store_true", help="redo days that already exist")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = make_multi_humanoid_env(num_agents=2, seed=args.seed)
    env.episodes = [e for e in env.episodes if e.episode_id == EPISODE_ID]

    summary = []
    t_all = time.time()
    for k in range(args.days):
        seed = args.seed + k
        rng = random.Random(seed)
        start = datetime.strptime(args.date, "%Y-%m-%d") + timedelta(
            days=k, minutes=rng.randint(6 * 60 + 30, 8 * 60 + 30))
        day_dir = os.path.join(args.out, f"day_{k:02d}")
        done_file = os.path.join(day_dir, "summary.json")
        if not args.fresh and os.path.exists(done_file):  # resume: keep finished days
            with open(done_file) as f:
                summary.append(json.load(f))
            print(f"day {k:02d}  already done, skipping", flush=True)
            continue

        env.reset()  # restores objects and furniture to the episode's start state
        rebuild_navmesh(env.sim)
        routines, plan = generate_day(seed)
        runner = RoutineRunner(env, HOME, routines)
        sensors = SensorSuite(runner, MOTION, CONTACTS, ITEMS, seed=seed)
        t0 = time.time()
        runner.run()
        sensors.finish()
        runner.save(day_dir)
        sensors.save(day_dir)
        export(day_dir, start.strftime("%Y-%m-%d %H:%M:%S"), pair_id=f"day_{k:02d}", labels=LABELS)
        with open(os.path.join(day_dir, "plan.json"), "w") as f:
            json.dump(plan, f, indent=1)

        shared = sum(";" in e[3] for e in sensors.events)
        row = {"day": k, "seed": seed, "start": start.strftime("%Y-%m-%d %H:%M"),
               "minutes": round(runner.t / 60, 1), "events": len(sensors.events),
               "shared_events": shared, "wall_s": round(time.time() - t0, 1), **plan}
        summary.append(row)
        with open(done_file, "w") as f:
            json.dump(row, f)
        print(f"day {k:02d}  {row['start']}  {row['minutes']:5.1f} min  {row['events']:4d} events "
              f"({shared} shared)  {row['wall_s']:5.1f}s  {plan}", flush=True)

    env.close()
    combine(args.out, args.days)
    with open(os.path.join(args.out, "days.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{args.days} days in {(time.time() - t_all) / 60:.1f} min -> {args.out}")


def combine(out_dir: str, n_days: int) -> None:
    """Concatenate the per-day exports into dataset-level files."""
    days = [os.path.join(out_dir, f"day_{k:02d}") for k in range(n_days)]
    for name, has_header in [("kyoto_format.csv", True), ("activities_gt.csv", True),
                             ("casas_cairo.txt", False), ("casas_smrt.txt", False)]:
        newline = "" if name.endswith(".csv") else None
        with open(os.path.join(out_dir, name), "w", newline=newline) as out:
            for i, d in enumerate(days):
                with open(os.path.join(d, name), newline=newline) as f:
                    lines = f.readlines()
                out.writelines(lines if (i == 0 or not has_header) else lines[1:])
    with open(os.path.join(out_dir, "activity_ids.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["activity_id", "activity"])
        w.writerows([[i + 1, n] for i, n in enumerate(LABELS)])


if __name__ == "__main__":
    main()
