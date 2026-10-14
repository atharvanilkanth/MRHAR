"""Compare a simulated run with the real CASAS Cairo and Kyoto datasets.

    python -m habitat_casas.compare outputs/routines

Reports simple, format-level statistics per dataset (Kyoto per session,
averaged): events/hour, sensor-type mix, motion ON durations, gaps between
events, and how often an event is shared by two residents.
"""
import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

DATA = os.path.expanduser("~/Desktop/FEAT/dataset")


def _ts(d, t):
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in t else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(f"{d} {t}", fmt)


def load_cairo(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                rows.append((_ts(p[0], p[1]), p[2], p[3], None))
    return {"cairo": rows}


def load_kyoto_csv(path):
    sessions = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            res = {r[k] for k in ("task_field_1", "task_field_3", "task_field_5")
                   if r[k].strip() in ("1", "2", "3")}
            sessions[r["pair_id"]].append(
                (_ts(r["date"], r["time"]), r["sensor"], r["message"].upper(), len(res)))
    return sessions


def stats(rows):
    rows = sorted(rows, key=lambda r: r[0])
    hours = max((rows[-1][0] - rows[0][0]).total_seconds() / 3600, 1e-9)
    types = Counter(r[1][0] for r in rows)
    gaps = np.diff([r[0].timestamp() for r in rows])
    on, durs = {}, []
    for t, s, v, _ in rows:
        if s[0] != "M":
            continue
        if v == "ON":
            on.setdefault(s, t)
        elif v.startswith("OF") and s in on:
            durs.append((t - on.pop(s)).total_seconds())
    shared = [r[3] for r in rows if r[3] is not None]
    return {
        "hours": hours,
        "events": len(rows),
        "events/h": len(rows) / hours,
        "sensors": len({r[1] for r in rows}),
        "%M": 100 * types["M"] / len(rows),
        "%D": 100 * types["D"] / len(rows),
        "%I": 100 * types["I"] / len(rows),
        "%T": 100 * types["T"] / len(rows),
        "PIR ON med (s)": float(np.median(durs)) if durs else np.nan,
        "gap med (s)": float(np.median(gaps)) if len(gaps) else np.nan,
        "% 2-resident": 100 * np.mean([n >= 2 for n in shared]) if shared else np.nan,
    }


def summarize(sessions):
    per = [stats(r) for r in sessions.values() if len(r) > 10]
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    out["hours"] = sum(p["hours"] for p in per)
    out["events"] = sum(p["events"] for p in per)
    out["sessions"] = len(per)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cairo", default=os.path.join(DATA, "cairo.txt"))
    ap.add_argument("--kyoto", default=os.path.join(DATA, "kyoto_multiresident_clean.csv"))
    args = ap.parse_args()
    for path in (args.cairo, args.kyoto):
        if not os.path.exists(path):
            ap.error(f"real dataset not found: {path} (pass --cairo / --kyoto)")

    table = {
        "Cairo (real)": summarize(load_cairo(args.cairo)),
        "Kyoto (real)": summarize(load_kyoto_csv(args.kyoto)),
        "Habitat (sim)": summarize(load_kyoto_csv(os.path.join(args.run_dir, "kyoto_format.csv"))),
    }
    keys = ["sessions", "hours", "events", "events/h", "sensors", "%M", "%D", "%I", "%T",
            "PIR ON med (s)", "gap med (s)", "% 2-resident"]
    names = list(table)
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in keys:
        vals = []
        for n in names:
            v = table[n].get(k, np.nan)
            vals.append("n/a" if v != v else (f"{v:.0f}" if abs(v) >= 100 or k in ("sessions", "events", "sensors") else f"{v:.1f}"))
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.run_dir, "comparison.md"), "w") as f:
        f.write("Kyoto values are averaged per session (pair); Cairo has no per-event "
                "resident labels, so '% 2-resident' is n/a.\n\n" + text + "\n")


if __name__ == "__main__":
    main()
