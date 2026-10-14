"""Export simulated sensor events as CASAS-format datasets.

Formats (matched against the real files in ~/Desktop/FEAT/dataset):

  casas_cairo.txt    CASAS Cairo raw format, tab separated:
                       2026-10-08 07:00:11.267000<TAB>D002<TAB>OPEN[<TAB>R1_Meal_Preparation begin]
                     Activity labels sit on the first/last event of each activity
                     interval that the resident caused, as in Cairo.
  casas_smrt.txt     Format read by smrt_impl/smrt_gamut/data/parser.py:
                       2026-10-08 07:00:11.267000 D002 OPEN R1 Meal_Preparation
                     (residents "R1,R2" and activities "A,B" for joint events)
  kyoto_format.csv   Same columns as kyoto_multiresident_clean.csv (read by
                     smrt_gamut/data/kyoto_csv.py): pair_id,date,time,sensor,message,
                     then (resident, activity id) pairs in task_field_1..6.
  activity_ids.csv   Activity name <-> id used in kyoto_format.csv.
  activities_gt.csv  Ground-truth activity intervals with timestamps.

Residents: agent_<i> -> R<i+1>.

    python -m habitat_casas.export outputs/routines --start "2026-10-08 07:00:00"
"""
import argparse
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

DEFAULT_START = "2026-10-08 07:00:00"


def _resident(agent_id: str) -> str:
    return f"R{int(agent_id.split('_')[1]) + 1}"


def _read(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def export(run_dir: str, start: str = DEFAULT_START, pair_id: str = "sim01",
           labels: Optional[List[str]] = None) -> Dict[str, str]:
    """labels: fixed activity list so ids stay the same across runs/days
    (default: the labels of this run, sorted)."""
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    events = _read(os.path.join(run_dir, "sensor_events.csv"))
    activities = _read(os.path.join(run_dir, "activities.csv"))

    def ts(t: str) -> datetime:
        return t0 + timedelta(seconds=float(t))

    def fmt(t: str) -> str:
        return ts(t).strftime("%Y-%m-%d %H:%M:%S.%f")

    seen = {a["activity"] for a in activities}
    labels = list(labels or []) + sorted(seen - set(labels or []))
    act_id = {name: i + 1 for i, name in enumerate(labels)}
    out = {}

    # --- Cairo-style raw log with begin/end annotations
    notes: Dict[int, str] = {}
    dropped = 0
    for a in activities:
        start_t, end_t = float(a["start_t"]), float(a["end_t"])
        mine = [k for k, e in enumerate(events)
                if a["agent_id"] in e["agent_ids"].split(";")
                and start_t <= float(e["t"]) <= end_t]
        if not mine:
            dropped += 1  # activity produced no sensor events (e.g. outside)
            continue
        label = f"{_resident(a['agent_id'])}_{a['activity']}"
        for k, tag in ((mine[0], "begin"), (mine[-1], "end")):
            if k in notes:  # one annotation per line in Cairo; keep the first
                dropped += 1
                continue
            notes[k] = f"{label} {tag}"
    path = os.path.join(run_dir, "casas_cairo.txt")
    with open(path, "w") as f:
        for k, e in enumerate(events):
            cols = [fmt(e["t"]), e["sensor_id"], e["value"]]
            if k in notes:
                cols.append(notes[k])
            f.write("\t".join(cols) + "\n")
    out["cairo"] = path
    if dropped:
        print(f"note: {dropped} activity begin/end labels could not be placed in casas_cairo.txt")

    # --- sMRT parser format (one line per event, residents + activity)
    path = os.path.join(run_dir, "casas_smrt.txt")
    with open(path, "w") as f:
        for e in events:
            agents = e["agent_ids"].split(";")
            res = ",".join(_resident(a) for a in agents) if e["agent_ids"] != "none" else "-"
            acts = ",".join(e["activities"].split(";"))
            f.write(f"{fmt(e['t'])} {e['sensor_id']} {e['value']} {res} {acts}\n")
    out["smrt"] = path

    # --- Kyoto multi-resident CSV
    path = os.path.join(run_dir, "kyoto_format.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "date", "time", "sensor", "message"]
                   + [f"task_field_{i}" for i in range(1, 7)])
        for e in events:
            d = ts(e["t"])
            fields = []
            if e["agent_ids"] != "none":
                for agent, act in zip(e["agent_ids"].split(";"), e["activities"].split(";")):
                    fields += [_resident(agent)[1:], act_id.get(act, "")]
            fields = (fields + [""] * 6)[:6]
            w.writerow([pair_id, d.strftime("%Y-%m-%d"), d.strftime("%H:%M:%S.%f"),
                        e["sensor_id"], e["value"], *fields])
    out["kyoto"] = path

    path = os.path.join(run_dir, "activity_ids.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["activity_id", "activity"])
        w.writerows([[i, n] for n, i in act_id.items()])
    out["activity_ids"] = path

    path = os.path.join(run_dir, "activities_gt.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resident", "activity", "begin", "end"])
        for a in activities:
            w.writerow([_resident(a["agent_id"]), a["activity"], fmt(a["start_t"]), fmt(a["end_t"])])
    out["activities_gt"] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--pair-id", default="sim01")
    args = ap.parse_args()
    for k, p in export(args.run_dir, args.start, args.pair_id).items():
        print(f"{k:14s} {p}")


if __name__ == "__main__":
    main()
