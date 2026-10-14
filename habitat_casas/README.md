# habitat_casas: multi-resident smart-home sensing in Habitat

Simulates two humanoid residents doing daily routines in a ReplicaCAD apartment
fitted with virtual ambient sensors (PIR motion zones, door/drawer contacts,
item pads), and writes **CASAS-style event logs** (`timestamp, sensor_id, value`)
with the ground-truth resident for every event. Output is formatted like the
CASAS **Cairo** and **Kyoto** multi-resident datasets.

Built on top of habitat-lab v0.3.3 (commit `094d6be2f`). All new code is in
this folder; habitat-lab itself is unchanged (only the two tutorial notebooks
in `examples/tutorials/` were run and edited).

---

## 1. Setup

Tested on macOS (Apple M1, CPU only) with Python 3.9 and habitat-sim 0.3.3.
Linux works the same way.

```bash
# 1. Python env (conda, mamba or micromamba all work)
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda activate habitat
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat -y

# 2. this repo (habitat-lab v0.3.3 + the habitat_casas/ folder)
git clone https://github.com/atharvanilkanth/MRHAR.git
cd MRHAR
pip install -e habitat-lab

# 3. Extra Python packages used by habitat_casas
pip install matplotlib pillow pygame imageio imageio-ffmpeg

# 4. Assets (about 2 GB, go into ./data)
python -m habitat_sim.utils.datasets_download \
    --uids replica_cad_dataset rearrange_dataset_v2 hab3_bench_assets ycb hab_fetch \
    --data-path data/
```

Check that these exist afterwards:

```
data/replica_cad/replicaCAD.scene_dataset_config.json
data/datasets/replica_cad/rearrange/v2/minival/rearrange.json.gz
data/hab3_bench_assets/humanoids/female_0/
data/objects/ycb/configs/
```

Run the commands below from the repo root folder (`MRHAR/`). The scripts find
`data/` relative to the repo, wherever it is cloned.

---

## 2. Quick start

```bash
# one scripted morning (~21 min of household time, ~1 min to run)
python -m habitat_casas.run_routines            # add --video for mp4s
# -> outputs/routines/

# compare it with the real Cairo / Kyoto statistics (needs the real datasets)
python -m habitat_casas.compare outputs/routines \
    --cairo path/to/cairo.txt --kyoto path/to/kyoto_multiresident_clean.csv
# -> outputs/routines/comparison.md
```

## 3. Generate a multi-day dataset

```bash
python -m habitat_casas.run_days --days 10 --seed 0
# -> outputs/dataset/
```

| File | Content |
|---|---|
| `casas_cairo.txt` | Cairo raw format, with `R<n>_<Activity> begin/end` labels |
| `kyoto_format.csv` | same columns as `kyoto_multiresident_clean.csv` (`pair_id=day_kk`) |
| `casas_smrt.txt` | format read by the sMRT/GAMUT `parser.py` |
| `activities_gt.csv`, `activity_ids.csv` | ground-truth activity intervals and ids |
| `day_kk/sensor_events.csv` | `t, sensor_id, value, agent_ids, activities` (agent_ids = who triggered it) |
| `day_kk/sensor_layout.csv` | sensor positions |
| `days.csv` | summary per day |

Runs are reproducible for a given `--seed`.

## 4. Visualise

```bash
python -m habitat_casas.blueprint                      # outputs/blueprint.png/.pdf: floor plan + all sensors
python -m habitat_casas.replay outputs/dataset/day_09 --speed 10
                                                       # top-down replay video with live event log
python -m habitat_casas.play_human --agents 2          # drive a resident with the keyboard
```

`play_human` keys: W/S walk, A/D turn, SHIFT run, E open/close, TAB next part,
F pick/drop, X switch resident, R reset, ESC quit.

---

## 5. How it works

| File | Role |
|---|---|
| `env.py` | Builds a habitat env with N humanoids (`agent_0`, `agent_1`, ...), per-agent navigation action, navmesh rebuild |
| `activities.py` | Routine engine: steps `Walk`, `Open`, `Close`, `Pick`, `Place`, `Wait`, `Sync` grouped into `Activity` and `Routine` per resident |
| `scenario_apt.py` | The apartment: named places, sensor layout (motion zones, contacts, items), example routines |
| `days.py` | Random daily schedules from a fixed activity list (`LABELS`, stable ids) |
| `sensors.py` | Virtual sensors. `PIRModel` adds realistic noise: hold time, refractory period, missed pulses, false alarms |
| `export.py` | Writes Cairo / Kyoto / sMRT formats |
| `run_routines.py`, `run_days.py` | Entry points (one session / many days) |
| `compare.py`, `replay.py`, `blueprint.py`, `play_human.py` | Analysis and visualisation |

Loop per simulation step (120 Hz): routines issue navigation/interaction
actions -> habitat moves the humanoids -> `SensorSuite` checks each sensor
against the agents' positions and the furniture joint states -> state changes
are logged with the responsible agent id.

### Adding a sensor
Add an entry to `MOTION`, `CONTACTS` or `ITEMS` in `scenario_apt.py`
(position/zone in metres, link name for contacts), and a plain-language name to
`NAMES` in `blueprint.py`.

### Adding an activity
Add a label to `LABELS` in `days.py` (append at the end so existing ids stay
the same) and describe its steps in the day generator.

---

## 6. Known issues

- Furniture in ReplicaCAD is DYNAMIC, so `rebuild_navmesh(sim)` must be called after every `env.reset()` (the scripts already do this).
- Use navigation mode 0 with a point from `nav_point_near()`; mode 1 can silently "finish" without moving.
- Humanoids don't avoid each other (paths can overlap for under 1 s).
- Only the front door, fridge, kitchen drawers/cupboards, cabinet and chest of drawers can open; other doors are part of the static mesh.
- Sensors that never fire with the current routines: D007 (kitchen cupboard,
  no activity opens it) and M012 (east room, only false alarms).
- Speed: ~310 steps/s headless, ~110 with video, on an M1.
