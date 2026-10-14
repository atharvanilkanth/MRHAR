# CLAUDE.md

## Thesis context

This repo is part of an MS thesis on **multi-resident smart-home human activity recognition (HAR)**.

Goal (set by the advisor): use Habitat to simulate ambient smart-home sensing with multiple agents.
- Add virtual ambient sensors to scenes:
  - motion/PIR zones
  - door/drawer contact sensors
  - item sensors (object picked up or moved)
- Run multiple agents (humanoids and/or robot) acting in the same scene and record what the sensors detect.
- Output **CASAS-style event logs**: `timestamp, sensor_id, value`, plus the **ground-truth agent ID** for each event.
- Logs should be comparable to the CASAS **Cairo** and **Kyoto** multi-resident datasets.

## Environment

- Machine: MacBook Air M1, no NVIDIA GPU (no CUDA, so keep runs CPU/Metal-friendly)
- Python env: micromamba `base` (Python 3.9.18, habitat-sim 0.3.3)
- **Never use the `AgentSense` env.**
- Data lives in `./data` (ReplicaCAD, hab_fetch, ycb, rearrange_pick, test scenes)

## Known gotchas

- `sim.reset()` fails when no episode is loaded. Use `sim.reset_agent(0)` instead.
- Notebooks must start with `os.chdir(os.path.expanduser("~/habitat-lab"))` so relative `data/` paths resolve.
- Humanoid assets: use `data/hab3_bench_assets/humanoids/female_0/` (URDF and motion `.pkl`). `data/humanoids` is not downloaded.
- `HumanoidRearrangeController` walk and `OracleNavCoordinateAction` (`motion_control="human_joints"`) both work on the hab3-hssd scene, at about 170 steps/s with rendering.

## Multi-agent notes (`habitat_casas/`)

- Agents must be named `agent_0`, `agent_1`, ...; habitat-lab gets the agent index from the `agent_<i>_` prefix of action names, and action args use the same prefix.
- The stock `OracleNavCoordinateAction` shares one humanoid controller across agents (stored on the task). Use `PerAgentOracleNavCoordinateAction` from `habitat_casas/env.py`.
- Humanoid speed: nav `lin_speed` is m/s and `ang_speed` is rad/s at `ctrl_freq` = 120 Hz. Defaults (10) are about 8x too fast; we use 1.2 m/s and 2 rad/s. Sim time = step / 120.
- Nav mode 1 (lookat only) samples a standing spot and, on failure, silently returns the agent's current position, so the nav "finishes" instantly. Use mode 0 with an explicit spot from `nav_point_near()` (snaps to the largest navmesh island).
- Raise `environment.max_episode_steps` (default 1000), otherwise you get "Episode over, call reset".
- Humanoids don't avoid each other: crossing paths overlap for under 1 s (checked), never while standing at spots.
- `env.reset()` with a single-episode list (`env.episodes = [...]`) restores the apartment each day, but furniture goes back to DYNAMIC: call `rebuild_navmesh` after every reset.
- Spawning uses global RNGs: seed `random` and `np.random` for reproducible runs.
- Speed: about 310 steps/s headless, about 110 with two third-person cameras.

## Scene notes

- Default scene is now ReplicaCAD (`v3_sc3_staging_16`, episode from `data/datasets/replica_cad/rearrange/v2/minival/rearrange.json.gz`): kitchen + openable furniture (fridge, kitchen_counter drawers, kitchenCupboard, chestOfDrawers, cabinet, door2). The hab3-hssd scene has no kitchen and no articulated objects.
- Call `rebuild_navmesh(sim)` after `env.reset()`: ReplicaCAD furniture is DYNAMIC, so it isn't in the navmesh and agents walk through counters. It is made STATIC and the navmesh rebuilt with doors open.
- Episodes start with some furniture open (the fridge top door); `RoutineRunner` closes everything first.

## External thesis data/code (outside this repo, read-only from here)

- Real datasets: `~/Desktop/FEAT/dataset/cairo.txt` (CASAS Cairo raw, tab separated, M + T sensors, labels only on begin/end lines) and `~/Desktop/FEAT/dataset/kyoto_multiresident_clean.csv` (26 sessions; task_field_1/3/5 = resident, 2/4/6 = activity id).
- sMRT/GAMUT tracker: `~/Desktop/FEAT/smrt_impl` (own `.venv`; loaders in `smrt_gamut/data/parser.py`, `kyoto_csv.py`).
- 10-day dataset (seeds 0-9, 12.2 h, 9,993 events) vs real: PIR ON median 3.3 s (Cairo 3.1, Kyoto 3.5), 838 events/h (531 / 1301), M/D/I 97.7/2.0/0.3% (Kyoto 93.7/5.6/0.8). Still off: two-resident events 3.2% of all events, 0.2% of active events (Kyoto 16.8%; our attribution is physical, i.e. who triggered it), median gap 2.4 s (real about 1 s).

## References

- Custom sensors and measures: `examples/register_new_sensors_and_measures.py`
- Tutorials: `examples/tutorials/articulated_agents_tutorial.ipynb`, `examples/tutorials/humanoids_tutorial.ipynb`

## Progress

Done:
- Install
- `examples/example.py`
- `interactive_play`
- `articulated_agents_tutorial`
- `humanoids_tutorial` (runs end to end with local `female_0`; SeqPose cells skip because `data/humanoids` is not downloaded)
- Two-agent scene: `habitat_casas/env.py` + `python -m habitat_casas.run_two_agents [--video]` -> `outputs/two_agents/` (trajectories.csv, topdown.png, per-agent mp4)

- Scripted routines: `habitat_casas/activities.py` (engine), `habitat_casas/scenario_apt.py` (places + two routines), `python -m habitat_casas.run_routines [--video]` -> `outputs/routines/` (trajectories/activities/interactions CSVs, timeline.png, topdown.png, mp4s). About 21 min of household time in about 1 min headless.

- Virtual sensors: `habitat_casas/sensors.py` (PIR zones with 2.5 s timeout + Poisson fidget retriggers, door/drawer contacts, item pads), layout in `scenario_apt.py` (12 M, 7 D, 2 I). The run writes `sensor_events.csv` (t, sensor_id, value, agent_ids, activities), `sensor_layout.csv`, `sensor_events.png`. Validated: every motion ON is credited to an agent who was inside that zone.

- CASAS export: `habitat_casas/export.py` (runs automatically) -> `casas_cairo.txt` (Cairo raw format + `R<n>_<Activity> begin/end`), `casas_smrt.txt` (format for `smrt_impl` `parser.py`), `kyoto_format.csv` (same columns as `kyoto_multiresident_clean.csv`), `activity_ids.csv`, `activities_gt.csv`. Both sMRT loaders read these files unmodified (verified).
- Comparison: `python -m habitat_casas.compare outputs/routines` -> `comparison.md`.

- Multi-day dataset: `python -m habitat_casas.run_days --days N [--seed S]` -> `outputs/dataset/` (per-day folders + combined `kyoto_format.csv` with `pair_id=day_kk`, `casas_cairo.txt`, `casas_smrt.txt`, `activities_gt.csv`, `activity_ids.csv`, `days.csv`). Random days come from `habitat_casas/days.py` (fixed `LABELS` -> stable activity ids). Joint activities (cook/eat/couch) are aligned with `Sync(tag)` steps; R2 uses the `_2` spots.
- Replay video: `python -m habitat_casas.replay outputs/dataset/day_09 [--speed 10] [--from-min A --to-min B]` -> `replay_<speed>x.mp4` (top-down floor plan, residents, sensors lighting up, live CASAS event log). Floor plan cached in `outputs/apt_map.npz`. Full day at 10x takes about 3.5 min to render.
- Blueprint: `python -m habitat_casas.blueprint` -> `outputs/blueprint.png`/`.pdf` (plan + plain-language list of all 29 sensors) and `outputs/blueprint_plan.png` (plan only). Sensor plain names live in `NAMES` there.
- Sensor realism (`PIRModel` in sensors.py): lognormal hold (median 3 s), pulse cap 3-8 s, refractory 0.3-1.2 s, 2% missed pulses, 0.3 false alarms/sensor/h, fidgets; area PIRs M013-M015 overlap the spot sensors; D008/D009 added.
- Keyboard play as a humanoid: `python -m habitat_casas.play_human [--agents 1|2]` (pygame window: chase camera + live sensor log; W/S walk, A/D turn, SHIFT run, E open/close the part you face, TAB next part, F pick/drop item, X switch resident, R reset, ESC quit -> `outputs/play/sensor_events.csv`). Only 26 articulated parts open (front door door2, fridge, counter drawers, cupboards, cabinet, chest of drawers); bedroom and other doors are part of the stage mesh and cannot open. With 1 agent habitat drops the `agent_0_` prefix from action arg names. Movement: snap to the navmesh from floor height on `sim.largest_island_idx` (from hip height it snaps to stair islands) and follow navmesh y (stairs climb to the ~1.1 m landing; the upper floor is a disconnected island). The navmesh is baked with doors open, so play_human blocks closed doors by hand (`_blocked_by_door`). Stock `interactive_play.py --control-humanoid` needs `data/humanoids` (not downloaded); the stock humanoid third camera ends up above the ReplicaCAD ceiling, so play_human sets the camera itself (`_chase_camera`).
