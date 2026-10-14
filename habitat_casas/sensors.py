"""Virtual ambient sensors driven by the RoutineRunner's world state.

Sensor models (CASAS-like):

  MotionZone  "M0xx"  ceiling PIR covering a floor rectangle (zones may
                      overlap, e.g. room-wide "area" sensors over spot sensors).
                      Fires ON when a resident inside moves (walking, reaching);
                      see PIRModel for hold time, pulse length, refractory
                      period, missed detections and false alarms. Residents who
                      are still (eating, watching TV) fidget at random and
                      retrigger it, as real PIRs do.
  Contact     "D0xx"  magnetic contact on a door/drawer link: OPEN/CLOSE when
                      the joint passes `open_frac` of its range.
  Item        "I0xx"  pressure pad under an object: ABSENT when the object is
                      more than `radius` from its starting spot, else PRESENT.

Every event carries ground truth: which agent(s) caused it and what activity
they were doing. Agents are never visible to the sensors themselves.
"""
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from habitat_casas.activities import Close, Open, RoutineRunner, find_link


@dataclass
class MotionZone:
    sensor_id: str
    name: str
    x: Tuple[float, float]  # (min, max)
    z: Tuple[float, float]

    def contains(self, p) -> bool:
        return self.x[0] <= p[0] <= self.x[1] and self.z[0] <= p[2] <= self.z[1]


@dataclass
class PIRModel:
    """Timing/noise of the motion sensors. Defaults tuned toward CASAS Cairo/Kyoto
    (median ON about 3 s, bursts of short pulses while someone keeps moving)."""

    hold_median: float = 3.0  # s of no motion before OFF (lognormal per pulse)
    hold_sigma: float = 0.35
    max_on: Tuple[float, float] = (3.0, 8.0)  # a pulse never lasts longer (s)
    refractory: Tuple[float, float] = (0.3, 1.2)  # dead time after OFF (s)
    miss_prob: float = 0.02  # a pulse that should start is missed
    false_per_hour: float = 0.3  # spurious triggers per sensor per hour
    fidget_rate: float = 1 / 30  # retriggers/s of a still resident


@dataclass
class Contact:
    sensor_id: str
    furniture: str
    link: str
    open_frac: float = 0.2


@dataclass
class Item:
    sensor_id: str
    obj: str
    radius: float = 0.1


@dataclass
class _PIRState:
    on: bool = False
    off_at: float = -np.inf
    max_off: float = -np.inf
    hold: float = 0.0
    quiet_until: float = -np.inf
    agents: set = field(default_factory=set)


class SensorSuite:
    def __init__(
        self,
        runner: RoutineRunner,
        motion: Sequence[MotionZone],
        contacts: Sequence[Contact],
        items: Sequence[Item],
        pir: PIRModel = PIRModel(),
        seed: int = 0,
    ):
        self.runner = runner
        self.motion = list(motion)
        self.contacts = list(contacts)
        self.items = list(items)
        self.pir = pir
        self.rng = np.random.default_rng(seed)
        self.events: List[list] = []

        sim = runner.sim
        self._pir = {m.sensor_id: _PIRState() for m in self.motion}
        self._contact = {}
        for c in self.contacts:
            ao, j, open_val = find_link(sim, c.furniture, c.link)
            self._contact[c.sensor_id] = [ao, j, abs(open_val) * c.open_frac, None]
        rom = sim.get_rigid_object_manager()
        self._item = {}
        for it in self.items:
            obj = rom.get_object_by_handle(it.obj)
            self._item[it.sensor_id] = [obj, np.array(obj.translation), None]

        self._t_last = runner.t
        runner.callbacks.append(self.update)
        self.update(runner)  # initial states (no events for the start state)

    # ------------------------------------------------------------ helpers
    def _emit(self, t: float, sensor_id: str, value: str, agents) -> None:
        agents = sorted(agents)
        acts = [self.runner.agents[i].activity for i in agents]
        self.events.append([
            round(float(t), 3), sensor_id, value,
            ";".join(f"agent_{i}" for i in agents) or "none",
            ";".join(a or "none" for a in acts) or "none",
        ])

    def _nearest_agent(self, pos) -> int:
        d = {i: np.linalg.norm((self.runner.driver.position(i) - pos)[[0, 2]])
             for i in self.runner.agents}
        return min(d, key=d.get)

    def _actor_for_link(self, c: Contact, pos) -> int:
        for i, st in self.runner.agents.items():
            s = st.step
            if isinstance(s, (Open, Close)) and s.furniture == c.furniture and s.link == c.link:
                return i
        return self._nearest_agent(pos)

    # ------------------------------------------------------------ update
    def update(self, runner: RoutineRunner) -> None:
        t0, t1 = self._t_last, runner.t
        self._t_last = t1
        pos = {i: runner.driver.position(i) for i in runner.agents}
        moving = {i: runner._needs_physics(i, st) for i, st in runner.agents.items()}
        self._update_motion(t0, t1, pos, moving)
        self._update_contacts(t1)
        self._update_items(t1)

    def _update_motion(self, t0, t1, pos, moving) -> None:
        pm, dt = self.pir, t1 - t0
        for m in self.motion:
            inside = [i for i in pos if m.contains(pos[i])]
            triggers = []  # (time, agent or None for a false alarm)
            if dt > 0:
                for i in inside:
                    if not moving[i]:
                        k = self.rng.poisson(pm.fidget_rate * dt)
                        triggers += [(t, i) for t in self.rng.uniform(t0, t1, k)]
                k = self.rng.poisson(pm.false_per_hour / 3600 * dt)
                triggers += [(t, None) for t in self.rng.uniform(t0, t1, k)]
            triggers.sort(key=lambda x: x[0])
            for t, i in triggers:
                self._pir_motion(m.sensor_id, t, [] if i is None else [i])
            movers = [i for i in inside if moving[i]]
            if movers:  # everyone moving in the zone right now shares the trigger
                self._pir_motion(m.sensor_id, t1, movers)
            self._pir_expire(m.sensor_id, t1)

    def _pir_motion(self, sid: str, t: float, agents: List[int]) -> None:
        st, pm = self._pir[sid], self.pir
        self._pir_expire(sid, t)
        if not st.on:
            if t < st.quiet_until:
                return
            hold = float(np.clip(pm.hold_median * self.rng.lognormal(0, pm.hold_sigma), 1.0, 8.0))
            if self.rng.random() < pm.miss_prob:
                st.quiet_until = t + hold  # missed: stays silent for a while
                return
            st.on, st.hold, st.agents = True, hold, set()
            st.off_at = t + hold
            st.max_off = t + self.rng.uniform(*pm.max_on)
            self._emit(t, sid, "ON", agents)
        else:
            st.off_at = min(st.max_off, max(st.off_at, t + st.hold))
        st.agents.update(agents)

    def _pir_expire(self, sid: str, t: float) -> None:
        st = self._pir[sid]
        if st.on and st.off_at <= t:
            st.on = False
            st.quiet_until = st.off_at + self.rng.uniform(*self.pir.refractory)
            self._emit(st.off_at, sid, "OFF", st.agents)

    def _update_contacts(self, t) -> None:
        for c in self.contacts:
            ao, j, thresh, was_open = self._contact[c.sensor_id]
            is_open = abs(ao.joint_positions[j]) > thresh
            if was_open is not None and is_open != was_open:
                link_pos = np.array(ao.translation)
                self._emit(t, c.sensor_id, "OPEN" if is_open else "CLOSE",
                           [self._actor_for_link(c, link_pos)])
            self._contact[c.sensor_id][3] = is_open

    def _update_items(self, t) -> None:
        for it in self.items:
            obj, home, was_present = self._item[it.sensor_id]
            p = np.array(obj.translation)
            present = np.linalg.norm(p - home) <= it.radius
            if was_present is not None and present != was_present:
                carrier = [i for i, st in self.runner.agents.items() if st.carrying == it.obj]
                self._emit(t, it.sensor_id, "PRESENT" if present else "ABSENT",
                           carrier or [self._nearest_agent(p)])
            self._item[it.sensor_id][2] = present

    def finish(self) -> None:
        """Flush motion sensors that are still ON at the end of the run."""
        for sid in self._pir:
            self._pir_expire(sid, np.inf)
        self.events.sort(key=lambda e: e[0])

    # ------------------------------------------------------------ output
    def layout(self) -> List[list]:
        rows = [[m.sensor_id, "motion", m.name, *m.x, *m.z] for m in self.motion]
        rows += [[c.sensor_id, "contact", f"{c.furniture}/{c.link}", "", "", "", ""]
                 for c in self.contacts]
        rows += [[it.sensor_id, "item", it.obj, "", "", "", ""] for it in self.items]
        return rows

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "sensor_events.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "sensor_id", "value", "agent_ids", "activities"])
            w.writerows(self.events)
        with open(os.path.join(out_dir, "sensor_layout.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sensor_id", "type", "name", "x_min", "x_max", "z_min", "z_max"])
            w.writerows(self.layout())
