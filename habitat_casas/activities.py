"""Scripted daily routines for several humanoids sharing one home.

A routine is a list of Activity objects; each activity is a list of primitive
steps executed in order:

    Walk(place)                       walk to a named place and face it
    Wait(seconds)                     stay put (time is fast-forwarded when
                                      every agent is waiting)
    Open(furniture, link) / Close(..) animate a door/drawer joint while reaching
    Pick(object)                      take an object into the right hand
    Place(object, place)              put the carried object at a place
    Sync(tag)                         wait until every resident whose routine
                                      has this tag reaches it (joint activities)

The runner records the ground truth that the virtual sensors (step 4) and the
CASAS log (step 5) are built from: per-step positions with the current
activity, activity intervals, and interaction events, all tagged by agent.
"""
import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import magnum as mn
import numpy as np

from habitat_casas.env import MultiHumanoidDriver, nav_point_near
from habitat_sim.physics import MotionType


# ---------------------------------------------------------------- primitives
@dataclass
class Walk:
    place: str


@dataclass
class Wait:
    seconds: float


@dataclass
class Open:
    furniture: str  # articulated object handle prefix, e.g. "fridge"
    link: str  # link name, e.g. "top_door"
    seconds: float = 1.0


@dataclass
class Close(Open):
    pass


@dataclass
class Pick:
    obj: str  # rigid object handle, e.g. "018_plum_:0000"
    seconds: float = 1.0


@dataclass
class Place:
    obj: str
    place: str
    seconds: float = 1.0


@dataclass
class Sync:
    tag: str


Step = Union[Walk, Wait, Open, Close, Pick, Place, Sync]


@dataclass
class Activity:
    label: str
    steps: List[Step]


@dataclass
class Routine:
    agent: int
    start_place: str
    activities: List[Activity]


# -------------------------------------------------------------- the home
@dataclass
class Home:
    """Named places (points to walk to and face) in a scene."""

    places: Dict[str, Tuple[float, float, float]]
    # Optional explicit standing spots, for places where snapping the look-at
    # point to the navmesh lands on the wrong side of a wall.
    stand: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def look_at(self, place: str) -> np.ndarray:
        return np.array(self.places[place], float)

    def stand_point(self, sim, place: str) -> np.ndarray:
        if place in self.stand:
            return np.array(self.stand[place], float)
        p = nav_point_near(sim, self.look_at(place))
        if p is None:
            raise ValueError(f"place {place!r} is not reachable from the floor")
        return p


# ------------------------------------------------------------ helpers
def find_link(sim, furniture: str, link: str):
    """Return (articulated object, joint position index, open value)."""
    aom = sim.get_articulated_object_manager()
    matches = list(aom.get_objects_by_handle_substring(furniture).values())
    if not matches:
        raise KeyError(f"no articulated object matching {furniture!r}")
    ao = matches[0]
    for lid in ao.get_link_ids():
        if ao.get_link_name(lid) == link:
            j = ao.get_link_joint_pos_offset(lid)
            lo, hi = ao.joint_position_limits
            # "open" is whichever limit is not the closed (0) position
            return ao, j, hi[j] if abs(hi[j]) > abs(lo[j]) else lo[j]
    raise KeyError(f"{furniture!r} has no link {link!r}")


def close_all_furniture(sim) -> None:
    """Start every episode with all doors and drawers closed."""
    aom = sim.get_articulated_object_manager()
    for h, ao in aom.get_objects_by_handle_substring().items():
        if "female" in h or ao.num_links == 0 or not np.isfinite(ao.joint_position_limits[0]).all():
            continue
        ao.joint_positions = [0.0] * len(ao.joint_positions)


# ------------------------------------------------------------ runner
class _AgentState:
    def __init__(self, routine: Routine):
        self.routine = routine
        self.queue: List[Tuple[str, Step]] = [
            (a.label, s) for a in routine.activities for s in a.steps
        ]
        self.activity: str = ""
        self.activity_start: float = 0.0
        self.step: Optional[Step] = None
        self.step_start: float = 0.0
        self.step_end: float = 0.0
        self.carrying: Optional[str] = None
        self.done = False
        # for Open/Close animations
        self.anim: Optional[Tuple] = None


class RoutineRunner:
    """Runs several agents' routines concurrently in one sim.

    Sim time advances 1/ctrl_freq per physics step while anyone is moving;
    when all agents are only waiting, the clock jumps to the next wake-up.
    """

    def __init__(self, env, home: Home, routines: Sequence[Routine]):
        self.env = env
        self.sim = env.sim
        self.home = home
        self.driver = MultiHumanoidDriver(env)
        self.dt = 1.0 / self.sim.ctrl_freq
        self.t = 0.0
        self.n_steps = 0
        self.agents = {r.agent: _AgentState(r) for r in routines}
        self.rom = self.sim.get_rigid_object_manager()

        # ground-truth records
        self.trajectory: List[list] = []
        self.activity_log: List[list] = []
        self.interactions: List[list] = []
        # called after every tick as cb(runner); step 4 hooks sensors here
        self.callbacks: List[Callable] = []
        self.frame_cb: Optional[Callable] = None

        close_all_furniture(self.sim)
        for i, st in self.agents.items():
            a = self.driver.agent(i)
            a.base_pos = mn.Vector3(*self.home.stand_point(self.sim, st.routine.start_place))

    # ---- primitive execution
    def _start_step(self, i: int, st: _AgentState) -> None:
        label, step = st.queue.pop(0)
        # waiting at a Sync still belongs to the previous activity; the new
        # label starts with the first real step after the partner arrives
        if label != st.activity and not isinstance(step, Sync):
            self._end_activity(i, st)
            st.activity, st.activity_start = label, self.t
        st.step, st.step_start, st.anim = step, self.t, None

        if isinstance(step, Walk):
            stand = self.home.stand_point(self.sim, step.place)
            self.driver.goals[i] = (stand, self.home.look_at(step.place))
            st.step_end = np.inf  # ends when the nav action reports arrival
        elif isinstance(step, Wait):
            st.step_end = self.t + step.seconds
        elif isinstance(step, (Open, Close)):
            ao, j, open_val = find_link(self.sim, step.furniture, step.link)
            start = ao.joint_positions[j]
            goal = 0.0 if isinstance(step, Close) else open_val
            st.anim = (ao, j, start, goal)
            st.step_end = self.t + step.seconds
        elif isinstance(step, (Pick, Place)):
            st.step_end = self.t + step.seconds  # reach; effect applied on finish
        elif isinstance(step, Sync):
            st.step_end = np.inf  # released by _sync_released

    def _finish_step(self, i: int, st: _AgentState) -> None:
        step = st.step
        if isinstance(step, (Open, Close)):
            self._interaction(i, "close" if isinstance(step, Close) else "open",
                              f"{step.furniture}/{step.link}")
        elif isinstance(step, Pick):
            obj = self.rom.get_object_by_handle(step.obj)
            obj.motion_type = MotionType.KINEMATIC
            st.carrying = step.obj
            self._interaction(i, "pick", step.obj)
        elif isinstance(step, Place):
            obj = self.rom.get_object_by_handle(step.obj)
            obj.translation = mn.Vector3(*self.home.look_at(step.place))
            obj.motion_type = MotionType.DYNAMIC
            st.carrying = None
            self._interaction(i, "place", f"{step.obj}@{step.place}")
        st.step = None

    def _end_activity(self, i: int, st: _AgentState) -> None:
        if st.activity:
            self.activity_log.append([f"agent_{i}", st.activity,
                                      round(st.activity_start, 3), round(self.t, 3)])

    def _interaction(self, i: int, kind: str, target: str) -> None:
        self.interactions.append([round(self.t, 3), f"agent_{i}", self.agents[i].activity, kind, target])

    def _step_finished(self, i: int, st: _AgentState) -> bool:
        if st.step is None:
            return True
        if isinstance(st.step, Walk):
            return self.driver.goals[i] is None
        if isinstance(st.step, Sync):
            return self._sync_released(st.step.tag)
        return self.t >= st.step_end - 1e-9

    def _sync_released(self, tag: str) -> bool:
        """True once every agent that has `tag` in its routine is waiting at it
        (or has already passed it)."""
        for j, other in self.agents.items():
            pending = [s for _, s in other.queue if isinstance(s, Sync) and s.tag == tag]
            at_it = isinstance(other.step, Sync) and other.step.tag == tag
            if pending and not at_it:
                return False
        return True

    def _needs_physics(self, i: int, st: _AgentState) -> bool:
        """Is this agent doing something that must be simulated step by step?"""
        return isinstance(st.step, (Walk, Open, Close, Pick, Place)) and not self._step_finished(i, st)

    def _reach_pose(self, i: int, target: np.ndarray) -> List[float]:
        ctrl = self.driver.controller(i)
        ctrl.obj_transform_base = self.driver.agent(i).base_transformation
        ctrl.calculate_reach_pose(mn.Vector3(*target), index_hand=0)
        return ctrl.get_pose()

    def _stand_pose(self, i: int) -> List[float]:
        ctrl = self.driver.controller(i)
        ctrl.obj_transform_base = self.driver.agent(i).base_transformation
        ctrl.calculate_stop_pose()
        return ctrl.get_pose()

    # ---- main loop
    def tick(self) -> bool:
        """Advance the world by one physics step or one time jump.
        Returns False when every routine is finished."""
        for i, st in self.agents.items():
            while not st.done and self._step_finished(i, st):
                if st.step is not None:
                    self._finish_step(i, st)
                if st.queue:
                    self._start_step(i, st)
                else:
                    self._end_activity(i, st)
                    st.activity, st.step, st.done = "", None, True

        if all(st.done for st in self.agents.values()):
            return False

        active = [i for i, st in self.agents.items() if self._needs_physics(i, st)]
        if not active:
            # everyone is waiting: jump to the earliest wake-up
            wake = min(st.step_end for st in self.agents.values() if not st.done)
            if not np.isfinite(wake):
                raise RuntimeError("all agents blocked on Sync steps (mismatched tags?)")
            self.t = wake
            self._record()
            return True

        poses = {}
        for i in active:
            st = self.agents[i]
            if isinstance(st.step, (Open, Close)):
                ao, j, start, goal = st.anim
                frac = min(1.0, (self.t - st.step_start + self.dt) / st.step.seconds)
                jp = list(ao.joint_positions)
                jp[j] = start + frac * (goal - start)
                ao.joint_positions = jp
                link_pos = np.array(ao.get_link_scene_node(
                    [l for l in ao.get_link_ids() if ao.get_link_joint_pos_offset(l) == j][0]
                ).absolute_translation)
                poses[i] = self._reach_pose(i, link_pos)
            elif isinstance(st.step, Pick):
                obj = self.rom.get_object_by_handle(st.step.obj)
                poses[i] = self._reach_pose(i, np.array(obj.translation))
            elif isinstance(st.step, Place):
                poses[i] = self._reach_pose(i, self.home.look_at(st.step.place))
        obs = self.driver.step(poses)
        self.t += self.dt
        self.n_steps += 1

        # carried objects follow the right hand
        for i, st in self.agents.items():
            if st.carrying:
                obj = self.rom.get_object_by_handle(st.carrying)
                obj.translation = self.driver.agent(i).ee_transform(0).translation
        if self.frame_cb:
            self.frame_cb(self, obs)
        self._record()
        return True

    def _record(self) -> None:
        for i, st in self.agents.items():
            p = self.driver.position(i)
            self.trajectory.append([self.n_steps, round(self.t, 3), f"agent_{i}",
                                    *np.round(p, 3), st.activity])
        for cb in self.callbacks:
            cb(self)

    def run(self, max_time: float = 6 * 3600) -> None:
        while self.t < max_time and self.tick():
            pass

    # ---- output
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        _write(os.path.join(out_dir, "trajectories.csv"),
               ["step", "t", "agent_id", "x", "y", "z", "activity"], self.trajectory)
        _write(os.path.join(out_dir, "activities.csv"),
               ["agent_id", "activity", "start_t", "end_t"], self.activity_log)
        _write(os.path.join(out_dir, "interactions.csv"),
               ["t", "agent_id", "activity", "type", "target"], self.interactions)


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
