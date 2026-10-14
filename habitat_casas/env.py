"""Build a RearrangeSim env with N humanoid agents sharing one scene.

Agents are named agent_0, agent_1, ... (habitat-lab derives the agent index
from the "agent_<i>_" prefix of each action name).
"""
import os
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from omegaconf import OmegaConf

from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.config.default_structured_configs import (
    AgentConfig,
    DatasetConfig,
    EnvironmentConfig,
    HabitatConfig,
    HumanoidJointActionConfig,
    OracleNavActionConfig,
    SimulatorConfig,
    TaskConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.env import Env
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavCoordinateAction,
)

# repo root = parent of this package, so data/ resolves wherever the repo is cloned
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMANOID_DIR = "data/hab3_bench_assets/humanoids/female_0"
HSSD_DIR = "data/hab3_bench_assets/hab3-hssd"
HSSD_DATASET = f"{HSSD_DIR}/hab3-hssd.scene_dataset_config.json"
HSSD_EPISODES = "data/hab3_bench_assets/episode_datasets/small_large.json.gz"

# ReplicaCAD apartment: has a kitchen and openable furniture (fridge, drawers,
# cupboards, door) for contact sensors. The HSSD scene above has neither.
REPLICA_DATASET = "data/replica_cad/replicaCAD.scene_dataset_config.json"
REPLICA_EPISODES = "data/datasets/replica_cad/rearrange/v2/minival/rearrange.json.gz"


@registry.register_task_action
class PerAgentOracleNavCoordinateAction(OracleNavCoordinateAction):  # type: ignore
    """OracleNavCoordinateAction with one humanoid controller per agent.

    The stock action caches a single controller on the task, so with two
    humanoids both nav actions would drive the same controller.
    """

    def lazy_inst_humanoid_controller(self, task, config):
        if not hasattr(task, "humanoid_controllers"):
            task.humanoid_controllers = {}
        if self._agent_index not in task.humanoid_controllers:
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            motion_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path
            ctrl = HumanoidRearrangeController(motion_path)
            ctrl.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controllers[self._agent_index] = ctrl
        return task.humanoid_controllers[self._agent_index]


def humanoid_agent_config(render: bool) -> AgentConfig:
    cfg = AgentConfig()
    cfg.articulated_agent_urdf = f"{HUMANOID_DIR}/female_0.urdf"
    cfg.articulated_agent_type = "KinematicHumanoid"
    cfg.motion_data_path = f"{HUMANOID_DIR}/female_0_motion_data_smplx.pkl"
    cfg.sim_sensors = {"third_rgb": ThirdRGBSensorConfig()} if render else {}
    return cfg


def make_multi_humanoid_env(
    num_agents: int = 2,
    render: bool = False,
    scene_dataset: str = REPLICA_DATASET,
    episodes: str = REPLICA_EPISODES,
    seed: int = 0,
    walk_speed: float = 1.2,
    turn_speed: float = 2.0,
    max_steps: int = 10_000_000,
) -> Env:
    """walk_speed in m/s, turn_speed in rad/s; sim time = steps / ctrl_freq (120 Hz)."""
    os.chdir(REPO_ROOT)  # all data/ paths are relative to the repo root

    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    sim_cfg.habitat_sim_v0.enable_hbao = False
    sim_cfg.habitat_sim_v0.enable_physics = True
    sim_cfg.scene_dataset = scene_dataset  # scene itself comes from the episode
    sim_cfg.additional_object_paths = ["data/objects/ycb/configs/"]
    sim = OmegaConf.create(sim_cfg)
    sim.agents = {
        f"agent_{i}": humanoid_agent_config(render) for i in range(num_agents)
    }
    sim.agents_order = list(sim.agents.keys())

    task = TaskConfig(type="RearrangeEmptyTask-v0")
    task.min_distance_start_agents = 2.0
    task.actions = {}
    for i in range(num_agents):
        # Direct pose control, used for reaching/standing during interactions.
        task.actions[f"agent_{i}_joint"] = HumanoidJointActionConfig()
    task.actions.update({
        f"agent_{i}_nav": OracleNavActionConfig(
            type="PerAgentOracleNavCoordinateAction",
            motion_control="human_joints",
            spawn_max_dist_to_obj=1.0,
            lin_speed=walk_speed,
            ang_speed=turn_speed,
        )
        for i in range(num_agents)
    })

    cfg = HabitatConfig()
    cfg.seed = seed
    cfg.environment = EnvironmentConfig(max_episode_steps=max_steps)
    cfg.task = task
    cfg.dataset = DatasetConfig(type="RearrangeDataset-v0", data_path=episodes)
    cfg.simulator = sim
    cfg.simulator.seed = seed
    return Env(OmegaConf.create(cfg))


class MultiHumanoidDriver:
    """Steps all agents at once, each walking toward its own goal (or idling)."""

    def __init__(self, env: Env):
        self.env = env
        self.sim = env.sim
        self.n = len(self.sim.agents_mgr)
        self.goals: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * self.n

    def agent(self, i: int):
        return self.sim.agents_mgr[i].articulated_agent

    def position(self, i: int) -> np.ndarray:
        return np.array(self.agent(i).base_pos)

    def set_goal(self, i: int, look_at: Optional[np.ndarray]) -> bool:
        """Walk agent i to a floor spot next to look_at, then face it.

        Returns False (and leaves the agent idle) if look_at is unreachable.
        """
        if look_at is None:
            self.goals[i] = None
            return True
        stand = nav_point_near(self.sim, look_at)
        if stand is None:
            self.goals[i] = None
            return False
        self.goals[i] = (stand, np.asarray(look_at, float))
        return True

    def arrived(self, i: int) -> bool:
        return self.goals[i] is None

    def controller(self, i: int) -> HumanoidRearrangeController:
        return self.env.task.humanoid_controllers[i]

    def step(self, poses: Optional[Dict[int, List[float]]] = None):
        """Advance one sim step. poses: {agent index: joint pose} for agents
        that are not walking (e.g. reaching toward a drawer)."""
        names, args = [], {}
        for i, pose in (poses or {}).items():
            names.append(f"agent_{i}_joint")
            args[f"agent_{i}_human_joints_trans"] = pose
        for i, goal in enumerate(self.goals):
            if goal is None:
                continue
            stand, look_at = goal
            names.append(f"agent_{i}_nav")
            # mode 0: explicit standing spot. Mode 1 samples one itself and
            # silently falls back to the agent's current position on failure.
            args[f"agent_{i}_oracle_nav_coord_action"] = stand
            args[f"agent_{i}_oracle_nav_lookat_action"] = look_at
            args[f"agent_{i}_mode"] = 0
        obs = self.env.step({"action": tuple(names), "action_args": args})
        for i in range(self.n):
            if self.goals[i] is not None and self.env.task.actions[f"agent_{i}_nav"].skill_done:
                self.goals[i] = None
        return obs


def nav_point_near(sim, pos, max_dist: float = 1.5) -> Optional[np.ndarray]:
    """Closest point on the main floor navmesh island within max_dist (horizontal) of pos."""
    snap = np.array(sim.pathfinder.snap_point(mn.Vector3(*pos), sim.largest_island_idx))
    if np.isnan(snap).any():
        return None
    d = snap - np.asarray(pos, float)
    return snap if np.hypot(d[0], d[2]) <= max_dist else None


def rebuild_navmesh(sim, open_doors=("door",)) -> None:
    """Make furniture block the navmesh.

    RearrangeSim only bakes STATIC objects into the navmesh, and ReplicaCAD's
    articulated furniture (fridge, counter, dresser, doors) is DYNAMIC, so
    agents would walk through it. We make it STATIC (joints can still be set
    kinematically) and rebuild with doors open so doorways stay passable.
    """
    from habitat_sim.physics import MotionType

    aom = sim.get_articulated_object_manager()
    doors = []
    for handle, ao in aom.get_objects_by_handle_substring().items():
        if any(a.articulated_agent.sim_obj.handle == handle for a in sim.agents_mgr):
            continue
        ao.motion_type = MotionType.STATIC
        if any(d in handle for d in open_doors):
            lo, hi = ao.joint_position_limits
            saved = list(ao.joint_positions)
            ao.joint_positions = [h if abs(h) > abs(l) else l for l, h in zip(lo, hi)]
            doors.append((ao, saved))
    sim._recompute_navmesh()
    for ao, saved in doors:
        ao.joint_positions = saved


def object_positions(sim) -> Dict[str, mn.Vector3]:
    rom = sim.get_rigid_object_manager()
    return {
        rom.get_object_by_id(i).handle: rom.get_object_by_id(i).translation
        for i in sim.scene_obj_ids
    }
