"""Walk a humanoid around the apartment with the keyboard and watch the
virtual sensors fire live.

    python -m habitat_casas.play_human            # 2 residents, X switches
    python -m habitat_casas.play_human --agents 1

Keys (click the window first):
  W / S        walk forward / walk back (turns around)
  A / D        turn left / right (hold with W to walk in a curve)
  SHIFT        walk faster
  E            open/close the door/drawer you face (shown in the panel). Only
               the front door, fridge, drawers and cupboards open: the other
               doors (bedrooms) are part of the walls in this scene
  TAB          next door/drawer in front of you (e.g. fridge top vs bottom)
  F            pick up / put down the nearest sensed item (within 1.2 m)
  X            switch resident (with --agents 2)
  R            new episode (residents respawn, sensors restart)
  ESC          quit (events are saved to outputs/play/sensor_events.csv)

The stock examples/interactive_play.py --control-humanoid needs data/humanoids,
which is not downloaded; this uses the local female_0 instead.
"""
import argparse
import os

import magnum as mn
import numpy as np
import habitat_sim
import pygame

from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat_casas.activities import close_all_furniture
from habitat_casas.env import REPO_ROOT, make_multi_humanoid_env, rebuild_navmesh
from habitat_casas.run_days import EPISODE_ID
from habitat_casas.scenario_apt import CONTACTS, ITEMS, MOTION
from habitat_casas.sensors import SensorSuite

FPS = 30
WALK, RUN, TURN = 1.2, 2.4, 2.0  # m/s, m/s, rad/s
CAM = 512
PANEL = 460
COLORS = {"ON": (255, 209, 102), "OFF": (120, 130, 150), "OPEN": (124, 227, 139),
          "CLOSE": (90, 160, 100), "ABSENT": (255, 143, 163), "PRESENT": (200, 110, 130)}


class _Resident:
    """The bits of RoutineRunner's per-agent state that SensorSuite reads."""

    def __init__(self):
        self.activity = "manual"
        self.carrying = None
        self.step = None


class _Driver:
    def __init__(self, sim):
        self.sim = sim

    def position(self, i):
        return np.array(self.sim.agents_mgr[i].articulated_agent.base_pos)


class LiveRunner:
    """Stands in for RoutineRunner so SensorSuite can run on keyboard input."""

    def __init__(self, env, n):
        self.sim = env.sim
        self.t = 0.0
        self.callbacks = []
        self.agents = {i: _Resident() for i in range(n)}
        self.driver = _Driver(env.sim)
        self.moving = {i: False for i in range(n)}

    def _needs_physics(self, i, st):
        return self.moving[i]


class Player:
    def __init__(self, n_agents):
        self.n = n_agents
        self.env = make_multi_humanoid_env(num_agents=n_agents, render=True)
        self.env.episodes = [e for e in self.env.episodes if e.episode_id == EPISODE_ID]
        self.links = {}
        self.new_episode()

    # ------------------------------------------------------------ setup
    def new_episode(self):
        self.obs = self.env.reset()
        sim = self.env.sim
        rebuild_navmesh(sim)
        close_all_furniture(sim)
        self.ctrl, self.heading, self.root_off = [], [], []
        for i in range(self.n):
            agent = sim.agents_mgr[i].articulated_agent
            c = HumanoidRearrangeController(sim.habitat_config.agents[f"agent_{i}"].motion_data_path)
            c.set_framerate_for_linspeed(WALK, TURN, FPS)
            c.reset(agent.base_transformation)
            self.ctrl.append(c)
            floor = sim.pathfinder.snap_point(agent.base_pos, sim.largest_island_idx)
            self.root_off.append(float(c.obj_transform_base.translation.y - floor.y))
            f = c.prev_orientation  # start facing where the agent already faces
            self.heading.append(float(np.arctan2(-f.z, f.x)))
        # every openable part in the scene (not only the sensored ones);
        # doors drawn into the walls (e.g. bedrooms) are not articulated
        sensored = {(c.furniture, c.link): c.sensor_id for c in CONTACTS}
        self.links = []  # (label, ao, joint index, open value, closed center)
        for h, ao in sim.get_articulated_object_manager().get_objects_by_handle_substring().items():
            if "female" in h or ao.num_links == 0:
                continue
            lo, hi = ao.joint_position_limits
            for lid in ao.get_link_ids():
                if ao.get_link_num_joint_pos(lid) == 0:
                    continue
                j = ao.get_link_joint_pos_offset(lid)
                node = ao.get_link_scene_node(lid)
                center = np.array(node.absolute_transformation().transform_point(node.cumulative_bb.center()))
                name = f"{h.split('_:')[0]} {ao.get_link_name(lid)}"
                sid = next((v for (f, l), v in sensored.items() if f in h and l == ao.get_link_name(lid)), None)
                label = f"{name} ({sid})" if sid else f"{name} (no sensor)"
                self.links.append((label, ao, j, hi[j] if abs(hi[j]) > abs(lo[j]) else lo[j], center))
        self._cand_labels, self.pick = [], 0
        # closed footprint of each real door: the navmesh is baked with doors
        # open (rebuild_navmesh), so a closed door must be blocked by hand
        self.doors = []  # (ao, joint index, (x, z) end 1, (x, z) end 2)
        for label, ao, j, _, _ in self.links:
            if not ao.handle.startswith("door"):
                continue
            lid = next(l for l in ao.get_link_ids() if ao.get_link_joint_pos_offset(l) == j
                       and ao.get_link_num_joint_pos(l) > 0)
            node = ao.get_link_scene_node(lid)
            bb, T = node.cumulative_bb, node.absolute_transformation()
            corners = np.array([np.array(T.transform_point(mn.Vector3(x, y, z)))[[0, 2]]
                                for x in (bb.min.x, bb.max.x) for y in (bb.min.y, bb.max.y)
                                for z in (bb.min.z, bb.max.z)])
            dists = np.linalg.norm(corners[:, None] - corners[None], axis=-1)
            e1, e2 = np.unravel_index(dists.argmax(), dists.shape)
            self.doors.append((ao, j, corners[e1], corners[e2]))
        self.runner = LiveRunner(self.env, self.n)
        self.sensors = SensorSuite(self.runner, MOTION, CONTACTS, ITEMS, seed=0)
        self.active = 0
        self.speed = WALK
        self.status = "ready"

    # ------------------------------------------------------------ actions
    def _pos(self, i):
        return np.array(self.env.sim.agents_mgr[i].articulated_agent.base_pos)

    def _blocked_by_door(self, a, b) -> bool:
        """True if stepping a -> b walks into a closed door."""
        pa, pb = np.array([a.x, a.z]), np.array([b.x, b.z])
        for ao, j, e1, e2 in self.doors:
            if abs(ao.joint_positions[j]) > 0.3:  # open enough to pass
                continue
            da, db = _seg_dist(pa, e1, e2), _seg_dist(pb, e1, e2)
            crosses = _side(pa, e1, e2) * _side(pb, e1, e2) < 0 and min(da, db) < 0.5
            if crosses or (db < 0.3 and db < da):
                return True
        return False

    def candidates(self):
        """Openable parts in front of the active resident, best first."""
        p = self._pos(self.active)
        h = self.heading[self.active]
        fwd = np.array([np.cos(h), -np.sin(h)])
        scored = []
        for link in self.links:
            v = (link[4] - p)[[0, 2]]
            d = np.linalg.norm(v)
            if d > 1.8:
                continue
            facing = float(v @ fwd) / max(d, 1e-6)  # 1 = straight ahead
            if facing < 0.3:  # behind or far off to the side
                continue
            scored.append((d * (2.0 - facing), link))
        return [link for _, link in sorted(scored, key=lambda x: x[0])]

    def target_link(self):
        """The part E will toggle; TAB cycles through the candidates."""
        cands = self.candidates()
        labels = [c[0] for c in cands]
        if labels != self._cand_labels:  # walked somewhere else: start over
            self._cand_labels, self.pick = labels, 0
        return cands[self.pick % len(cands)] if cands else None

    def toggle_nearest(self):
        link = self.target_link()
        if link is None:
            self.status = "nothing openable in front of you"
            return
        label, ao, j, open_val, _ = link
        q = list(ao.joint_positions)
        opening = abs(q[j]) < abs(open_val) * 0.5
        q[j] = open_val if opening else 0.0
        ao.joint_positions = q
        self.status = f"{'opened' if opening else 'closed'} {label}"

    def toggle_item(self):
        st = self.runner.agents[self.active]
        rom = self.env.sim.get_rigid_object_manager()
        if st.carrying:
            obj = rom.get_object_by_handle(st.carrying)
            obj.motion_type = obj.motion_type.__class__.DYNAMIC
            self.status = f"put down {st.carrying}"
            st.carrying = None
            return
        p = self._pos(self.active)
        best, best_d = None, 1.2
        for it in ITEMS:
            obj = rom.get_object_by_handle(it.obj)
            d = np.linalg.norm((np.array(obj.translation) - p)[[0, 2]])
            if d < best_d:
                best, best_d = it.obj, d
        if best is None:
            self.status = "no item within reach"
            return
        obj = rom.get_object_by_handle(best)
        obj.motion_type = obj.motion_type.__class__.KINEMATIC
        st.carrying = best
        self.status = f"picked up {best}"

    # ------------------------------------------------------------ step
    def step(self, keys):
        sim = self.env.sim
        i = self.active
        c = self.ctrl[i]
        fwd, back = keys[pygame.K_w], keys[pygame.K_s]
        turn = keys[pygame.K_a] - keys[pygame.K_d]
        speed = RUN if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else WALK
        if speed != self.speed:
            c.set_framerate_for_linspeed(speed, TURN, FPS)
            self.speed = speed
        self.heading[i] += turn * TURN / FPS
        h = self.heading[i]
        d = mn.Vector3(np.cos(h), 0, -np.sin(h))  # heading 0 = +x
        old = mn.Vector3(c.obj_transform_base.translation)
        if fwd or back:
            c.calculate_walk_pose(d if fwd else -d)
        elif turn:
            c.calculate_turn_pose(d)
        else:
            c.calculate_stop_pose()
        # the controller ignores walls: slide the new root along the main
        # floor navmesh. Snap from floor height, on the floor island only:
        # from hip height the nearest navmesh is a stair step, a separate
        # island, and the resident would walk through the stairs.
        new = c.obj_transform_base.translation
        pf, island = sim.pathfinder, sim.largest_island_idx
        floor_y = old.y - self.root_off[i]
        a = pf.snap_point(mn.Vector3(old.x, floor_y, old.z), island)
        s = pf.try_step(a, mn.Vector3(new.x, a.y, new.z))
        if self._blocked_by_door(a, s):
            s = a
        # follow the navmesh height (stairs); root stays at hip height above it
        c.obj_transform_base.translation = mn.Vector3(s.x, s.y + self.root_off[i], s.z)
        self.runner.moving = {k: False for k in range(self.n)}
        self.runner.moving[i] = bool(fwd or back or turn)
        for k in range(self.n):
            self._chase_camera(k)

        # with a single agent habitat drops the agent_<i>_ prefix from arg names
        arg = f"agent_{i}_human_joints_trans" if self.n > 1 else "human_joints_trans"
        self.obs = self.env.step({"action": (f"agent_{i}_joint",),
                                  "action_args": {arg: c.get_pose()}})

        # carried item rides in front of the chest
        for k, st in self.runner.agents.items():
            if st.carrying:
                agent = sim.agents_mgr[k].articulated_agent
                base = np.array(agent.base_pos)
                hk = self.heading[k]
                obj = sim.get_rigid_object_manager().get_object_by_handle(st.carrying)
                obj.translation = mn.Vector3(base[0] + 0.35 * np.cos(hk), base[1] + 0.2,
                                             base[2] - 0.35 * np.sin(hk))

        self.runner.t += 1.0 / FPS
        self.sensors.update(self.runner)

    def _chase_camera(self, k):
        """Place agent k's third-person camera behind and above its head,
        pulled in when a wall is in the way (the stock offset ends up above
        the ReplicaCAD ceiling)."""
        sim = self.env.sim
        agent = sim.agents_mgr[k].articulated_agent
        agent._auto_update_sensor_transforms = False
        root = self.ctrl[k].obj_transform_base.translation
        floor_y = root.y - self.root_off[k]
        hk = self.heading[k]
        d = mn.Vector3(np.cos(hk), 0, -np.sin(hk))
        head = mn.Vector3(root.x, floor_y + 1.45, root.z)
        want = head - d * 2.0 + mn.Vector3(0, 0.55, 0)
        start = head - d * 0.35  # start behind the body so the ray skips it
        ray = habitat_sim.geo.Ray(start, want - start)
        hits = sim.cast_ray(ray, max_distance=1.0)
        eye = want
        if hits.has_hits():
            frac = max(hits.hits[0].ray_distance - 0.15, 0.0)
            eye = start + (want - start) * frac
        world = mn.Matrix4.look_at(eye, head + d * 1.0 - mn.Vector3(0, 0.5, 0), mn.Vector3(0, 1, 0))
        inv = sim._default_agent.scene_node.transformation.inverted()
        for name in agent._cameras.get("third" if self.n == 1 else f"agent_{k}_third", []):
            sim._sensors[name]._sensor_object.node.transformation = inv @ world

    def frame(self):
        key = "third_rgb" if self.n == 1 else f"agent_{self.active}_third_rgb"
        return self.obs[key][..., :3]

    def save(self):
        self.sensors.finish()
        out = os.path.join(REPO_ROOT, "outputs", "play")
        self.sensors.save(out)
        return out


def draw_panel(screen, font, small, player):
    x0 = CAM
    screen.fill((27, 58, 107), (x0, 0, PANEL, CAM))
    t = player.runner.t
    head = f"Resident R{player.active + 1}  |  t = {int(t // 60):02d}:{t % 60:04.1f}"
    screen.blit(font.render(head, True, (232, 241, 255)), (x0 + 14, 10))
    lit = [sid for sid, st in player.sensors._pir.items() if st.on]
    screen.blit(small.render("Motion ON: " + (" ".join(lit) or "-"), True, COLORS["ON"]), (x0 + 14, 38))
    tgt = player.target_link()
    n = len(player._cand_labels)
    more = f"  [TAB {player.pick % n + 1}/{n}]" if n > 1 else ""
    screen.blit(small.render("E -> " + (tgt[0] if tgt else "nothing openable here") + more, True, (124, 227, 139)), (x0 + 14, 60))
    screen.blit(small.render(player.status, True, (180, 200, 230)), (x0 + 14, CAM - 70))
    screen.blit(small.render("time      sensor  value    who", True, (143, 211, 255)), (x0 + 14, 92))
    for k, (te, sid, val, who, _) in enumerate(player.sensors.events[-19:]):
        line = f"{te:8.1f}  {sid}    {val:<7}  {_who(who)}"
        screen.blit(small.render(line, True, COLORS.get(val, (232, 241, 255))), (x0 + 14, 114 + k * 20))
    for k, hint in enumerate(["W/S walk  A/D turn  SHIFT run  F pick/drop",
                              "E open/close  TAB next part  X switch  R reset"]):
        screen.blit(small.render(hint, True, (150, 170, 200)), (x0 + 14, CAM - 44 + k * 20))


def _seg_dist(p, a, b) -> float:
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _side(p, a, b) -> float:
    return float(np.sign((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0]))


def _who(who):
    if who == "none":
        return "noise"
    return ";".join(f"R{int(a.split('_')[1]) + 1}" for a in who.split(";"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=2, choices=[1, 2])
    args = ap.parse_args()
    player = Player(args.agents)

    pygame.init()
    screen = pygame.display.set_mode((CAM + PANEL, CAM))
    pygame.display.set_caption("Habitat: walk a resident, watch the sensors")
    font = pygame.font.SysFont("Menlo", 17, bold=True)
    small = pygame.font.SysFont("Menlo", 13)
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_TAB:
                    player.pick += 1
                elif ev.key == pygame.K_e:
                    player.toggle_nearest()
                elif ev.key == pygame.K_f:
                    player.toggle_item()
                elif ev.key == pygame.K_x and player.n > 1:
                    player.active = (player.active + 1) % player.n
                    player.status = f"now controlling R{player.active + 1}"
                elif ev.key == pygame.K_r:
                    player.new_episode()
        player.step(pygame.key.get_pressed())
        img = np.ascontiguousarray(player.frame())
        screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        draw_panel(screen, font, small, player)
        pygame.display.flip()
        clock.tick(FPS)
    out = player.save()
    print(f"{len(player.sensors.events)} sensor events saved to {out}/sensor_events.csv")
    pygame.quit()
    player.env.close()


if __name__ == "__main__":
    main()
