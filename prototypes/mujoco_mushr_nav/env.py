from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np

try:
    from .layout_generator import StaticHardLayoutGenerator
    from .dynamic_generator import DynamicHardObstacleGenerator
except ImportError:
    from layout_generator import StaticHardLayoutGenerator
    from dynamic_generator import DynamicHardObstacleGenerator


ROOT = Path(__file__).resolve().parent
HARD_TEMPLATE_XML = ROOT / "assets" / "mushr_nav_hard_template.xml"
DEFAULT_XML = HARD_TEMPLATE_XML
DYNAMIC_HARD_XML = HARD_TEMPLATE_XML


@dataclass
class MuSHRNavConfig:
    """任务级参数，和 MuJoCo 具体 API 分开。

    这样写是为了让任务语义保持清楚：以后迁移到 Isaac Lab 时，观测、
    奖励、终止条件等定义可以尽量复用，只替换底层仿真器访问方式。
    """

    xml_path: Path = DEFAULT_XML
    max_episode_steps: int = 500
    frame_skip: int = 5
    goal_radius: float = 0.35
    max_steering: float = 0.38
    max_throttle: float = 5.0
    success_bonus: float = 10.0
    collision_penalty: float = 10.0
    time_penalty: float = 0.01
    control_penalty: float = 0.01
    progress_scale: float = 1.0
    reward_mode: str = "euclidean"
    near_obstacle_penalty: float = 0.2
    near_obstacle_margin: float = 0.8
    near_miss_margin: float = 0.4
    ttc_threshold: float = 1.0
    ego_collision_radius: float = 0.35
    lidar_num_rays: int = 32
    lidar_range: float = 5.0
    lidar_fov: float = 2.0 * np.pi
    history_len: int = 4
    procedural_layout: bool = True
    layout_seed: Optional[int] = None
    layout_template: Optional[str] = None
    dynamic_seed: Optional[int] = None
    dynamic_mode: str = "none"
    dynamic_random_phase: bool = True
    dynamic_random_speed: bool = True
    dynamic_random_amplitude: bool = True


class MuSHRNavEnv(gym.Env):
    """MuSHR 小车的最小 MuJoCo 点目标导航环境。

    全局位姿放在 info 里，用于调试、
    reward 计算和画图。
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[MuSHRNavConfig] = None):
        super().__init__()
        self.cfg = cfg or MuSHRNavConfig()
        if self.cfg.history_len < 1:
            raise ValueError("history_len must be >= 1")
        self._validate_reward_mode(self.cfg.reward_mode)
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data = mujoco.MjData(self.model)

        self.car_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "buddy")
        self.goal_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.steering_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "buddy_steering_wheel")
        self.steering_qpos_addr = self.model.jnt_qposadr[self.steering_joint_id]
        self.car_free_joint_id = self._find_body_free_joint(self.car_body_id)
        self.car_free_qpos_addr = self.model.jnt_qposadr[self.car_free_joint_id]
        self.car_free_qvel_addr = self.model.jnt_dofadr[self.car_free_joint_id]

        self.car_body_ids = self._collect_body_subtree(self.car_body_id)
        self.wall_slot_ids = self._collect_slot_names("wall_slot_")
        self.box_slot_ids = self._collect_slot_names("box_obs_slot_")
        self.cylinder_slot_ids = self._collect_slot_names("cyl_obs_slot_")
        self.obstacle_geom_ids = self._collect_obstacle_geoms()
        self.dynamic_geom_ids = self._collect_dynamic_slot_names()
        self._slot_variants = self._collect_slot_variants(
            self.wall_slot_ids + self.box_slot_ids + self.cylinder_slot_ids + list(self.dynamic_geom_ids)
        )
        self._static_slot_joints = self._collect_static_slot_joints(
            [
                geom_id
                for slot_name in self.wall_slot_ids + self.box_slot_ids + self.cylinder_slot_ids
                for geom_id, _ in self._slot_variants[slot_name]
            ]
        )
        self._dynamic_mocap_ids = self._collect_dynamic_mocap_ids()
        self._dynamic_params = []
        self._layout_generator = StaticHardLayoutGenerator()
        self._dynamic_generator = DynamicHardObstacleGenerator(
            max_obstacles=min(8, max(1, len(self.dynamic_geom_ids))),
            random_phase=self.cfg.dynamic_random_phase,
            random_speed=self.cfg.dynamic_random_speed,
            random_amplitude=self.cfg.dynamic_random_amplitude,
        )
        self._current_layout = None
        self._current_dynamic_scene = None
        self._current_dynamic_seed = None
        self._active_static_geom_ids = []
        self._active_dynamic_geom_ids = []
        self._static_slot_targets = {}
        self._static_slots_initialized = False
        self._layout_episode_idx = 0
        self._validate_dynamic_mode(self.cfg.dynamic_mode)
        if not self.cfg.procedural_layout and self._normalized_dynamic_mode() != "none":
            raise ValueError(
                "dynamic_mode requires procedural_layout=True because dynamic-hard "
                "obstacles are generated from the current LayoutSpec"
            )

        # 当前动作空间：
        #   action[0] = steering command，归一化到 [-1, 1]，再映射到 [-max_steering, max_steering] rad
        #   action[1] = throttle command，归一化到 [-1, 1]，再映射到 [-max_throttle, max_throttle]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._lidar_angles = np.linspace(
            -0.5 * self.cfg.lidar_fov,
            0.5 * self.cfg.lidar_fov,
            self.cfg.lidar_num_rays,
            endpoint=False,
            dtype=np.float64,
        )

        # 单帧 raw observation = 8 维低维状态 + K 维 lidar proxy：
        #   obs[0] = ego_vx          车体坐标系 x 方向速度
        #   obs[1] = ego_vy          车体坐标系 y 方向速度
        #   obs[2] = yaw_rate        车体 yaw 角速度
        #   obs[3] = steering_angle  当前转向关节角
        #   obs[4] = goal_dx_body    目标在车体坐标系下的 x
        #   obs[5] = goal_dy_body    目标在车体坐标系下的 y
        #   obs[6] = goal_distance   到目标的欧氏距离
        #   obs[7] = goal_angle      目标相对车头方向角
        #   obs[8:] = lidar_ranges   车体坐标系下的归一化射线距离，1.0 表示 range 内无障碍
        # 对 static/dynamic 统一堆叠历史帧，使动态障碍速度可由 lidar 变化推断。
        self.raw_obs_dim = 8 + self.cfg.lidar_num_rays
        obs_dim = self.raw_obs_dim * self.cfg.history_len
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_goal_distance = 0.0
        self._prev_reward_distance = 0.0
        self._obs_history = np.zeros((self.cfg.history_len, self.raw_obs_dim), dtype=np.float32)
        self._episode_initial_goal_distance = 0.0
        self._episode_path_length = 0.0
        self._episode_min_obstacle_distance = float("inf")
        self._episode_min_dynamic_distance = float("inf")
        self._episode_min_ttc = float("inf")
        self._episode_near_miss = False
        self._episode_dynamic_near_miss = False
        self._episode_ttc_violation = False
        self._prev_path_pos = np.zeros(2, dtype=np.float64)
        self._geodesic_cache_key = None
        self._geodesic_xs = None
        self._geodesic_ys = None
        self._geodesic_distances = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._apply_reset_layout(seed, options)
        self._reset_dynamic_obstacles(options)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_goal_distance = self._goal_distance()
        self._prev_reward_distance = self._reward_distance()
        car_pos, _ = self._car_pose()
        self._episode_initial_goal_distance = self._layout_geodesic_distance()
        self._episode_path_length = 0.0
        self._episode_min_obstacle_distance = float("inf")
        self._episode_min_dynamic_distance = float("inf")
        self._episode_min_ttc = float("inf")
        self._episode_near_miss = False
        self._episode_dynamic_near_miss = False
        self._episode_ttc_violation = False
        self._prev_path_pos = car_pos[:2].copy()
        self._update_episode_metrics()
        raw_obs = self._get_raw_obs()
        self._reset_obs_history(raw_obs)
        obs = self._get_obs()
        info = self._get_info(progress=0.0, success=False, collision=False, collision_type="none")
        info["timeout"] = False
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self._map_action(action)

        self.data.ctrl[:] = ctrl
        collision = False
        collision_type = "none"
        for _ in range(self.cfg.frame_skip):
            self._pin_static_slots()
            self._update_dynamic_obstacles(self.data.time)
            mujoco.mj_step(self.model, self.data)
            self._pin_static_slots()
            self._update_dynamic_obstacles(self.data.time)
            collision_type = self._collision_type()
            collision = collision_type in ("static", "dynamic")
            if collision or self._has_bad_state():
                break

        self._step_count += 1
        car_pos, _ = self._car_pose()
        self._episode_path_length += float(np.linalg.norm(car_pos[:2] - self._prev_path_pos))
        self._prev_path_pos = car_pos[:2].copy()
        self._update_episode_metrics()

        distance = self._goal_distance()
        reward_distance = self._reward_distance()
        progress = self._prev_reward_distance - reward_distance
        self._prev_goal_distance = distance
        self._prev_reward_distance = reward_distance

        bad_state = self._has_bad_state()
        if bad_state and not collision:
            collision_type = "bad_state"
        collision = collision or bad_state
        success = distance <= self.cfg.goal_radius
        terminated = bool(success or collision)
        truncated = self._step_count >= self.cfg.max_episode_steps

        min_obstacle_distance = self._min_obstacle_distance()
        reward = self._compute_reward(progress, action, success, collision, min_obstacle_distance)
        self._append_obs_history(self._get_raw_obs())
        obs = self._get_obs()
        info = self._get_info(progress=progress, success=success, collision=collision, collision_type=collision_type)
        info["bad_state"] = bool(bad_state)
        info["timeout"] = bool(truncated and not success and not collision)
        return obs, reward, terminated, truncated, info

    @property
    def max_episode_steps(self):
        return self.cfg.max_episode_steps

    def _validate_reward_mode(self, mode: str):
        if mode not in ("euclidean", "geodesic"):
            raise ValueError(f"Unknown reward mode: {mode}")

    def _required_id(self, obj_type, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return obj_id

    def _find_body_free_joint(self, body_id: int) -> int:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_bodyid[joint_id] == body_id and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return joint_id
        raise ValueError("MuSHR body must have a free joint")

    def _collect_slot_names(self, prefix: str) -> list[str]:
        slots = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(prefix):
                base_name = self._slot_name_from_geom_name(name)
                suffix = base_name.removeprefix(prefix)
                order = int(suffix) if suffix.isdigit() else len(slots)
                slots.append((order, base_name))
        return [base_name for _, base_name in sorted(set(slots))]

    def _collect_slot_variants(self, slot_names: list[str]) -> dict[str, list[tuple[int, np.ndarray]]]:
        slot_set = set(slot_names)
        variants = {slot_name: [] for slot_name in slot_names}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            base_name = self._slot_name_from_geom_name(name)
            if base_name not in slot_set:
                continue
            variants[base_name].append((geom_id, self.model.geom_size[geom_id].copy()))
        for slot_name, items in variants.items():
            if not items:
                raise ValueError(f"slot {slot_name!r} has no geom variants")
            items.sort(key=lambda item: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, item[0]) or "")
        return variants

    def _has_slot_layout(self) -> bool:
        return bool(self.wall_slot_ids or self.box_slot_ids or self.cylinder_slot_ids)

    def _apply_reset_layout(self, seed: Optional[int], options: Optional[dict]):
        if not self._has_slot_layout():
            return
        if self.cfg.procedural_layout:
            layout_seed = self._resolve_layout_seed(seed, options)
            template_name = self._resolve_layout_template(options)
            layout = self._layout_generator.generate(seed=layout_seed, template_name=template_name)
            self._apply_layout_spec(layout)
        else:
            self._apply_default_slot_layout()

    def _resolve_layout_seed(self, seed: Optional[int], options: Optional[dict]) -> int:
        if options and "layout_seed" in options:
            return int(options["layout_seed"])
        if self.cfg.layout_seed is not None:
            return int(self.cfg.layout_seed)
        if seed is not None:
            return int(seed)
        layout_seed = self._layout_episode_idx
        self._layout_episode_idx += 1
        return layout_seed

    def _resolve_layout_template(self, options: Optional[dict]) -> Optional[str]:
        if options and "layout_template" in options:
            return options["layout_template"]
        return self.cfg.layout_template

    def _validate_dynamic_mode(self, mode: str):
        if mode not in ("none", "hard", "moving", "frozen"):
            raise ValueError(f"Unknown dynamic mode: {mode}")

    def _normalized_dynamic_mode(self) -> str:
        if self.cfg.dynamic_mode == "hard":
            return "moving"
        return self.cfg.dynamic_mode

    def _resolve_dynamic_seed(self, options: Optional[dict]) -> int:
        if options and "dynamic_seed" in options:
            return int(options["dynamic_seed"])
        if self.cfg.dynamic_seed is not None:
            return int(self.cfg.dynamic_seed)
        if self._current_layout is not None:
            return int(self._current_layout.seed)
        return 0

    def _static_slot_names(self) -> list[str]:
        return self.wall_slot_ids + self.box_slot_ids + self.cylinder_slot_ids

    def _clear_geom_slots(self, slot_names_to_prepare: Optional[list[str]] = None):
        previous_active_geom_ids = list(self._active_static_geom_ids)
        self._active_static_geom_ids = []
        static_slot_names = self._static_slot_names()

        if not self._static_slots_initialized:
            for idx, slot_name in enumerate(static_slot_names):
                self._disable_slot(slot_name, idx, update_qpos0=True)
            self._static_slots_initialized = True
            return

        for idx, geom_id in enumerate(previous_active_geom_ids):
            self._disable_geom_slot(geom_id, idx)

        if not slot_names_to_prepare:
            return
        slot_order = {slot_name: idx for idx, slot_name in enumerate(static_slot_names)}
        for slot_name in sorted(set(slot_names_to_prepare), key=lambda name: slot_order[name]):
            self._disable_slot(slot_name, slot_order[slot_name])

    def _apply_layout_spec(self, layout):
        slot_names_to_prepare = (
            self.wall_slot_ids[: len(layout.walls)]
            + self.box_slot_ids[: len(layout.boxes)]
            + self.cylinder_slot_ids[: len(layout.cylinders)]
        )
        self._clear_geom_slots(slot_names_to_prepare)
        self._current_layout = layout
        self._configure_spec_slots(self.wall_slot_ids, layout.walls)
        self._configure_spec_slots(self.box_slot_ids, layout.boxes)
        self._configure_spec_slots(self.cylinder_slot_ids, layout.cylinders)
        self._set_goal_xy(layout.goal_xy)
        self._set_car_start(layout.start_xy, layout.start_yaw)

    def _configure_spec_slots(self, slot_ids: list[str], specs):
        if len(specs) > len(slot_ids):
            raise ValueError(f"Not enough MuJoCo geom slots: need {len(specs)}, have {len(slot_ids)}")
        for idx, spec in enumerate(specs):
            slot_name = slot_ids[idx]
            geom_id = self._select_slot_variant(slot_name, spec.size)
            self._configure_geom_slot(geom_id, spec.pos, spec.size)
            self._active_static_geom_ids.append(geom_id)

    def _apply_default_slot_layout(self):
        if not self._has_slot_layout():
            return

        wall_specs = [
            ((0.0, 5.05, 0.25), (5.05, 0.05, 0.25)),
            ((0.0, -5.05, 0.25), (5.05, 0.05, 0.25)),
            ((5.05, 0.0, 0.25), (0.05, 5.05, 0.25)),
            ((-5.05, 0.0, 0.25), (0.05, 5.05, 0.25)),
            ((-1.8, 0.7, 0.25), (0.08, 1.35, 0.25)),
            ((1.7, -0.8, 0.25), (0.08, 1.20, 0.25)),
            ((0.7, 2.2, 0.25), (1.25, 0.08, 0.25)),
        ]
        box_specs = [
            ((-3.0, -1.3, 0.18), (0.35, 0.35, 0.18)),
            ((2.9, 1.3, 0.18), (0.45, 0.30, 0.18)),
        ]
        cylinder_specs = [
            ((-0.4, -2.4, 0.22), (0.28, 0.22, 0.01)),
            ((3.4, -2.2, 0.22), (0.24, 0.22, 0.01)),
        ]

        slot_names_to_prepare = (
            self.wall_slot_ids[: len(wall_specs)]
            + self.box_slot_ids[: len(box_specs)]
            + self.cylinder_slot_ids[: len(cylinder_specs)]
        )
        self._clear_geom_slots(slot_names_to_prepare)
        self._current_layout = None

        for idx, (pos, size) in enumerate(wall_specs):
            geom_id = self._select_slot_variant(self.wall_slot_ids[idx], np.asarray(size, dtype=np.float64))
            self._configure_geom_slot(geom_id, pos, size)
            self._active_static_geom_ids.append(geom_id)
        for idx, (pos, size) in enumerate(box_specs):
            geom_id = self._select_slot_variant(self.box_slot_ids[idx], np.asarray(size, dtype=np.float64))
            self._configure_geom_slot(geom_id, pos, size)
            self._active_static_geom_ids.append(geom_id)
        for idx, (pos, size) in enumerate(cylinder_specs):
            geom_id = self._select_slot_variant(self.cylinder_slot_ids[idx], np.asarray(size, dtype=np.float64))
            self._configure_geom_slot(geom_id, pos, size)
            self._active_static_geom_ids.append(geom_id)

        self._set_goal_xy(np.array([3.8, -3.6], dtype=np.float64))
        self._set_car_start(np.array([-4.0, 3.6], dtype=np.float64), yaw=0.0)

    def _configure_geom_slot(self, geom_id: int, pos, size):
        pos = np.asarray(pos, dtype=np.float64)
        self._set_static_slot_pose(geom_id, pos)

    def _disable_geom_slot(self, geom_id: int, idx: int):
        pos = np.array([1000.0 + 50.0 * idx, -1000.0, 50.0], dtype=np.float64)
        self._set_static_slot_pose(geom_id, pos)

    def _disable_slot(self, slot_name: str, idx: int, update_qpos0: bool = False):
        for variant_idx, (geom_id, _size) in enumerate(self._slot_variants[slot_name]):
            pos = np.array([1000.0 + 10000.0 * idx + 50.0 * variant_idx, 1000.0, 50.0], dtype=np.float64)
            self._set_static_slot_pose(geom_id, pos, update_qpos0=update_qpos0)

    def _select_slot_variant(self, slot_name: str, size: np.ndarray) -> int:
        target = np.asarray(size, dtype=np.float64)
        for geom_id, variant_size in self._slot_variants[slot_name]:
            if self._sizes_match(geom_id, variant_size, target):
                return geom_id
        available = [variant_size.tolist() for _, variant_size in self._slot_variants[slot_name]]
        raise ValueError(f"slot {slot_name!r} has no compiled variant for size {target.tolist()}; available={available[:8]}...")

    def _sizes_match(self, geom_id: int, candidate: np.ndarray, target: np.ndarray) -> bool:
        geom_type = self.model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return bool(np.allclose(candidate[:2], target[:2], atol=1e-8, rtol=0.0))
        return bool(np.allclose(candidate, target, atol=1e-8, rtol=0.0))

    def _set_static_slot_pose(self, geom_id: int, pos: np.ndarray, update_qpos0: bool = False):
        if geom_id not in self._static_slot_joints:
            raise KeyError(f"static obstacle geom {geom_id} has no free-joint slot")
        qpos_addr, qvel_addr = self._static_slot_joints[geom_id]
        self.data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        if update_qpos0:
            self.model.qpos0[qpos_addr : qpos_addr + 3] = pos
            self.model.qpos0[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._static_slot_targets[geom_id] = pos.copy()

    def _pin_static_slots(self):
        for geom_id in self._active_static_geom_ids:
            pos = self._static_slot_targets[geom_id]
            qpos_addr, qvel_addr = self._static_slot_joints[geom_id]
            self.data.qpos[qpos_addr : qpos_addr + 3] = pos
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def _set_goal_xy(self, xy: np.ndarray):
        self.model.site_pos[self.goal_site_id, 0] = xy[0]
        self.model.site_pos[self.goal_site_id, 1] = xy[1]
        self.model.site_pos[self.goal_site_id, 2] = 0.08

    def _set_car_start(self, xy: np.ndarray, yaw: float):
        qpos_addr = self.car_free_qpos_addr
        qvel_addr = self.car_free_qvel_addr
        self.data.qpos[qpos_addr : qpos_addr + 7] = np.array(
            [xy[0], xy[1], 0.0, np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
            dtype=np.float64,
        )
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def _collect_body_subtree(self, root_body_id: int) -> set[int]:
        body_ids = set()
        for body_id in range(self.model.nbody):
            cur = body_id
            while cur != 0:
                if cur == root_body_id:
                    body_ids.add(body_id)
                    break
                cur = self.model.body_parentid[cur]
        return body_ids

    def _collect_obstacle_geoms(self) -> list[int]:
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("wall") or name.startswith("inner_wall") or "obs" in name:
                geom_ids.append(geom_id)
        return geom_ids

    def _collect_dynamic_geoms(self) -> dict[str, int]:
        geom_ids = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("dyn_obs_"):
                geom_ids[name] = geom_id
        return geom_ids

    def _collect_dynamic_slot_names(self) -> tuple[str, ...]:
        names = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("dyn_obs_"):
                base_name = self._slot_name_from_geom_name(name)
                suffix = base_name.removeprefix("dyn_obs_")
                order = int(suffix) if suffix.isdigit() else len(names)
                names.append((order, base_name))
        return tuple(base_name for _, base_name in sorted(set(names)))

    def _slot_name_for_geom(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        return self._slot_name_from_geom_name(name)

    @staticmethod
    def _slot_name_from_geom_name(name: str) -> str:
        return name.rsplit("_v", 1)[0]

    def _collect_static_slot_joints(self, geom_ids: list[int]) -> dict[int, tuple[int, int]]:
        slots = {}
        for geom_id in geom_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            if joint_id < 0:
                raise ValueError(f"static obstacle geom {name!r} must be attached to a free joint named {name}_free")
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"static obstacle joint {name}_free must be a free joint")
            slots[geom_id] = (
                int(self.model.jnt_qposadr[joint_id]),
                int(self.model.jnt_dofadr[joint_id]),
            )
        return slots

    def _collect_dynamic_mocap_ids(self) -> dict[str, int]:
        mocap_ids = {}
        for name in self.dynamic_geom_ids:
            for geom_id, _ in self._slot_variants[name]:
                body_id = int(self.model.geom_bodyid[geom_id])
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id < 0:
                    geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id)
                    raise ValueError(f"dynamic obstacle geom {geom_name!r} must be attached to a mocap body")
                mocap_ids[geom_id] = mocap_id
        return mocap_ids

    def _reset_dynamic_obstacles(self, options: Optional[dict] = None):
        self._dynamic_params = []
        self._current_dynamic_scene = None
        self._current_dynamic_seed = None
        self._active_dynamic_geom_ids = []
        self._disable_unused_dynamic_slots(set())

        mode = self._normalized_dynamic_mode()
        if mode == "none":
            return
        if self._current_layout is None:
            return
        if not self.dynamic_geom_ids:
            raise ValueError("Dynamic obstacle mode requires dyn_obs_N geoms in XML")

        dynamic_seed = self._resolve_dynamic_seed(options)
        scene = self._dynamic_generator.generate(self._current_layout, seed=dynamic_seed, mode=mode)
        missing = [spec.name for spec in scene.obstacles if spec.name not in self.dynamic_geom_ids]
        if missing:
            raise ValueError(f"Dynamic obstacle geoms missing from XML: {missing}")

        active_geom_ids = set()
        for spec in scene.obstacles:
            geom_id = self._select_slot_variant(spec.name, spec.size)
            self._dynamic_params.append(
                {
                    "geom_id": geom_id,
                    "mocap_id": self._dynamic_mocap_ids[geom_id],
                    "spec": spec,
                }
            )
            active_geom_ids.add(geom_id)
            self._active_dynamic_geom_ids.append(geom_id)
        self._current_dynamic_scene = scene
        self._current_dynamic_seed = dynamic_seed
        self._disable_unused_dynamic_slots(active_geom_ids)
        self._update_dynamic_obstacles(0.0)

    def _disable_unused_dynamic_slots(self, active_geom_ids: set[int]):
        for idx, name in enumerate(self.dynamic_geom_ids, start=100):
            for variant_idx, (geom_id, _size) in enumerate(self._slot_variants[name]):
                if geom_id in active_geom_ids:
                    continue
                mocap_id = self._dynamic_mocap_ids[geom_id]
                self.data.mocap_pos[mocap_id] = np.array([1000.0 + 10000.0 * idx + 50.0 * variant_idx, 1000.0, 50.0], dtype=np.float64)
                self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _update_dynamic_obstacles(self, time: float):
        if not self._dynamic_params:
            return
        for params in self._dynamic_params:
            spec = params["spec"]
            xy = spec.position_at(time)
            mocap_id = params["mocap_id"]
            self.data.mocap_pos[mocap_id] = np.array([xy[0], xy[1], spec.z], dtype=np.float64)
            self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        steering = float(action[0]) * self.cfg.max_steering
        throttle = float(action[1]) * self.cfg.max_throttle
        return np.array([steering, throttle], dtype=np.float64)

    def _get_raw_obs(self) -> np.ndarray:
        car_pos, yaw = self._car_pose()
        rot = self._yaw_rotation(yaw)

        # MuJoCo free joint 的 qvel 顺序是 [linear_xyz, angular_xyz]。
        vel_world_xy = self.data.qvel[:2].copy()
        vel_body_xy = rot.T @ vel_world_xy
        yaw_rate = float(self.data.qvel[5])
        steering_angle = float(self.data.qpos[self.steering_qpos_addr])

        goal_xy = self._goal_xy()
        rel_goal_world = goal_xy - car_pos[:2]
        rel_goal_body = rot.T @ rel_goal_world
        goal_distance = float(np.linalg.norm(rel_goal_body))
        goal_angle = float(np.arctan2(rel_goal_body[1], rel_goal_body[0]))

        low_dim = np.array(
            [
                vel_body_xy[0],
                vel_body_xy[1],
                yaw_rate,
                steering_angle,
                rel_goal_body[0],
                rel_goal_body[1],
                goal_distance,
                goal_angle,
            ],
            dtype=np.float32,
        )
        lidar = self._get_lidar_ranges(car_pos[:2], yaw).astype(np.float32)
        return np.concatenate([low_dim, lidar])

    def _reset_obs_history(self, raw_obs: np.ndarray):
        self._obs_history[:] = raw_obs

    def _append_obs_history(self, raw_obs: np.ndarray):
        self._obs_history[:-1] = self._obs_history[1:].copy()
        self._obs_history[-1] = raw_obs

    def _get_obs(self) -> np.ndarray:
        return self._obs_history.reshape(-1).astype(np.float32, copy=False)

    def _compute_reward(
        self,
        progress: float,
        action: np.ndarray,
        success: bool,
        collision: bool,
        min_obstacle_distance: float,
    ) -> float:
        reward = self.cfg.progress_scale * progress
        reward -= self.cfg.time_penalty
        reward -= self.cfg.control_penalty * float(np.square(action).sum())
        if min_obstacle_distance < self.cfg.near_obstacle_margin:
            proximity = 1.0 - max(min_obstacle_distance, 0.0) / self.cfg.near_obstacle_margin
            reward -= self.cfg.near_obstacle_penalty * proximity
        if success:
            reward += self.cfg.success_bonus
        if collision:
            reward -= self.cfg.collision_penalty
        return float(reward)

    def _get_info(self, progress: float, success: bool, collision: bool, collision_type: str) -> dict:
        car_pos, yaw = self._car_pose()
        shortest_path = max(self._episode_initial_goal_distance, 1e-6)
        spl = float(success) * shortest_path / max(self._episode_path_length, shortest_path)
        min_dynamic_distance = self._episode_min_dynamic_distance
        min_ttc = self._episode_min_ttc
        info = {
            "time": float(self.data.time),
            "layout_seed": -1 if self._current_layout is None else int(self._current_layout.seed),
            "layout_template": "fixed" if self._current_layout is None else self._current_layout.template_name,
            "dynamic_mode": self._normalized_dynamic_mode()
            if self._current_dynamic_scene is not None
            else self.cfg.dynamic_mode,
            "dynamic_seed": -1 if self._current_dynamic_seed is None else int(self._current_dynamic_seed),
            "x": float(car_pos[0]),
            "y": float(car_pos[1]),
            "yaw": float(yaw),
            "goal_x": float(self._goal_xy()[0]),
            "goal_y": float(self._goal_xy()[1]),
            "distance_to_goal": self._goal_distance(),
            "euclidean_distance_to_goal": self._goal_distance(),
            "geodesic_distance_to_goal": self._info_geodesic_distance_to_goal(),
            "reward_distance_to_goal": self._reward_distance(),
            "reward_mode": self.cfg.reward_mode,
            "progress": float(progress),
            "success": bool(success),
            "collision": bool(collision),
            "collision_type": collision_type,
            "static_collision": bool(collision and collision_type == "static"),
            "dynamic_collision": bool(collision and collision_type == "dynamic"),
            "path_length": float(self._episode_path_length),
            "episode_geodesic_distance": float(self._episode_initial_goal_distance),
            "spl": spl,
            "min_obstacle_distance": self._min_obstacle_distance(),
            "episode_min_obstacle_distance": float(self._episode_min_obstacle_distance),
            "near_miss": bool(self._episode_near_miss),
            "dynamic_min_distance": float(min_dynamic_distance),
            "dynamic_near_miss": bool(self._episode_dynamic_near_miss),
            "min_ttc": float(min_ttc),
            "ttc_violation": bool(self._episode_ttc_violation),
            "step": self._step_count,
        }
        for state in self._dynamic_obstacle_states():
            prefix = state["name"]
            info[f"{prefix}_x"] = state["x"]
            info[f"{prefix}_y"] = state["y"]
            info[f"{prefix}_vx"] = state["vx"]
            info[f"{prefix}_vy"] = state["vy"]
            info[f"{prefix}_pattern"] = state["pattern"]
            info[f"{prefix}_trajectory_type"] = state["trajectory_type"]
        return {
            **info,
        }

    def _has_bad_state(self) -> bool:
        return not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def _car_pose(self) -> tuple[np.ndarray, float]:
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.car_body_id].copy()
        mat = self.data.xmat[self.car_body_id].reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return pos, yaw

    def _goal_xy(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.goal_site_id, :2].copy()

    def _goal_distance(self) -> float:
        car_pos, _ = self._car_pose()
        return float(np.linalg.norm(self._goal_xy() - car_pos[:2]))

    def _layout_geodesic_distance(self) -> float:
        if self._current_layout is None:
            return self._goal_distance()
        return float(self._current_layout.geodesic_distance)

    def _reward_distance(self) -> float:
        if self.cfg.reward_mode == "euclidean":
            return self._goal_distance()
        return self._geodesic_distance_to_goal()

    def _info_geodesic_distance_to_goal(self) -> float:
        if self.cfg.reward_mode != "geodesic":
            return float("nan")
        return self._geodesic_distance_to_goal()

    def _geodesic_distance_to_goal(self) -> float:
        if self._current_layout is None:
            return self._goal_distance()
        self._ensure_geodesic_distance_field()
        car_pos, _ = self._car_pose()
        idx = self._layout_generator._xy_to_grid(car_pos[:2], self._geodesic_xs, self._geodesic_ys)
        if idx is None:
            return self._goal_distance()
        distance = float(self._geodesic_distances[idx])
        if np.isfinite(distance):
            return distance
        nearest = self._nearest_finite_geodesic_distance(idx)
        return nearest if np.isfinite(nearest) else self._goal_distance()

    def _ensure_geodesic_distance_field(self):
        cache_key = self._geodesic_cache_key_for_current_layout()
        if cache_key == self._geodesic_cache_key and self._geodesic_distances is not None:
            return
        if self._current_layout is None:
            self._geodesic_cache_key = None
            self._geodesic_xs = None
            self._geodesic_ys = None
            self._geodesic_distances = None
            return

        xs, ys = self._layout_generator._grid_axes()
        occupied = self._layout_occupied_grid(xs, ys)
        goal_idx = self._layout_generator._xy_to_grid(self._current_layout.goal_xy, xs, ys)
        if goal_idx is None or occupied[goal_idx]:
            raise RuntimeError("current layout goal is outside the geodesic grid or occupied")

        self._geodesic_cache_key = cache_key
        self._geodesic_xs = xs
        self._geodesic_ys = ys
        self._geodesic_distances = self._dijkstra_distance_field(occupied, goal_idx)

    def _geodesic_cache_key_for_current_layout(self):
        if self._current_layout is None:
            return None
        goal = tuple(np.round(self._current_layout.goal_xy.astype(np.float64), 6))
        return (
            self._current_layout.template_name,
            int(self._current_layout.seed),
            goal,
        )

    def _layout_occupied_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        occupied = np.zeros((len(xs), len(ys)), dtype=bool)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")

        for box in [*self._current_layout.walls, *self._current_layout.boxes]:
            half = box.size[:2] + self._layout_generator.inflation_radius
            center = box.pos[:2]
            occupied |= (np.abs(xx - center[0]) <= half[0]) & (np.abs(yy - center[1]) <= half[1])

        for cylinder in self._current_layout.cylinders:
            radius = float(cylinder.size[0] + self._layout_generator.inflation_radius)
            center = cylinder.pos[:2]
            occupied |= (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius
        return occupied

    def _dijkstra_distance_field(self, occupied: np.ndarray, goal_idx: tuple[int, int]) -> np.ndarray:
        width, height = occupied.shape
        distances = np.full_like(occupied, np.inf, dtype=np.float64)
        distances[goal_idx] = 0.0
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, goal_idx)]
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)),
            (-1, 1, np.sqrt(2.0)),
            (1, -1, np.sqrt(2.0)),
            (1, 1, np.sqrt(2.0)),
        )

        while queue:
            distance, (ix, iy) = heappop(queue)
            if distance > distances[ix, iy]:
                continue
            for dx, dy, multiplier in neighbors:
                nx = ix + dx
                ny = iy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if occupied[nx, ny]:
                    continue
                step = self._layout_generator.grid_resolution * float(multiplier)
                new_distance = distance + step
                if new_distance < distances[nx, ny]:
                    distances[nx, ny] = new_distance
                    heappush(queue, (new_distance, (nx, ny)))
        return distances

    def _nearest_finite_geodesic_distance(self, idx: tuple[int, int]) -> float:
        ix, iy = idx
        width, height = self._geodesic_distances.shape
        best = float("inf")
        max_radius = int(np.ceil(self.cfg.near_obstacle_margin / self._layout_generator.grid_resolution))
        for radius in range(1, max_radius + 1):
            xmin = max(0, ix - radius)
            xmax = min(width, ix + radius + 1)
            ymin = max(0, iy - radius)
            ymax = min(height, iy + radius + 1)
            window = self._geodesic_distances[xmin:xmax, ymin:ymax]
            finite = np.isfinite(window)
            if not finite.any():
                continue
            local_idxs = np.argwhere(finite)
            for local_ix, local_iy in local_idxs:
                gx = xmin + int(local_ix)
                gy = ymin + int(local_iy)
                offset = self._layout_generator.grid_resolution * float(np.hypot(gx - ix, gy - iy))
                best = min(best, float(self._geodesic_distances[gx, gy]) + offset)
            return best
        return best

    def _has_collision(self) -> bool:
        return self._collision_type() in ("static", "dynamic")

    def _collision_type(self) -> str:
        dynamic_geom_set = set(self._active_dynamic_geom_ids)
        active_static_set = set(self._active_static_geom_ids) if self._has_slot_layout() else set(self.obstacle_geom_ids)
        all_static_slot_geoms = set()
        all_dynamic_slot_geoms = set()
        for slot_name in self.wall_slot_ids + self.box_slot_ids + self.cylinder_slot_ids:
            all_static_slot_geoms.update(geom_id for geom_id, _ in self._slot_variants[slot_name])
        for slot_name in self.dynamic_geom_ids:
            all_dynamic_slot_geoms.update(geom_id for geom_id, _ in self._slot_variants[slot_name])
        has_static_collision = False
        has_dynamic_collision = False
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            car1 = body1 in self.car_body_ids
            car2 = body2 in self.car_body_ids
            if car1 == car2:
                continue
            other_geom = contact.geom2 if car1 else contact.geom1
            if other_geom in dynamic_geom_set:
                has_dynamic_collision = True
                continue
            if other_geom in active_static_set:
                has_static_collision = True
                continue
            if other_geom in all_dynamic_slot_geoms:
                continue
            if other_geom in all_static_slot_geoms:
                continue
        if has_dynamic_collision:
            return "dynamic"
        if has_static_collision:
            return "static"
        return "none"

    def _min_obstacle_distance(self) -> float:
        car_pos, _ = self._car_pose()
        point = car_pos[:2]
        geom_ids = self._active_static_geom_ids if self._has_slot_layout() else self.obstacle_geom_ids
        dists = [self._distance_to_geom_xy(point, geom_id) for geom_id in geom_ids]
        return float(min(dists)) if dists else float("inf")

    def _min_dynamic_distance(self) -> float:
        if not self._dynamic_params:
            return float("inf")
        car_pos, _ = self._car_pose()
        point = car_pos[:2]
        dists = [self._distance_to_geom_xy(point, params["geom_id"]) for params in self._dynamic_params]
        return float(min(dists)) if dists else float("inf")

    def _update_episode_metrics(self):
        min_obstacle_distance = self._min_obstacle_distance()
        min_dynamic_distance = self._min_dynamic_distance()
        min_ttc = self._min_dynamic_ttc()
        self._episode_min_obstacle_distance = min(self._episode_min_obstacle_distance, min_obstacle_distance)
        self._episode_min_dynamic_distance = min(self._episode_min_dynamic_distance, min_dynamic_distance)
        self._episode_min_ttc = min(self._episode_min_ttc, min_ttc)
        self._episode_near_miss = self._episode_near_miss or min_obstacle_distance < self.cfg.near_miss_margin
        self._episode_dynamic_near_miss = (
            self._episode_dynamic_near_miss or min_dynamic_distance < self.cfg.near_miss_margin
        )
        self._episode_ttc_violation = self._episode_ttc_violation or min_ttc < self.cfg.ttc_threshold

    def _dynamic_obstacle_states(self) -> list[dict]:
        states = []
        for idx, params in enumerate(self._dynamic_params):
            geom_id = params["geom_id"]
            spec = params["spec"]
            name = spec.name or self._slot_name_for_geom(geom_id) or f"dyn_obs_{idx}"
            vel = spec.velocity_at(self.data.time)
            pos = self.data.geom_xpos[geom_id, :2]
            states.append(
                {
                    "name": name,
                    "geom_id": geom_id,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "vx": float(vel[0]),
                    "vy": float(vel[1]),
                    "pattern": spec.pattern,
                    "trajectory_type": spec.trajectory_type,
                }
            )
        return states

    def _min_dynamic_ttc(self) -> float:
        if not self._dynamic_params:
            return float("inf")
        car_pos, _ = self._car_pose()
        ego_pos = car_pos[:2]
        ego_vel = self.data.qvel[:2].copy()
        ttcs = []
        for state in self._dynamic_obstacle_states():
            geom_id = state["geom_id"]
            rel_pos = np.array([state["x"], state["y"]], dtype=np.float64) - ego_pos
            rel_vel = np.array([state["vx"], state["vy"]], dtype=np.float64) - ego_vel
            radius = self.cfg.ego_collision_radius + self._geom_planar_radius(geom_id)
            ttc = self._disc_ttc(rel_pos, rel_vel, radius)
            ttcs.append(ttc)
        return float(min(ttcs)) if ttcs else float("inf")

    def _geom_planar_radius(self, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.linalg.norm(size[:2]))
        return float(max(size[0], size[1]))

    def _disc_ttc(self, rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float) -> float:
        c = float(np.dot(rel_pos, rel_pos) - radius * radius)
        if c <= 0.0:
            return 0.0
        a = float(np.dot(rel_vel, rel_vel))
        if a < 1e-8:
            return float("inf")
        b = 2.0 * float(np.dot(rel_pos, rel_vel))
        if b >= 0.0:
            return float("inf")
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return float("inf")
        t = (-b - np.sqrt(disc)) / (2.0 * a)
        return float(t) if t >= 0.0 else float("inf")

    def _distance_to_geom_xy(self, point: np.ndarray, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        pos = self.data.geom_xpos[geom_id, :2]
        size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(np.linalg.norm(point - pos) - size[0])

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[:2]
            delta = np.abs(point - pos) - half
            outside = np.maximum(delta, 0.0)
            outside_dist = np.linalg.norm(outside)
            inside_dist = min(max(delta[0], delta[1]), 0.0)
            return float(outside_dist + inside_dist)

        return float(np.linalg.norm(point - pos))

    def _get_lidar_ranges(self, origin: np.ndarray, yaw: float) -> np.ndarray:
        """计算简化 2D lidar。

        当前只对 XML 中的 wall/obstacle geoms 做几何求交，不依赖 MuJoCo
        OpenGL 渲染。返回值归一化到 [0, 1]，越小表示越近。
        """

        ranges = np.full(self.cfg.lidar_num_rays, self.cfg.lidar_range, dtype=np.float64)
        geom_ids = list(self._active_static_geom_ids) if self._has_slot_layout() else list(self.obstacle_geom_ids)
        if self._dynamic_params:
            geom_ids.extend(params["geom_id"] for params in self._dynamic_params)
        for ray_idx, rel_angle in enumerate(self._lidar_angles):
            angle = yaw + rel_angle
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            for geom_id in geom_ids:
                hit = self._ray_geom_intersection(origin, direction, geom_id)
                if hit is not None and 0.0 <= hit < ranges[ray_idx]:
                    ranges[ray_idx] = hit
        return np.clip(ranges / self.cfg.lidar_range, 0.0, 1.0)

    def _ray_geom_intersection(self, origin: np.ndarray, direction: np.ndarray, geom_id: int) -> Optional[float]:
        geom_type = self.model.geom_type[geom_id]
        pos = self.data.geom_xpos[geom_id, :2]
        size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return self._ray_aabb_intersection(origin, direction, pos, size[:2])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return self._ray_circle_intersection(origin, direction, pos, size[0])
        return None

    def _ray_aabb_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
    ) -> Optional[float]:
        bounds_min = center - half_extents
        bounds_max = center + half_extents
        t_min = -np.inf
        t_max = np.inf

        for axis in range(2):
            if abs(direction[axis]) < 1e-8:
                if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                    return None
                continue
            inv_dir = 1.0 / direction[axis]
            t1 = (bounds_min[axis] - origin[axis]) * inv_dir
            t2 = (bounds_max[axis] - origin[axis]) * inv_dir
            t_near, t_far = min(t1, t2), max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return None

        if t_max < 0.0:
            return None
        return float(max(t_min, 0.0))

    def _ray_circle_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> Optional[float]:
        offset = origin - center
        b = 2.0 * float(np.dot(direction, offset))
        c = float(np.dot(offset, offset) - radius * radius)
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = np.sqrt(disc)
        roots = [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]
        positive = [t for t in roots if t >= 0.0]
        if not positive:
            return None
        return float(min(positive))

    @staticmethod
    def _yaw_rotation(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=np.float64)
