from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "assets" / "mushr_nav_static.xml"
DYNAMIC_EASY_XML = ROOT / "assets" / "mushr_nav_dynamic_easy.xml"


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
    near_obstacle_penalty: float = 0.2
    near_obstacle_margin: float = 0.8
    lidar_num_rays: int = 32
    lidar_range: float = 5.0
    lidar_fov: float = 2.0 * np.pi
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
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data = mujoco.MjData(self.model)

        self.car_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "buddy")
        self.goal_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.steering_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "buddy_steering_wheel")
        self.steering_qpos_addr = self.model.jnt_qposadr[self.steering_joint_id]

        self.car_body_ids = self._collect_body_subtree(self.car_body_id)
        self.obstacle_geom_ids = self._collect_obstacle_geoms()
        self.dynamic_geom_ids = self._collect_dynamic_geoms()
        self._dynamic_specs = self._build_dynamic_specs()
        self._dynamic_params = []

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

        # 当前 observation space = 8 维低维状态 + K 维 lidar proxy：
        #   obs[0] = ego_vx          车体坐标系 x 方向速度
        #   obs[1] = ego_vy          车体坐标系 y 方向速度
        #   obs[2] = yaw_rate        车体 yaw 角速度
        #   obs[3] = steering_angle  当前转向关节角
        #   obs[4] = goal_dx_body    目标在车体坐标系下的 x
        #   obs[5] = goal_dy_body    目标在车体坐标系下的 y
        #   obs[6] = goal_distance   到目标的欧氏距离
        #   obs[7] = goal_angle      目标相对车头方向角
        #   obs[8:] = lidar_ranges   车体坐标系下的归一化射线距离，1.0 表示 range 内无障碍
        obs_dim = 8 + self.cfg.lidar_num_rays
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_goal_distance = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._reset_dynamic_obstacles()
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_goal_distance = self._goal_distance()
        obs = self._get_obs()
        info = self._get_info(progress=0.0, success=False, collision=False)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self._map_action(action)

        self.data.ctrl[:] = ctrl
        collision = False
        for _ in range(self.cfg.frame_skip):
            self._update_dynamic_obstacles(self.data.time)
            mujoco.mj_step(self.model, self.data)
            self._update_dynamic_obstacles(self.data.time)
            collision = self._has_collision()
            if collision or self._has_bad_state():
                break

        self._step_count += 1
        distance = self._goal_distance()
        progress = self._prev_goal_distance - distance
        self._prev_goal_distance = distance

        bad_state = self._has_bad_state()
        collision = collision or bad_state
        success = distance <= self.cfg.goal_radius
        terminated = bool(success or collision)
        truncated = self._step_count >= self.cfg.max_episode_steps

        min_obstacle_distance = self._min_obstacle_distance()
        reward = self._compute_reward(progress, action, success, collision, min_obstacle_distance)
        obs = self._get_obs()
        info = self._get_info(progress=progress, success=success, collision=collision)
        info["bad_state"] = bool(bad_state)
        return obs, reward, terminated, truncated, info

    @property
    def max_episode_steps(self):
        return self.cfg.max_episode_steps

    def _required_id(self, obj_type, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return obj_id

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

    def _build_dynamic_specs(self) -> list[dict]:
        if self.cfg.dynamic_mode == "none":
            return []
        if self.cfg.dynamic_mode != "easy":
            raise ValueError(f"Unknown dynamic mode: {self.cfg.dynamic_mode}")

        specs = [
            {
                "name": "dyn_obs_0",
                "center": np.array([0.30, 0.75], dtype=np.float64),
                "axis": np.array([1.0, 0.0], dtype=np.float64),
                "amplitude": 0.90,
                "speed_range": (0.35, 0.65),
            },
            {
                "name": "dyn_obs_1",
                "center": np.array([2.45, -2.25], dtype=np.float64),
                "axis": np.array([0.0, 1.0], dtype=np.float64),
                "amplitude": 0.65,
                "speed_range": (0.30, 0.55),
            },
        ]
        missing = [spec["name"] for spec in specs if spec["name"] not in self.dynamic_geom_ids]
        if missing:
            raise ValueError(f"Dynamic obstacle geoms missing from XML: {missing}")
        return specs

    def _reset_dynamic_obstacles(self):
        self._dynamic_params = []
        if not self._dynamic_specs:
            return

        for idx, spec in enumerate(self._dynamic_specs):
            amplitude = spec["amplitude"]
            if self.cfg.dynamic_random_amplitude:
                amplitude *= float(self.np_random.uniform(0.90, 1.10))

            low, high = spec["speed_range"]
            max_speed = float(self.np_random.uniform(low, high)) if self.cfg.dynamic_random_speed else 0.5 * (low + high)
            omega = max_speed / max(amplitude, 1e-6)
            phase = float(self.np_random.uniform(0.0, 2.0 * np.pi)) if self.cfg.dynamic_random_phase else idx * np.pi

            self._dynamic_params.append(
                {
                    "geom_id": self.dynamic_geom_ids[spec["name"]],
                    "center": spec["center"],
                    "axis": spec["axis"],
                    "amplitude": amplitude,
                    "omega": omega,
                    "phase": phase,
                    "z": float(self.model.geom_pos[self.dynamic_geom_ids[spec["name"]], 2]),
                }
            )
        self._update_dynamic_obstacles(0.0)

    def _update_dynamic_obstacles(self, time: float):
        if not self._dynamic_params:
            return
        for params in self._dynamic_params:
            offset = params["axis"] * params["amplitude"] * np.sin(params["omega"] * time + params["phase"])
            xy = params["center"] + offset
            geom_id = params["geom_id"]
            self.model.geom_pos[geom_id, 0] = xy[0]
            self.model.geom_pos[geom_id, 1] = xy[1]
            self.model.geom_pos[geom_id, 2] = params["z"]
        mujoco.mj_forward(self.model, self.data)

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        steering = float(action[0]) * self.cfg.max_steering
        throttle = float(action[1]) * self.cfg.max_throttle
        return np.array([steering, throttle], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
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

    def _get_info(self, progress: float, success: bool, collision: bool) -> dict:
        car_pos, yaw = self._car_pose()
        return {
            "x": float(car_pos[0]),
            "y": float(car_pos[1]),
            "yaw": float(yaw),
            "distance_to_goal": self._goal_distance(),
            "progress": float(progress),
            "success": bool(success),
            "collision": bool(collision),
            "min_obstacle_distance": self._min_obstacle_distance(),
            "step": self._step_count,
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

    def _has_collision(self) -> bool:
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            car1 = body1 in self.car_body_ids
            car2 = body2 in self.car_body_ids
            if car1 == car2:
                continue
            other_geom = contact.geom2 if car1 else contact.geom1
            if other_geom in self.obstacle_geom_ids:
                return True
        return False

    def _min_obstacle_distance(self) -> float:
        car_pos, _ = self._car_pose()
        point = car_pos[:2]
        dists = [self._distance_to_geom_xy(point, geom_id) for geom_id in self.obstacle_geom_ids]
        return float(min(dists)) if dists else float("inf")

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
        for ray_idx, rel_angle in enumerate(self._lidar_angles):
            angle = yaw + rel_angle
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            for geom_id in self.obstacle_geom_ids:
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
