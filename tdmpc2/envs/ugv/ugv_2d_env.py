import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.ugv.dynamics import step_differential_drive, body_frame_vector, wrap_angle
from common.ugv_rendering import UGVRenderer


class UGV2DEnv(gym.Env):
    """
    Minimal 2D differential-drive UGV environment.

    Version 1:
    - No obstacles
    - Goal reaching only
    - State observation
    - Continuous action [v_cmd, omega_cmd]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str = "goal",
        seed: int = 0,
        map_size: float = 8.0,
        dt: float = 0.1,
        episode_length: int = 300,
        robot_radius: float = 0.25,
        goal_radius: float = 0.35,
        v_max: float = 1.0,
        omega_max: float = 1.5,
        acc_limit: float = 1.0,
        omega_acc_limit: float = 3.0,
    ):
        super().__init__()

        self.task = task
        self.map_size = map_size
        self.dt = dt
        self.episode_length = episode_length

        self.robot_radius = robot_radius
        self.goal_radius = goal_radius

        self.v_max = v_max
        self.omega_max = omega_max
        self.acc_limit = acc_limit
        self.omega_acc_limit = omega_acc_limit

        self.rng = np.random.default_rng(seed)

        # TD-MPC2 expects continuous Box action.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # obs = [goal_dx_body, goal_dy_body, v, omega]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.robot_state = None
        self.goal = None
        self.step_count = 0
        self.prev_goal_dist = None
        self.path_length = 0.0
        self.trajectory = []

        # Renderer
        self._renderer = UGVRenderer(
            map_size=self.map_size,
            robot_radius=self.robot_radius,
            goal_radius=self.goal_radius
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.path_length = 0.0
        self.trajectory = []

        # Renderer
        self._renderer = UGVRenderer(
            map_size=self.map_size,
            robot_radius=self.robot_radius,
            goal_radius=self.goal_radius
        )

        # Start near left side, goal near right side.
        x = self.rng.uniform(-0.8 * self.map_size / 2, -0.4 * self.map_size / 2)
        y = self.rng.uniform(-1.0, 1.0)
        theta = self.rng.uniform(-0.3, 0.3)

        self.robot_state = np.array([x, y, theta, 0.0, 0.0], dtype=np.float32)

        gx = self.rng.uniform(0.4 * self.map_size / 2, 0.8 * self.map_size / 2)
        gy = self.rng.uniform(-1.5, 1.5)
        self.goal = np.array([gx, gy], dtype=np.float32)

        self.prev_goal_dist = self._goal_distance()

        self.trajectory = [self.robot_state[:2].copy()]

        obs = self._get_obs()
        info = self._get_info(success=False, collision=False)
        return obs, info

    def step(self, action):
        old_xy = self.robot_state[:2].copy()

        self.robot_state = step_differential_drive(
            state=self.robot_state,
            action=np.asarray(action, dtype=np.float32),
            dt=self.dt,
            v_max=self.v_max,
            omega_max=self.omega_max,
            acc_limit=self.acc_limit,
            omega_acc_limit=self.omega_acc_limit,
        )

        new_xy = self.robot_state[:2]
        self.path_length += float(np.linalg.norm(new_xy - old_xy))
        self.step_count += 1
        self.trajectory.append(self.robot_state[:2].copy())

        goal_dist = self._goal_distance()
        progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist

        success = goal_dist < self.goal_radius
        out_of_bounds = np.any(np.abs(self.robot_state[:2]) > self.map_size / 2)
        timeout = self.step_count >= self.episode_length

        # Reward: progress + success bonus - mild control penalty
        action = np.asarray(action, dtype=np.float32)
        reward = 10.0 * progress - 0.01 * float(np.sum(np.square(action)))

        if success:
            reward += 50.0

        if out_of_bounds:
            reward -= 20.0

        terminated = bool(success or out_of_bounds)
        truncated = bool(timeout and not terminated)

        obs = self._get_obs()
        info = self._get_info(success=success, collision=False)
        info["out_of_bounds"] = bool(out_of_bounds)

        return obs, float(reward), terminated, truncated, info

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.robot_state[:2]))

    def _get_obs(self) -> np.ndarray:
        x, y, theta, v, omega = self.robot_state

        dx = float(self.goal[0] - x)
        dy = float(self.goal[1] - y)

        goal_body = body_frame_vector(dx, dy, theta)

        obs = np.array(
            [
                goal_body[0],
                goal_body[1],
                v,
                omega,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self, success: bool, collision: bool) -> dict:
        return {
            "success": float(success),
            "collision": float(collision),
            "goal_distance": self._goal_distance(),
            "path_length": self.path_length,
            "episode_step": self.step_count,
            "x": float(self.robot_state[0]),
            "y": float(self.robot_state[1]),
            "theta": float(wrap_angle(float(self.robot_state[2]))),
        }

    def render(self):
        """
        Render the environment using the unified renderer.
        Returns RGB array for video recording.
        """
        return self._renderer.render(
            robot_state=self.robot_state,
            goal=self.goal,
            trajectory=self.trajectory,
            obstacles=getattr(self, "obstacles", None),
            map_size=self.map_size,
            robot_radius=self.robot_radius,
            goal_radius=self.goal_radius,
            step_count=self.step_count,
        )
