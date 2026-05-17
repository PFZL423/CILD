from __future__ import annotations

import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz

from .cild_nav_cfg import CILDNavEnvCfg


class CILDNavEnv(DirectRLEnv):
    """CILD Navigation environment with static/dynamic obstacles."""

    cfg: CILDNavEnvCfg

    def __init__(self, cfg: CILDNavEnvCfg, render_mode: str | None = None, **kwargs):
        # Must set before super().__init__ because _setup_scene is called there
        self._half_size = cfg.arena_size / 2.0

        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        device = self.device

        # Robot state (managed kinematically)
        self._robot_pos = torch.zeros(N, 3, device=device)
        self._robot_quat = torch.zeros(N, 4, device=device)
        self._robot_quat[:, 0] = 1.0  # w=1
        self._robot_yaw = torch.zeros(N, device=device)
        self._robot_lin_vel = torch.zeros(N, 2, device=device)  # body frame (vx, vy)
        self._robot_ang_vel = torch.zeros(N, device=device)

        # Goal positions (xy)
        self._goal_pos = torch.zeros(N, 2, device=device)
        self._last_goal_dist = torch.zeros(N, device=device)

        # Hazard positions (fixed per episode)
        self._hazard_pos = torch.zeros(N, self.cfg.num_hazards, 2, device=device)

        # Vase positions and velocities
        self._vase_pos = torch.zeros(N, self.cfg.num_vases, 2, device=device)
        self._vase_vel = torch.zeros(N, self.cfg.num_vases, 2, device=device)
        self._vase_step_counter = torch.zeros(N, dtype=torch.long, device=device)

        # Frame stack buffer
        single_obs_dim = self.cfg.observation_space
        self._obs_buffer = torch.zeros(N, self.cfg.frame_stack, single_obs_dim, device=device)

        # Actions
        self._actions = torch.zeros(N, 2, device=device)

        # Override observation_space for frame-stacked output
        self.cfg.observation_space = single_obs_dim * self.cfg.frame_stack

        # Collision flag (computed in _get_dones, read in _get_rewards)
        self._in_collision = torch.zeros(N, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # We don't need PhysX simulation — all dynamics are computed analytically.
        # Just spawn a ground plane so DirectRLEnv's scene is valid.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments (required by DirectRLEnv)
        self.scene.clone_environments(copy_from_source=False)

        # Precompute lidar ray directions (body frame, 32 rays, 360 degrees)
        num_rays = self.cfg.lidar_num_rays
        angles = torch.linspace(0, 2 * math.pi, num_rays + 1, device=self.device)[:-1]
        self._lidar_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (32, 2)

        # Lights (for visualization only)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        N = self.num_envs
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Decode actions
        forward_vel = self._actions[:, 0] * self.cfg.max_linear_vel
        yaw_rate = self._actions[:, 1] * self.cfg.max_angular_vel

        # Update yaw
        self._robot_yaw += yaw_rate * dt
        # Wrap to [-pi, pi]
        self._robot_yaw = torch.atan2(torch.sin(self._robot_yaw), torch.cos(self._robot_yaw))

        # Update position (unicycle model)
        dx = forward_vel * torch.cos(self._robot_yaw) * dt
        dy = forward_vel * torch.sin(self._robot_yaw) * dt
        self._robot_pos[:, 0] += dx
        self._robot_pos[:, 1] += dy

        # Clamp to arena bounds (wall collision)
        limit = self._half_size - self.cfg.robot_radius
        self._robot_pos[:, 0].clamp_(-limit, limit)
        self._robot_pos[:, 1].clamp_(-limit, limit)

        # Store velocity in body frame
        self._robot_lin_vel[:, 0] = forward_vel
        self._robot_lin_vel[:, 1] = 0.0  # no lateral velocity for unicycle
        self._robot_ang_vel = yaw_rate

        # Update robot quaternion
        zeros = torch.zeros(N, device=self.device)
        self._robot_quat = quat_from_euler_xyz(zeros, zeros, self._robot_yaw)

        # Update dynamic vases
        if self.cfg.obstacle_mode != "static":
            self._step_vases(dt)

    def _step_vases(self, dt: float):
        """Move vases according to obstacle_mode."""
        N = self.num_envs
        nv = self.cfg.num_vases

        if self.cfg.obstacle_mode == "random":
            # Random direction change
            self._vase_step_counter += 1
            change_mask = (self._vase_step_counter % self.cfg.obstacle_turn_interval == 0)
            if change_mask.any():
                n_change = change_mask.sum().item()
                angles = torch.rand(n_change, nv, device=self.device) * 2 * math.pi
                speed = self.cfg.obstacle_speed
                new_vel = torch.stack([torch.cos(angles) * speed, torch.sin(angles) * speed], dim=-1)
                self._vase_vel[change_mask] = new_vel

        # Move vases
        self._vase_pos += self._vase_vel * dt

        # Bounce off walls
        limit = self._half_size - self.cfg.vase_radius
        for dim in range(2):
            over_max = self._vase_pos[:, :, dim] > limit
            under_min = self._vase_pos[:, :, dim] < -limit
            self._vase_pos[:, :, dim] = self._vase_pos[:, :, dim].clamp(-limit, limit)
            self._vase_vel[:, :, dim] = torch.where(over_max | under_min, -self._vase_vel[:, :, dim], self._vase_vel[:, :, dim])

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        N = self.num_envs

        # Ego velocity (body frame)
        ego_vx = self._robot_lin_vel[:, 0:1]
        ego_vy = self._robot_lin_vel[:, 1:2]
        yaw_rate = self._robot_ang_vel.unsqueeze(-1)
        steering = self._robot_yaw.unsqueeze(-1)

        # Goal in body frame
        goal_world = self._goal_pos - self._robot_pos[:, :2]
        cos_yaw = torch.cos(self._robot_yaw).unsqueeze(-1)
        sin_yaw = torch.sin(self._robot_yaw).unsqueeze(-1)
        goal_dx_body = (goal_world[:, 0:1] * cos_yaw + goal_world[:, 1:2] * sin_yaw)
        goal_dy_body = (-goal_world[:, 0:1] * sin_yaw + goal_world[:, 1:2] * cos_yaw)
        goal_distance = torch.norm(goal_world, dim=-1, keepdim=True)
        goal_angle = torch.atan2(goal_dy_body, goal_dx_body)

        # Lidar ranges (normalized) — computed analytically
        lidar_ranges = self._compute_lidar()  # (N, 32), already normalized [0, 1]

        # Concatenate single frame obs
        obs_single = torch.cat([
            ego_vx, ego_vy, yaw_rate, steering,
            goal_dx_body, goal_dy_body, goal_distance, goal_angle,
            lidar_ranges,
        ], dim=-1)  # (N, 40)

        # Frame stack: shift buffer and insert new frame
        if self.cfg.frame_stack > 1:
            self._obs_buffer = torch.roll(self._obs_buffer, shifts=-1, dims=1)
            self._obs_buffer[:, -1] = obs_single
            obs = self._obs_buffer.reshape(N, -1)
        else:
            obs = obs_single

        return {"policy": obs}

    def _compute_lidar(self) -> torch.Tensor:
        """Compute 32-ray pseudo-lidar by ray-circle and ray-line intersection."""
        N = self.num_envs
        num_rays = self.cfg.lidar_num_rays
        max_dist = self.cfg.lidar_max_dist

        robot_xy = self._robot_pos[:, :2]  # (N, 2)
        yaw = self._robot_yaw  # (N,)

        # Rotate lidar directions to world frame
        cos_yaw = torch.cos(yaw)  # (N,)
        sin_yaw = torch.sin(yaw)  # (N,)
        # self._lidar_dirs: (32, 2) — local frame
        dirs_local = self._lidar_dirs.unsqueeze(0).expand(N, -1, -1)  # (N, 32, 2)
        dirs_world_x = dirs_local[:, :, 0] * cos_yaw.unsqueeze(1) - dirs_local[:, :, 1] * sin_yaw.unsqueeze(1)
        dirs_world_y = dirs_local[:, :, 0] * sin_yaw.unsqueeze(1) + dirs_local[:, :, 1] * cos_yaw.unsqueeze(1)
        dirs_world = torch.stack([dirs_world_x, dirs_world_y], dim=-1)  # (N, 32, 2)

        # Initialize distances to max
        distances = torch.full((N, num_rays), max_dist, device=self.device)

        # Ray-circle intersection with all obstacles (hazards + vases)
        all_obs_pos = torch.cat([self._hazard_pos, self._vase_pos], dim=1)  # (N, 13, 2)
        all_obs_radii = torch.cat([
            torch.full((N, self.cfg.num_hazards), self.cfg.hazard_radius, device=self.device),
            torch.full((N, self.cfg.num_vases), self.cfg.vase_radius, device=self.device),
        ], dim=1)  # (N, 13)

        num_obs = all_obs_pos.shape[1]
        # Vector from robot to obstacle center: (N, num_obs, 2)
        to_obs = all_obs_pos - robot_xy.unsqueeze(1)

        # For each ray, compute intersection with each circle
        # Project obstacle center onto ray direction
        # dot = to_obs . dir for each ray
        # to_obs: (N, num_obs, 2), dirs_world: (N, num_rays, 2)
        # We need (N, num_rays, num_obs)
        dot = torch.einsum('nrd,nod->nro', dirs_world, to_obs)  # (N, 32, 13)
        # Perpendicular distance squared
        to_obs_sq = (to_obs ** 2).sum(dim=-1)  # (N, 13)
        perp_sq = to_obs_sq.unsqueeze(1) - dot ** 2  # (N, 32, 13)

        radii_sq = (all_obs_radii ** 2).unsqueeze(1)  # (N, 1, 13)
        hit_mask = (perp_sq < radii_sq) & (dot > 0)  # ray must go toward obstacle

        # Distance to intersection point: dot - sqrt(r^2 - perp^2)
        safe_perp_sq = perp_sq.clamp(max=radii_sq - 1e-6)
        hit_dist = dot - torch.sqrt((radii_sq - safe_perp_sq).clamp(min=0))  # (N, 32, 13)
        hit_dist = torch.where(hit_mask, hit_dist, torch.tensor(max_dist, device=self.device))

        # Min across obstacles
        obs_min_dist = hit_dist.min(dim=-1).values  # (N, 32)
        distances = torch.min(distances, obs_min_dist)

        # Ray-line intersection with walls (4 walls)
        hs = self._half_size
        # Wall segments: (x_min, y_min, x_max, y_max) or as lines
        # North wall: y = hs, East wall: x = hs, South wall: y = -hs, West wall: x = -hs
        # For axis-aligned walls, intersection is simple:
        # North (y=hs): t = (hs - robot_y) / dir_y
        # South (y=-hs): t = (-hs - robot_y) / dir_y
        # East (x=hs): t = (hs - robot_x) / dir_x
        # West (x=-hs): t = (-hs - robot_x) / dir_x
        robot_x = robot_xy[:, 0:1]  # (N, 1)
        robot_y = robot_xy[:, 1:2]  # (N, 1)
        dir_x = dirs_world[:, :, 0]  # (N, 32)
        dir_y = dirs_world[:, :, 1]  # (N, 32)

        eps = 1e-8
        # North wall
        t_north = (hs - robot_y) / (dir_y + eps)
        x_hit_north = robot_x + t_north * dir_x
        valid_north = (t_north > 0) & (x_hit_north.abs() <= hs)

        # South wall
        t_south = (-hs - robot_y) / (dir_y + eps)
        x_hit_south = robot_x + t_south * dir_x
        valid_south = (t_south > 0) & (x_hit_south.abs() <= hs)

        # East wall
        t_east = (hs - robot_x) / (dir_x + eps)
        y_hit_east = robot_y + t_east * dir_y
        valid_east = (t_east > 0) & (y_hit_east.abs() <= hs)

        # West wall
        t_west = (-hs - robot_x) / (dir_x + eps)
        y_hit_west = robot_y + t_west * dir_y
        valid_west = (t_west > 0) & (y_hit_west.abs() <= hs)

        wall_dists = torch.stack([
            torch.where(valid_north, t_north, torch.tensor(max_dist, device=self.device)),
            torch.where(valid_south, t_south, torch.tensor(max_dist, device=self.device)),
            torch.where(valid_east, t_east, torch.tensor(max_dist, device=self.device)),
            torch.where(valid_west, t_west, torch.tensor(max_dist, device=self.device)),
        ], dim=-1)  # (N, 32, 4)
        wall_min_dist = wall_dists.min(dim=-1).values  # (N, 32)
        distances = torch.min(distances, wall_min_dist)

        # Normalize
        return (distances / max_dist).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        N = self.num_envs
        robot_xy = self._robot_pos[:, :2]

        # Goal distance
        goal_dist = torch.norm(robot_xy - self._goal_pos, dim=-1)
        goal_reached = goal_dist < self.cfg.goal_radius

        # Progress reward
        progress = (self._last_goal_dist - goal_dist) * self.cfg.rew_scale_progress
        self._last_goal_dist = goal_dist.clone()

        # Goal reward
        goal_rew = torch.where(goal_reached, torch.tensor(self.cfg.rew_scale_goal, device=self.device), torch.zeros(1, device=self.device))

        # Collision reward (collision already computed in _get_dones)
        in_collision = self._in_collision

        # Collision reward
        if self.cfg.terminate_on_collision:
            collision_rew = torch.where(in_collision, torch.tensor(self.cfg.rew_scale_collision_term, device=self.device), torch.zeros(1, device=self.device))
        else:
            collision_rew = torch.where(in_collision, torch.tensor(self.cfg.rew_scale_collision_cost, device=self.device), torch.zeros(1, device=self.device))

        # Alive reward
        alive_rew = torch.full((N,), self.cfg.rew_scale_alive, device=self.device)

        total_reward = progress + goal_rew + collision_rew + alive_rew

        # Respawn goal if reached
        if goal_reached.any():
            self._respawn_goal(goal_reached.nonzero(as_tuple=False).squeeze(-1))

        # Privileged info for CILD cost head labels
        label_goal_dist = torch.norm(robot_xy - self._goal_pos, dim=-1)
        lidar_ranges_m = self._compute_lidar() * self.cfg.lidar_max_dist
        min_lidar_dist = lidar_ranges_m.min(dim=-1).values
        collision_threshold = float(getattr(self.cfg, "collision_lidar_threshold", 0.3))
        self.extras["cost"] = in_collision.float()
        self.extras["obstacle_positions"] = torch.cat([self._hazard_pos, self._vase_pos], dim=1)
        self.extras["obstacle_velocities"] = torch.cat([
            torch.zeros_like(self._hazard_pos), self._vase_vel,
        ], dim=1)
        self.extras["goal_reached"] = goal_reached.float()
        self.extras["collision_flag"] = (min_lidar_dist < collision_threshold).float()
        self.extras["min_lidar_dist"] = min_lidar_dist.float()
        self.extras["goal_dist"] = label_goal_dist.float()

        return total_reward

    def _check_collision(self) -> torch.Tensor:
        """Check if robot collides with any hazard or vase."""
        robot_xy = self._robot_pos[:, :2]

        # Check hazards
        diff_h = robot_xy.unsqueeze(1) - self._hazard_pos  # (N, num_hazards, 2)
        dist_h = torch.norm(diff_h, dim=-1)  # (N, num_hazards)
        hit_hazard = (dist_h < (self.cfg.robot_radius + self.cfg.hazard_radius)).any(dim=-1)

        # Check vases
        diff_v = robot_xy.unsqueeze(1) - self._vase_pos  # (N, num_vases, 2)
        dist_v = torch.norm(diff_v, dim=-1)  # (N, num_vases)
        hit_vase = (dist_v < (self.cfg.robot_radius + self.cfg.vase_radius)).any(dim=-1)

        return hit_hazard | hit_vase

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Compute collision here since _get_dones is called before _get_rewards
        self._in_collision = self._check_collision()

        if self.cfg.terminate_on_collision:
            terminated = self._in_collision
        else:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device)

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Reset robot position
        robot_xy = (torch.rand(n, 2, device=self.device) * 2 - 1) * self.cfg.robot_spawn_range
        self._robot_pos[env_ids, 0] = robot_xy[:, 0]
        self._robot_pos[env_ids, 1] = robot_xy[:, 1]
        self._robot_pos[env_ids, 2] = self.cfg.robot_height / 2

        # Random yaw
        self._robot_yaw[env_ids] = (torch.rand(n, device=self.device) * 2 - 1) * math.pi
        zeros = torch.zeros(n, device=self.device)
        self._robot_quat[env_ids] = quat_from_euler_xyz(zeros, zeros, self._robot_yaw[env_ids])

        # Reset velocities
        self._robot_lin_vel[env_ids] = 0.0
        self._robot_ang_vel[env_ids] = 0.0

        # Place hazards randomly (fixed for episode)
        self._hazard_pos[env_ids] = self._sample_positions(n, self.cfg.num_hazards, self.cfg.hazard_radius)

        # Place vases randomly
        self._vase_pos[env_ids] = self._sample_positions(n, self.cfg.num_vases, self.cfg.vase_radius)

        # Initialize vase velocities for dynamic modes
        if self.cfg.obstacle_mode != "static":
            angles = torch.rand(n, self.cfg.num_vases, device=self.device) * 2 * math.pi
            speed = self.cfg.obstacle_speed
            self._vase_vel[env_ids] = torch.stack([
                torch.cos(angles) * speed,
                torch.sin(angles) * speed,
            ], dim=-1)
        else:
            self._vase_vel[env_ids] = 0.0

        self._vase_step_counter[env_ids] = 0

        # Place goal
        self._respawn_goal(env_ids)

        # Reset frame stack buffer
        self._obs_buffer[env_ids] = 0.0

        # Reset collision flag
        self._in_collision[env_ids] = False

    def _sample_positions(self, n_envs: int, n_objects: int, obj_radius: float) -> torch.Tensor:
        """Sample random non-overlapping positions within arena bounds."""
        limit = self._half_size - obj_radius - self.cfg.wall_thickness
        positions = (torch.rand(n_envs, n_objects, 2, device=self.device) * 2 - 1) * limit
        return positions

    def _respawn_goal(self, env_ids: torch.Tensor):
        """Respawn goal for given environments, ensuring minimum distance from robot."""
        n = len(env_ids)
        limit = self._half_size - self.cfg.goal_radius - self.cfg.wall_thickness
        robot_xy = self._robot_pos[env_ids, :2]

        for _ in range(100):  # rejection sampling
            candidates = (torch.rand(n, 2, device=self.device) * 2 - 1) * limit
            dist_to_robot = torch.norm(candidates - robot_xy, dim=-1)
            valid = dist_to_robot > self.cfg.goal_min_dist
            if valid.all():
                self._goal_pos[env_ids] = candidates
                break
            # Replace invalid ones
            if valid.any():
                self._goal_pos[env_ids[valid]] = candidates[valid]
            # Retry only invalid
            env_ids = env_ids[~valid]
            n = len(env_ids)
            robot_xy = self._robot_pos[env_ids, :2]
            if n == 0:
                break

        # Recompute for all originally passed env_ids
        self._last_goal_dist = torch.norm(self._robot_pos[:, :2] - self._goal_pos, dim=-1)
