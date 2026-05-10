from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class CILDNavEnvCfg(DirectRLEnvCfg):
    # --- env timing ---
    decimation = 4
    episode_length_s = 33.3  # 1000 steps * decimation * dt = 1000 * 4 * (1/120)
    action_space = 2
    observation_space = 40  # single frame; multiplied by frame_stack at runtime
    state_space = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=10.0,
        replicate_physics=True,
    )

    # --- arena ---
    arena_size: float = 4.0  # half-extent is arena_size/2 = 2.0m
    wall_height: float = 0.3
    wall_thickness: float = 0.05

    # --- robot ---
    robot_radius: float = 0.1
    robot_height: float = 0.2
    max_linear_vel: float = 1.0  # m/s
    max_angular_vel: float = 1.5  # rad/s

    # --- obstacles ---
    num_hazards: int = 8
    hazard_radius: float = 0.2
    num_vases: int = 5
    vase_radius: float = 0.1
    obstacle_mode: str = "static"  # static / linear / random
    obstacle_speed: float = 0.3  # m/s for dynamic vases
    obstacle_turn_interval: int = 50  # steps between random direction changes

    # --- goal ---
    goal_radius: float = 0.3
    goal_min_dist: float = 1.0  # minimum spawn distance from robot
    goal_keepout: float = 0.5  # minimum distance from obstacles

    # --- reward ---
    rew_scale_progress: float = 1.0
    rew_scale_goal: float = 10.0
    rew_scale_collision_term: float = -10.0  # when terminate_on_collision=True
    rew_scale_collision_cost: float = -1.0  # when terminate_on_collision=False
    rew_scale_alive: float = 0.01
    terminate_on_collision: bool = True

    # --- observation ---
    frame_stack: int = 1  # 1 for debugging, 4 for experiments
    lidar_num_rays: int = 32
    lidar_max_dist: float = 4.0

    # --- reset ---
    robot_spawn_range: float = 1.0  # robot spawns in [-range, range]
