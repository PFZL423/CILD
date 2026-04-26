# Obstacle Task Implementation Plan

## Overview

Extend UGV environment to support static and dynamic obstacles while maintaining compatibility with the existing goal-reaching task.

---

## 1. Task Registration (envs/ugv/__init__.py)

```python
_TASKS = {
    "ugv-goal": {
        "task": "goal",
        "episode_length": 300,
        "num_obstacles": 0,
        "obstacle_mode": "none",
    },
    "ugv-static-obstacle": {
        "task": "static_obstacle",
        "episode_length": 300,
        "num_obstacles": 3,
        "obstacle_mode": "static",
        "obstacle_radius": 0.35,
    },
    "ugv-dynamic-obstacle": {
        "task": "dynamic_obstacle",
        "episode_length": 300,
        "num_obstacles": 3,
        "obstacle_mode": "dynamic",
        "obstacle_radius": 0.35,
        "obstacle_speed": 0.5,
    },
}


def make_env(cfg):
    """Factory for UGV environments."""
    task_name = str(cfg.task)

    if task_name not in _TASKS:
        raise ValueError(f"Unknown UGV task: {task_name}")

    task_cfg = _TASKS[task_name]

    env = UGV2DEnv(
        task=task_cfg["task"],
        seed=int(cfg.seed),
        episode_length=int(task_cfg["episode_length"]),
        num_obstacles=int(task_cfg.get("num_obstacles", 0)),
        obstacle_mode=task_cfg.get("obstacle_mode", "none"),
        obstacle_radius=float(task_cfg.get("obstacle_radius", 0.35)),
        obstacle_speed=float(task_cfg.get("obstacle_speed", 0.5)),
        nearest_k=5,  # Only observe nearest K obstacles
    )

    env = TimeLimit(env, max_episode_steps=int(task_cfg["episode_length"]))
    env.max_episode_steps = int(task_cfg["episode_length"])
    return env
```

---

## 2. UGV2DEnv Updates (envs/ugv/ugv_2d_env.py)

### 2.1 Add to __init__

```python
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
    # NEW: Obstacle parameters
    num_obstacles: int = 0,
    obstacle_mode: str = "none",
    obstacle_radius: float = 0.35,
    obstacle_speed: float = 0.5,
    nearest_k: int = 5,
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

    # NEW: Obstacle parameters
    self.num_obstacles = num_obstacles
    self.obstacle_mode = obstacle_mode
    self.obstacle_radius = obstacle_radius
    self.obstacle_speed = obstacle_speed
    self.nearest_k = nearest_k

    self.rng = np.random.default_rng(seed)

    # Observation space
    # Base: [goal_dx_body, goal_dy_body, v, omega]
    # + K obstacles: [dx_body, dy_body, dvx_body, dvy_body, radius] each
    obs_dim = 4 + self.nearest_k * 5
    
    self.action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )

    self.robot_state = None
    self.goal = None
    self.obstacles = None  # [N, 5]: [x, y, vx, vy, radius]
    self.step_count = 0
    self.prev_goal_dist = None
    self.path_length = 0.0
    self.trajectory = []
    self.collision_total = 0

    # Renderer
    from common.ugv_rendering import UGVRenderer
    self._renderer = UGVRenderer(
        map_size=self.map_size,
        robot_radius=self.robot_radius,
        goal_radius=self.goal_radius
    )
```

### 2.2 Update reset()

```python
def reset(self, seed=None, options=None):
    if seed is not None:
        self.rng = np.random.default_rng(seed)

    self.step_count = 0
    self.path_length = 0.0
    self.collision_total = 0

    # Robot start position
    x = self.rng.uniform(-0.8 * self.map_size / 2, -0.4 * self.map_size / 2)
    y = self.rng.uniform(-1.0, 1.0)
    theta = self.rng.uniform(-0.3, 0.3)
    self.robot_state = np.array([x, y, theta, 0.0, 0.0], dtype=np.float32)

    # Goal position
    gx = self.rng.uniform(0.4 * self.map_size / 2, 0.8 * self.map_size / 2)
    gy = self.rng.uniform(-1.5, 1.5)
    self.goal = np.array([gx, gy], dtype=np.float32)

    # Initialize obstacles
    self._reset_obstacles()

    self.prev_goal_dist = self._goal_distance()
    self.trajectory = [self.robot_state[:2].copy()]

    obs = self._get_obs()
    info = self._get_info(success=False, collision=False)
    return obs, info


def _reset_obstacles(self):
    """Initialize obstacles based on task mode."""
    if self.num_obstacles == 0 or self.obstacle_mode == "none":
        self.obstacles = np.zeros((0, 5), dtype=np.float32)
        return

    obstacles = []
    
    for _ in range(self.num_obstacles):
        # Place obstacles in middle region between start and goal
        ox = self.rng.uniform(-1.0, 1.0)
        oy = self.rng.uniform(-2.0, 2.0)
        
        # Ensure obstacle doesn't overlap with start or goal
        while (np.linalg.norm([ox - self.robot_state[0], oy - self.robot_state[1]]) < 1.0 or
               np.linalg.norm([ox - self.goal[0], oy - self.goal[1]]) < 1.0):
            ox = self.rng.uniform(-1.0, 1.0)
            oy = self.rng.uniform(-2.0, 2.0)
        
        if self.obstacle_mode == "static":
            vx, vy = 0.0, 0.0
        elif self.obstacle_mode == "dynamic":
            # Random velocity for dynamic obstacles
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(0.3, self.obstacle_speed)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        else:
            vx, vy = 0.0, 0.0
        
        obstacles.append([ox, oy, vx, vy, self.obstacle_radius])
    
    self.obstacles = np.array(obstacles, dtype=np.float32)
```

### 2.3 Update step()

```python
def step(self, action):
    old_xy = self.robot_state[:2].copy()

    # Update robot
    self.robot_state = step_differential_drive(
        state=self.robot_state,
        action=np.asarray(action, dtype=np.float32),
        dt=self.dt,
        v_max=self.v_max,
        omega_max=self.omega_max,
        acc_limit=self.acc_limit,
        omega_acc_limit=self.omega_acc_limit,
    )

    # Update obstacles (dynamic mode)
    if self.obstacle_mode == "dynamic":
        self._update_obstacles()

    new_xy = self.robot_state[:2]
    self.path_length += float(np.linalg.norm(new_xy - old_xy))
    self.step_count += 1
    self.trajectory.append(self.robot_state[:2].copy())

    # Check collisions
    collision = self._check_collision()
    if collision:
        self.collision_total += 1

    # Check success and termination
    goal_dist = self._goal_distance()
    progress = self.prev_goal_dist - goal_dist
    self.prev_goal_dist = goal_dist

    success = goal_dist < self.goal_radius
    out_of_bounds = np.any(np.abs(self.robot_state[:2]) > self.map_size / 2)
    timeout = self.step_count >= self.episode_length

    # Reward
    action = np.asarray(action, dtype=np.float32)
    reward = 10.0 * progress - 0.01 * float(np.sum(np.square(action)))

    if success:
        reward += 50.0

    if collision:
        reward -= 100.0

    if out_of_bounds:
        reward -= 20.0

    # Add clearance penalty (soft constraint)
    if len(self.obstacles) > 0:
        min_dist, clearance = self._min_obstacle_distance()
        if clearance < 0.5:  # Within 0.5m of obstacle
            near_penalty = np.exp(-max(clearance, 0.0) / 0.3)
            reward -= 0.2 * near_penalty

    terminated = bool(success or out_of_bounds or collision)
    truncated = bool(timeout and not terminated)

    obs = self._get_obs()
    info = self._get_info(success=success, collision=collision)
    info["out_of_bounds"] = bool(out_of_bounds)
    
    if len(self.obstacles) > 0:
        min_dist, clearance = self._min_obstacle_distance()
        info["min_obstacle_distance"] = float(min_dist)
        info["clearance"] = float(clearance)

    return obs, float(reward), terminated, truncated, info


def _update_obstacles(self):
    """Update dynamic obstacle positions."""
    if len(self.obstacles) == 0:
        return
    
    # Update positions
    self.obstacles[:, 0] += self.obstacles[:, 2] * self.dt  # x += vx * dt
    self.obstacles[:, 1] += self.obstacles[:, 3] * self.dt  # y += vy * dt
    
    # Bounce off boundaries
    half_size = self.map_size / 2
    for i in range(len(self.obstacles)):
        if abs(self.obstacles[i, 0]) > half_size:
            self.obstacles[i, 2] *= -1  # Reverse vx
        if abs(self.obstacles[i, 1]) > half_size:
            self.obstacles[i, 3] *= -1  # Reverse vy


def _check_collision(self):
    """Check if robot collides with any obstacle."""
    if len(self.obstacles) == 0:
        return False
    
    robot_xy = self.robot_state[:2]
    for obs in self.obstacles:
        obs_xy = obs[:2]
        dist = np.linalg.norm(robot_xy - obs_xy)
        if dist < self.robot_radius + obs[4]:  # obs[4] is radius
            return True
    return False


def _min_obstacle_distance(self):
    """Return minimum distance to obstacles and clearance."""
    if len(self.obstacles) == 0:
        return float('inf'), float('inf')
    
    robot_xy = self.robot_state[:2]
    distances = np.linalg.norm(self.obstacles[:, :2] - robot_xy, axis=1)
    min_dist = float(np.min(distances))
    min_idx = np.argmin(distances)
    clearance = min_dist - self.robot_radius - self.obstacles[min_idx, 4]
    return min_dist, clearance
```

### 2.4 Update _get_obs()

```python
def _get_obs(self) -> np.ndarray:
    """
    Get observation.
    
    Base: [goal_dx_body, goal_dy_body, v, omega]
    + K nearest obstacles: [dx_body, dy_body, dvx_body, dvy_body, radius] each
    """
    from envs.ugv.dynamics import body_frame_vector
    
    x, y, theta, v, omega = self.robot_state

    # Goal in body frame
    dx = float(self.goal[0] - x)
    dy = float(self.goal[1] - y)
    goal_body = body_frame_vector(dx, dy, theta)

    obs = [goal_body[0], goal_body[1], v, omega]

    # Add nearest K obstacles
    if len(self.obstacles) > 0:
        # Find nearest K obstacles
        robot_xy = self.robot_state[:2]
        distances = np.linalg.norm(self.obstacles[:, :2] - robot_xy, axis=1)
        nearest_indices = np.argsort(distances)[:self.nearest_k]
        
        for idx in nearest_indices:
            obs_state = self.obstacles[idx]
            
            # Position in body frame
            dx = obs_state[0] - x
            dy = obs_state[1] - y
            pos_body = body_frame_vector(dx, dy, theta)
            
            # Velocity in body frame
            vx, vy = obs_state[2:4]
            vel_body = body_frame_vector(vx, vy, theta)
            
            # Radius
            radius = obs_state[4]
            
            obs.extend([pos_body[0], pos_body[1], vel_body[0], vel_body[1], radius])
        
        # Pad with zeros if fewer than K obstacles
        for _ in range(self.nearest_k - len(nearest_indices)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # No obstacles - pad with zeros
        obs.extend([0.0] * (self.nearest_k * 5))

    return np.array(obs, dtype=np.float32)
```

---

## 3. Testing Plan

### Phase 1: Static Obstacles
```bash
# Train on static obstacles
python train.py \
    task=ugv-static-obstacle \
    episodic=true \
    steps=100000 \
    model_size=1 \
    seed=1

# Record rollouts
python tools/record_ugv_rollout.py \
    task=ugv-static-obstacle \
    checkpoint=logs/ugv-static-obstacle/1/default/model.pt \
    episodes=10
```

### Phase 2: Dynamic Obstacles
```bash
# Train on dynamic obstacles
python train.py \
    task=ugv-dynamic-obstacle \
    episodic=true \
    steps=100000 \
    model_size=1 \
    seed=1

# Record rollouts
python tools/record_ugv_rollout.py \
    task=ugv-dynamic-obstacle \
    checkpoint=logs/ugv-dynamic-obstacle/1/default/model.pt \
    episodes=10
```

---

## 4. Expected Observation Dimensions

- **ugv-goal**: 4D (goal_dx, goal_dy, v, omega)
- **ugv-static-obstacle**: 29D (4 + 5*5 for K=5 obstacles)
- **ugv-dynamic-obstacle**: 29D (same, but velocities are non-zero)

---

## 5. Success Criteria

### Static Obstacles
- Success rate > 70%
- Collision rate < 10%
- Agent clearly navigates around obstacles in videos

### Dynamic Obstacles
- Success rate > 50%
- Collision rate < 20%
- Agent shows predictive avoidance behavior
