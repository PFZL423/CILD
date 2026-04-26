# TODO: UGV Development

## Immediate (Required for rendering)

### Apply rendering patch
```bash
./apply_rendering_patch.sh
```

Verifies rendering works:
```bash
python tools/record_ugv_rollout.py +episodes=1 checkpoint=null
```

Expected: Video shows robot, goal, trajectory (not white frames).

---

## Phase 1: Static Obstacles

### 1.1 Update environment (envs/ugv/ugv_2d_env.py)

Add to `__init__`:
```python
num_obstacles: int = 0,
obstacle_mode: str = "none",
obstacle_radius: float = 0.35,
nearest_k: int = 5,
```

Update observation space:
```python
obs_dim = 4 + self.nearest_k * 5  # Base + K obstacles
```

### 1.2 Add obstacle initialization

```python
def _reset_obstacles(self):
    """Initialize static obstacles."""
    if self.num_obstacles == 0:
        self.obstacles = np.zeros((0, 5), dtype=np.float32)
        return
    
    obstacles = []
    for _ in range(self.num_obstacles):
        ox = self.rng.uniform(-1.0, 1.0)
        oy = self.rng.uniform(-2.0, 2.0)
        # Ensure no overlap with start/goal
        while (np.linalg.norm([ox - self.robot_state[0], oy - self.robot_state[1]]) < 1.0 or
               np.linalg.norm([ox - self.goal[0], oy - self.goal[1]]) < 1.0):
            ox = self.rng.uniform(-1.0, 1.0)
            oy = self.rng.uniform(-2.0, 2.0)
        
        obstacles.append([ox, oy, 0.0, 0.0, self.obstacle_radius])
    
    self.obstacles = np.array(obstacles, dtype=np.float32)
```

### 1.3 Update observation

```python
def _get_obs(self):
    """Add nearest K obstacles to observation."""
    from envs.ugv.dynamics import body_frame_vector
    
    x, y, theta, v, omega = self.robot_state
    
    # Goal in body frame
    dx = float(self.goal[0] - x)
    dy = float(self.goal[1] - y)
    goal_body = body_frame_vector(dx, dy, theta)
    
    obs = [goal_body[0], goal_body[1], v, omega]
    
    # Add nearest K obstacles
    if len(self.obstacles) > 0:
        robot_xy = self.robot_state[:2]
        distances = np.linalg.norm(self.obstacles[:, :2] - robot_xy, axis=1)
        nearest_indices = np.argsort(distances)[:self.nearest_k]
        
        for idx in nearest_indices:
            obs_state = self.obstacles[idx]
            dx = obs_state[0] - x
            dy = obs_state[1] - y
            pos_body = body_frame_vector(dx, dy, theta)
            obs.extend([pos_body[0], pos_body[1], 0.0, 0.0, obs_state[4]])
        
        # Pad with zeros if fewer than K obstacles
        for _ in range(self.nearest_k - len(nearest_indices)):
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        obs.extend([0.0] * (self.nearest_k * 5))
    
    return np.array(obs, dtype=np.float32)
```

### 1.4 Add collision detection

```python
def _check_collision(self):
    """Check robot-obstacle collision."""
    if len(self.obstacles) == 0:
        return False
    
    robot_xy = self.robot_state[:2]
    for obs in self.obstacles:
        dist = np.linalg.norm(robot_xy - obs[:2])
        if dist < self.robot_radius + obs[4]:
            return True
    return False

def _min_obstacle_distance(self):
    """Return minimum distance and clearance."""
    if len(self.obstacles) == 0:
        return float('inf'), float('inf')
    
    robot_xy = self.robot_state[:2]
    distances = np.linalg.norm(self.obstacles[:, :2] - robot_xy, axis=1)
    min_dist = float(np.min(distances))
    min_idx = np.argmin(distances)
    clearance = min_dist - self.robot_radius - self.obstacles[min_idx, 4]
    return min_dist, clearance
```

### 1.5 Update step() reward

```python
# Check collision
collision = self._check_collision()
if collision:
    self.collision_total += 1

# Add clearance penalty
if len(self.obstacles) > 0:
    min_dist, clearance = self._min_obstacle_distance()
    if clearance < 0.5:
        near_penalty = np.exp(-max(clearance, 0.0) / 0.3)
        reward -= 0.2 * near_penalty

if collision:
    reward -= 100.0
```

### 1.6 Register task (envs/ugv/__init__.py)

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
}

def make_env(cfg):
    task_cfg = _TASKS[task_name]
    env = UGV2DEnv(
        task=task_cfg["task"],
        seed=int(cfg.seed),
        episode_length=int(task_cfg["episode_length"]),
        num_obstacles=int(task_cfg.get("num_obstacles", 0)),
        obstacle_mode=task_cfg.get("obstacle_mode", "none"),
        obstacle_radius=float(task_cfg.get("obstacle_radius", 0.35)),
        nearest_k=5,
    )
    # ... rest unchanged
```

### 1.7 Test

```bash
python train.py task=ugv-static-obstacle episodic=true steps=10000 model_size=1 seed=1
python tools/record_ugv_rollout.py task=ugv-static-obstacle +episodes=5 checkpoint=null
```

---

## Phase 2: Dynamic Obstacles

### 2.1 Add dynamic updates

```python
def _update_obstacles(self):
    """Update dynamic obstacle positions."""
    if len(self.obstacles) == 0 or self.obstacle_mode != "dynamic":
        return
    
    self.obstacles[:, 0] += self.obstacles[:, 2] * self.dt  # x += vx * dt
    self.obstacles[:, 1] += self.obstacles[:, 3] * self.dt  # y += vy * dt
    
    # Bounce off boundaries
    half_size = self.map_size / 2
    for i in range(len(self.obstacles)):
        if abs(self.obstacles[i, 0]) > half_size:
            self.obstacles[i, 2] *= -1
        if abs(self.obstacles[i, 1]) > half_size:
            self.obstacles[i, 3] *= -1
```

Call in `step()`:
```python
if self.obstacle_mode == "dynamic":
    self._update_obstacles()
```

### 2.2 Initialize with velocity

In `_reset_obstacles()`:
```python
if self.obstacle_mode == "dynamic":
    angle = self.rng.uniform(0, 2 * np.pi)
    speed = self.rng.uniform(0.3, 0.5)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
else:
    vx, vy = 0.0, 0.0
```

### 2.3 Register task

```python
"ugv-dynamic-obstacle": {
    "task": "dynamic_obstacle",
    "episode_length": 300,
    "num_obstacles": 3,
    "obstacle_mode": "dynamic",
    "obstacle_radius": 0.35,
    "obstacle_speed": 0.5,
},
```

### 2.4 Test

```bash
python train.py task=ugv-dynamic-obstacle episodic=true steps=10000 model_size=1 seed=1
python tools/record_ugv_rollout.py task=ugv-dynamic-obstacle +episodes=5 checkpoint=null
```

---

## Phase 3: MuJoCo Physics (Future)

See OBSTACLE_TASKS.md for details.

Key: Create `UGVMuJoCoEnv` with same interface, reuse all rendering/recording infrastructure.
