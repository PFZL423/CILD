# Commit Summary: UGV Navigation Integration

## Overview

Integrated UGV (Unmanned Ground Vehicle) navigation tasks into TD-MPC2 with unified rendering and recording infrastructure.

## Changes

### Core Infrastructure

**New Files:**
- `common/ugv_rendering.py` - Unified renderer for all UGV variants
- `common/rollout_recorder.py` - Episode recorder (video, trajectory, data)
- `common/trajectory_viz.py` - Trajectory plotting utilities
- `tools/record_ugv_rollout.py` - Recording script for trained/random policies

**Modified Files:**
- `envs/wrappers/tensor.py` - Fixed Gymnasium API compatibility (5-value step return)
- `trainer/online_trainer.py` - Added debug metrics (goal_distance, path_length, out_of_bounds)
- `envs/__init__.py` - Added UGV environment import

### UGV Environment

**New Files:**
- `envs/ugv/__init__.py` - Task registration
- `envs/ugv/ugv_2d_env.py` - 2D differential-drive environment
- `envs/ugv/dynamics.py` - Kinematic dynamics

**Task:** `ugv-goal`
- Observation: 4D (goal_dx_body, goal_dy_body, v, omega)
- Action: 2D continuous (v_cmd, omega_cmd)
- Episode: 300 steps, terminates on success/out-of-bounds
- Reward: Progress + success bonus - control penalty - boundary penalty

### Scripts & Documentation

**Scripts:**
- `train_ugv.sh` - Training launcher with configurable parameters
- `apply_rendering_patch.sh` - One-time setup for rendering
- `test_ugv_rollout.py` - Simple visualization script

**Documentation:**
- `UGV_README.md` - Complete usage guide
- `TODO.md` - Next steps (obstacle tasks)
- `OBSTACLE_TASKS.md` - Detailed obstacle implementation plan

## Key Features

### 1. Unified Rendering
Single renderer supports:
- Current: 2D kinematic UGV
- Future: Static obstacles, dynamic obstacles, MuJoCo physics

### 2. Episode Recording
Records to:
- `.mp4` - Video with rendered frames
- `.png` - Trajectory plot
- `.npz` - Episode data (states, actions, rewards, metadata)

### 3. Debug Metrics
Trainer logs:
- `episode_goal_distance` - Final distance to goal
- `episode_path_length` - Total path traveled
- `episode_out_of_bounds` - Boundary violations

### 4. Gymnasium API Compatibility
Fixed `TensorWrapper` to handle:
- New API: `reset()` returns `(obs, info)`
- New API: `step()` returns 5 values (obs, reward, terminated, truncated, info)
- Backward compatibility with old API

## Usage

### Training
```bash
python train.py task=ugv-goal episodic=true steps=500000 model_size=1 seed=1
# or
./train_ugv.sh --size 5 --steps 100000
```

### Recording
```bash
# Random policy
python tools/record_ugv_rollout.py +episodes=5 checkpoint=null

# Trained agent
python tools/record_ugv_rollout.py \
    +checkpoint=logs/ugv-goal/1/default/model.pt \
    +episodes=10
```

## Setup Required

Run once inside Docker as root:
```bash
./apply_rendering_patch.sh
```

This updates `envs/ugv/ugv_2d_env.py` to use the unified renderer.

## Design Principles

Following CLAUDE.md:
1. **Simplicity** - Minimal code, no speculative features
2. **Surgical changes** - Only modified what's needed
3. **Reusability** - Infrastructure works for all UGV variants
4. **Goal-driven** - Clear success criteria

## Future Work

See `TODO.md` and `OBSTACLE_TASKS.md`:
1. Static obstacle avoidance (29D obs, collision detection)
2. Dynamic obstacle avoidance (moving obstacles)
3. MuJoCo physics version (realistic dynamics)

## Testing

```bash
# Verify rendering works
python tools/record_ugv_rollout.py +episodes=1 checkpoint=null

# Should generate video with robot, goal, trajectory (not white frames)
```

## Files Changed

```
Modified:
  envs/wrappers/tensor.py
  trainer/online_trainer.py
  envs/__init__.py

Added:
  envs/ugv/
  common/ugv_rendering.py
  common/rollout_recorder.py
  common/trajectory_viz.py
  tools/record_ugv_rollout.py
  train_ugv.sh
  apply_rendering_patch.sh
  test_ugv_rollout.py
  UGV_README.md
  TODO.md
  OBSTACLE_TASKS.md
```
