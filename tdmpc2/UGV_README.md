# UGV Navigation Tasks for TD-MPC2

This directory contains UGV (Unmanned Ground Vehicle) navigation tasks integrated with TD-MPC2.

## Overview

The UGV environment implements a 2D differential-drive robot navigating to goal positions. It serves as a foundation for testing TD-MPC2 on mobile robot navigation with plans to extend to obstacle avoidance and MuJoCo physics simulation.

**Current Status:**
- ✅ Goal reaching task (`ugv-goal`)
- ✅ Unified rendering and recording infrastructure
- ✅ Debug metrics (goal_distance, path_length, out_of_bounds)
- 🚧 Static obstacle avoidance (planned)
- 🚧 Dynamic obstacle avoidance (planned)
- 🚧 MuJoCo physics version (planned)

## Quick Start

### Training

```bash
# Basic training
python train.py task=ugv-goal episodic=true steps=500000 model_size=1 seed=1

# Using the training script
./train_ugv.sh --size 5 --steps 100000 --eval_freq 5000
```

**Important:** Always set `episodic=true` for UGV tasks (environment terminates on success/failure).

### Recording Rollouts

```bash
# Random policy
python tools/record_ugv_rollout.py +episodes=5 checkpoint=null

# Trained agent
python tools/record_ugv_rollout.py \
    +checkpoint=logs/ugv-goal/1/default/model.pt \
    +episodes=10

# Output: logs/ugv-goal/1/default/rollouts/
#   episode_000.mp4   (video)
#   episode_000.png   (trajectory plot)
#   episode_000.npz   (episode data)
```

### Visualization

```bash
# Simple trajectory visualization
python test_ugv_rollout.py 3
```

## Environment Details

### Task: `ugv-goal`

**Observation Space:** 4D continuous
- `goal_dx_body`: Goal x-position in robot body frame
- `goal_dy_body`: Goal y-position in robot body frame
- `v`: Linear velocity
- `omega`: Angular velocity

**Action Space:** 2D continuous `[-1, 1]`
- `v_cmd`: Linear velocity command
- `omega_cmd`: Angular velocity command

**Episode:**
- Length: 300 steps
- Success: Distance to goal < 0.35m
- Termination: Success or out of bounds
- Map size: 8m × 8m

**Reward:**
```python
reward = 10.0 * progress          # Progress toward goal
       - 0.01 * action_cost       # Control penalty
       + 50.0 (if success)        # Success bonus
       - 20.0 (if out_of_bounds)  # Boundary penalty
```

## File Structure

```
tdmpc2/
├── envs/ugv/
│   ├── __init__.py              # Task registration
│   ├── ugv_2d_env.py           # Main environment
│   └── dynamics.py             # Differential drive dynamics
├── common/
│   ├── ugv_rendering.py        # Unified renderer
│   ├── rollout_recorder.py     # Episode recorder
│   └── trajectory_viz.py       # Trajectory plotting
├── tools/
│   └── record_ugv_rollout.py   # Recording script
├── test_ugv_rollout.py         # Simple visualization
├── train_ugv.sh                # Training launcher
├── apply_rendering_patch.sh    # Setup script (run once)
├── TODO.md                     # Next steps
└── OBSTACLE_TASKS.md          # Obstacle implementation plan
```

## Setup

### 1. Apply Rendering Patch (One-time)

The environment needs rendering support. Run inside Docker as root:

```bash
./apply_rendering_patch.sh
```

This updates `envs/ugv/ugv_2d_env.py` to use the unified renderer.

### 2. Verify Setup

```bash
python tools/record_ugv_rollout.py +episodes=1 checkpoint=null
```

Should generate video with robot, goal, and trajectory (not white frames).

## Architecture

### Unified Rendering

All UGV variants (2D kinematic, future MuJoCo) use the same renderer:

```python
from common.ugv_rendering import UGVRenderer

renderer = UGVRenderer(map_size, robot_radius, goal_radius)
frame = renderer.render(
    robot_state=robot_state,
    goal=goal,
    trajectory=trajectory,
    obstacles=obstacles,  # Future: static/dynamic
    step_count=step_count
)
```

### Episode Recording

```python
from common.rollout_recorder import RolloutRecorder

recorder = RolloutRecorder(save_dir, save_video=True, save_traj=True)
recorder.reset()
recorder.record_step(state, action, reward, frame)
recorder.save_episode(episode_idx, info)
```

Outputs:
- `.mp4` - Video with rendered frames
- `.png` - Trajectory plot
- `.npz` - Episode data (states, actions, rewards, success, collision, obstacles)

## Debug Metrics

The trainer logs UGV-specific metrics during evaluation:

- `episode_reward` - Total reward
- `episode_success` - Success rate
- `episode_collision` - Collision rate (future)
- `episode_length` - Episode length
- `episode_goal_distance` - Final distance to goal
- `episode_path_length` - Total path traveled
- `episode_out_of_bounds` - Boundary violation rate

## Future Work

See `OBSTACLE_TASKS.md` for detailed implementation plans.

### Static Obstacles (`ugv-static-obstacle`)

- Observation: 29D (4 base + 5×5 obstacles)
- Each obstacle: `[dx_body, dy_body, dvx_body, dvy_body, radius]`
- Collision detection and clearance penalty
- Success criteria: >70% success, <10% collision

### Dynamic Obstacles (`ugv-dynamic-obstacle`)

- Same observation as static
- Obstacles move with velocity
- Bounce off boundaries
- Success criteria: >50% success, <20% collision

### MuJoCo Physics

- Create `UGVMuJoCoEnv` with same interface
- Reuse all rendering/recording infrastructure
- Add realistic dynamics, friction, inertia

## Design Principles

Following CLAUDE.md:

1. **Simplicity** - Minimal code, no speculative features
2. **Surgical changes** - Only touch what's needed
3. **Reusability** - Same infrastructure for all UGV variants
4. **Goal-driven** - Clear success criteria for each phase

## Citation

If you use this UGV integration, please cite the original TD-MPC2 paper:

```bibtex
@article{hansen2023tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  journal={arXiv preprint arXiv:2310.16828},
  year={2023}
}
```
