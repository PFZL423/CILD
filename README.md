# UGV Navigation with TD-MPC2 (ugv_dync branch)

This branch extends [TD-MPC2](https://www.tdmpc2.com) with UGV (Unmanned Ground Vehicle) navigation tasks. The goal is to develop a robust model-based RL agent for mobile robot navigation, progressing from goal-reaching to dynamic obstacle avoidance, and eventually to full physics simulation.

## Status

- ✅ Goal reaching (`ugv-goal`) — 2D differential-drive, kinematic
- ✅ Unified rendering / recording infrastructure
- ✅ Debug metrics (goal_distance, path_length, out_of_bounds)
- 🚧 Static obstacle avoidance (`ugv-static-obstacle`)
- 🚧 Dynamic obstacle avoidance (`ugv-dynamic-obstacle`)
- 🚧 MuJoCo physics version

## Design Philosophy

We build the visualization, evaluation, and recording toolchain **first**, then layer obstacle complexity on top. This dramatically reduces debug cost when training on harder tasks. The same renderer and recorder support every UGV variant — kinematic, MuJoCo, and (later) real robot logs — so the analysis pipeline never gets rewritten.

Key decisions:
- **Body-frame observations** — robot perceives the world relative to itself, making the policy invariant to global pose.
- **Episodic termination** — episodes end on success or out-of-bounds, exposing terminal value to the world model.
- **Nearest-K obstacle observation** — fixed-size obstacle representation regardless of map content, ordered by distance.
- **Unified rendering** — a single renderer accepts robot state, goal, trajectory, and obstacles; all variants reuse it.

## Environment

### Task: `ugv-goal`

| Field | Value |
|---|---|
| Observation | 4D (goal in body frame, linear vel, angular vel) |
| Action | 2D continuous in `[-1, 1]` (v_cmd, omega_cmd) |
| Episode length | 300 steps |
| Success | distance to goal < 0.35 m |
| Termination | success or out-of-bounds |
| Map | 8 m × 8 m |

**Reward** combines progress toward the goal, a small control penalty, a success bonus, and an out-of-bounds penalty. See `envs/ugv/ugv_2d_env.py` for the exact form.

### Future Tasks

- **`ugv-static-obstacle`** — adds 3 fixed circular obstacles. Observation extends to 29D (4 base + 5 obstacles × 5 features).
- **`ugv-dynamic-obstacle`** — same observation, but obstacles move with constant velocity and bounce off boundaries. Robot must observe relative velocity to predict future positions.

## Repository Layout

```
tdmpc2/
├── envs/ugv/                   # UGV environment
│   ├── ugv_2d_env.py           # 2D kinematic env
│   └── dynamics.py             # Differential drive
├── common/
│   ├── ugv_rendering.py        # Unified renderer
│   ├── rollout_recorder.py     # Episode recorder
│   └── trajectory_viz.py       # Trajectory plots
├── tools/
│   └── record_ugv_rollout.py   # Recording entry point
├── train_ugv.sh                # Training launcher
├── apply_rendering_patch.sh    # One-time render setup
├── UGV_README.md               # Detailed UGV docs
├── TODO.md                     # Next steps
└── OBSTACLE_TASKS.md           # Obstacle implementation plan
```

## Usage

### Setup (once)

Inside the Docker container, apply the rendering patch:

```bash
./apply_rendering_patch.sh
```

This wires `UGV2DEnv.render()` to the unified renderer so videos contain real frames instead of blank images.

### Train

```bash
python train.py task=ugv-goal episodic=true steps=500000 model_size=1 seed=1
# or
./train_ugv.sh --size 5 --steps 100000 --eval_freq 5000
```

`episodic=true` is required because UGV episodes terminate.

### Record Rollouts

```bash
# Random policy
python tools/record_ugv_rollout.py +episodes=5 checkpoint=null

# Trained agent
python tools/record_ugv_rollout.py \
    +checkpoint=logs/ugv-goal/1/default/model.pt \
    +episodes=10
```

Each episode produces:
- `episode_NNN.mp4` — video
- `episode_NNN.png` — trajectory plot
- `episode_NNN.npz` — states, actions, rewards, success/collision flags, obstacles

### Quick Visualization

```bash
python test_ugv_rollout.py 3   # 3 random rollouts, save trajectory plots
```

## Debug Metrics

The trainer logs UGV-specific metrics at every evaluation step:

- `episode_goal_distance` — final distance to goal
- `episode_path_length` — total path traveled
- `episode_out_of_bounds` — boundary violation rate

Use these alongside `episode_reward` and `episode_success` to diagnose policy behavior — e.g., a high reward with low success often means the agent stalls near the goal.

## Roadmap

1. Static obstacles → verify the agent learns avoidance with collision penalty
2. Dynamic obstacles → verify predictive avoidance using relative velocity
3. MuJoCo physics → swap kinematic dynamics for full physics, reuse renderer/recorder unchanged
4. Real robot replay → feed real logs through the same recorder for evaluation

See `TODO.md` for concrete next steps and `OBSTACLE_TASKS.md` for the full obstacle design.

## Acknowledgement

Built on top of [TD-MPC2](https://www.tdmpc2.com) by Hansen, Su, and Wang. The base algorithm and infrastructure are unchanged; this branch adds environment, rendering, and recording on top.
