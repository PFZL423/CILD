# MuSHR Static/Dynamic Hard Navigation

This repo extends TD-MPC2 with a MuJoCo MuSHR benchmark for comparing complex static navigation against the same layouts with time-varying dynamic obstacles.

## Implemented

Main files:

```text
prototypes/mujoco_mushr_nav/
tdmpc2/envs/mushr_nav.py
tdmpc2/evaluate_mushr_traj.py
```

Tasks:

```text
mushr-nav-static-hard       # procedural static hard maps
mushr-nav-dynamic-frozen    # generated dynamic obstacles, frozen
mushr-nav-dynamic-hard      # same layouts plus moving obstacles
```

Features:

- Static-hard maps: random start/goal, narrow passages, corners, dead ends, clutter, reachability checks.
- Dynamic-hard: layout-aware moving obstacles for crossing, bottleneck, intersection, occlusion, and goal-area risk.
- Frozen control: same dynamic obstacle bodies as moving mode, but velocity zero.
- Observation: 4-frame history, raw 40 dim -> TD-MPC2 160 dim.
- Metrics: SPL, path length, geodesic distance, near-miss, TTC, timeout, static/dynamic collision type.
- Collision logging: `train.csv` and `eval.csv` include total, static, and dynamic episode collision counts.
- Paired evaluation: same `layout_seed + dynamic_seed` under `static`, `frozen`, and `moving`.

## Environment Checks

```bash
cd ~/self_project/tdmpc2
conda run --no-capture-output -n tdmpc2 python prototypes/mujoco_mushr_nav/test_env.py --dynamic-mode hard --episodes 1 --max-steps 500
conda run --no-capture-output -n tdmpc2 python prototypes/mujoco_mushr_nav/test_env_batch.py --dynamic-mode hard --episodes 20 --max-steps 500 --policy heading --out /tmp/mushr_heading_dynamic_check
```

Batch output: `heading_episodes.csv`, `heading_trajectories.csv`, `heading_trajectories.png`.

## Paired Evaluation

```bash
cd ~/self_project/tdmpc2
conda run --no-capture-output -n tdmpc2 python prototypes/mujoco_mushr_nav/eval_paired.py --split seen --episodes 10 --max-steps 500 --policy random --out /tmp/mushr_eval_seen_random
```

Outputs: `paired_eval.csv`, `summary.csv`.

Splits in `prototypes/mujoco_mushr_nav/eval_splits.py`: `train-smoke`, `seen`, `unseen-layout`, `unseen-dynamic`, `combo`.

## Training

Training requires CUDA. Use `save_video=false` because the MuSHR TD-MPC2 wrapper has no RGB render.

```bash
cd ~/self_project/tdmpc2/tdmpc2
conda activate tdmpc2
```

Smoke:

```bash
python -u train.py task=mushr-nav-dynamic-hard model_size=1 steps=5000 eval_freq=2500 eval_episodes=3 save_video=false enable_wandb=false compile=false seed=1 exp_name=smoke_dynamic
```

50K debug:

```bash
python -u train.py task=mushr-nav-dynamic-hard model_size=1 steps=50000 eval_freq=10000 eval_episodes=3 checkpoint_freq=10000 save_video=false save_agent=true enable_wandb=false compile=false seed=1 exp_name=debug_dynamic_50k
```

500K:

```bash
python -u train.py task=mushr-nav-dynamic-hard model_size=1 steps=500000 eval_freq=25000 eval_episodes=10 checkpoint_freq=50000 save_video=false save_agent=true enable_wandb=false compile=false seed=1 exp_name=dynamic_500k_seed1
```

For fair comparison, run the same hyperparameters on `mushr-nav-static-hard`, `mushr-nav-dynamic-frozen`, and `mushr-nav-dynamic-hard`.

## Logs

```text
tdmpc2/logs/<task>/<seed>/<exp_name>/train.csv
tdmpc2/logs/<task>/<seed>/<exp_name>/eval.csv
tdmpc2/logs/<task>/<seed>/<exp_name>/models/*.pt
```

```bash
tail -f logs/mushr-nav-dynamic-hard/1/smoke_dynamic/train.csv
tail -f logs/mushr-nav-dynamic-hard/1/smoke_dynamic/eval.csv
```

`R` is episode return, not single-step reward. Returns like `-40` can be normal in failed 500-step hard episodes because penalties accumulate.

Collision columns:

```text
episode_collision          # total collision count in the episode
episode_static_collision   # collisions with active static obstacles/walls
episode_dynamic_collision  # collisions with active dynamic obstacles
```

## Checkpoint Trajectories

```bash
cd ~/self_project/tdmpc2/tdmpc2
python -u evaluate_mushr_traj.py task=mushr-nav-dynamic-hard model_size=1 checkpoint=logs/mushr-nav-dynamic-hard/1/debug_dynamic_50k/models/50000.pt save_video=false enable_wandb=false compile=false +eval_split=seen +traj_episodes=10 +traj_max_steps=500 +plot_episodes=6 +out_dir=/tmp/mushr_learned_traj_50k
```

Outputs: `learned_episodes.csv`, `learned_trajectories.csv`, `summary.csv`, `learned_trajectories.png`.

The script rolls out the same checkpoint under `static`, `frozen`, and `moving` for each seed pair.

Seed rules:

- Training `seed` controls model initialization, exploration, and log directory.
- `layout_seed` controls map/start/goal.
- `dynamic_seed` controls obstacle pattern/speed/phase.
- Evaluation seeds do not need to equal checkpoint training seed.
- Keep `eval_split` fixed across methods.

## Caveats

- `--fixed-layout` cannot be combined with `--dynamic-mode hard/frozen`.
- Observations are state + lidar history, not RGB/depth.
- Dynamic obstacle generation is layout-aware but not a full time-expanded planner; inspect suspicious seeds with trajectory CSV/PNG.
