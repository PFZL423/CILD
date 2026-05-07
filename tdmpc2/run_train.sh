#!/bin/bash
# Baseline TD-MPC2 on Safety Gymnasium navigation tasks (motivation experiment).
# Uses default TD-MPC2 hyperparameters — do NOT tune reward / cfg overrides here.
# Per RESEARCH_NOTES.md: baseline performance is an observation, not a tunable.

set -e

python train.py task=SafetyPointGoal1-v0 steps=500000 model_size=1 seed=1 enable_wandb=false eval_freq=5000 eval_episodes=1
python train.py task=SafetyPointGoal2-v0 steps=500000 model_size=1 seed=1 enable_wandb=false eval_freq=5000 eval_episodes=1
python train.py task=SafetyCarGoal1-v0   steps=500000 model_size=1 seed=1 enable_wandb=false eval_freq=5000 eval_episodes=1
