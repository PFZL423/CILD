#!/bin/bash
python train.py task=walker-walk-static-obstacle steps=500000 model_size=1 seed=1 enable_wandb=false eval_freq=5000 eval_episodes=1 && python train.py task=walker-walk-dynamic-obstacle steps=500000 model_size=1 seed=1 enable_wandb=false eval_freq=5000 eval_episodes=1
