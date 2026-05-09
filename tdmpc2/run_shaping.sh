#!/bin/bash
# Reward-shaping control experiment for CILD motivation.
# 4 GPUs, one run each: PointGoal1/2 x lambda in {0.3, 1.0}.
# Vanilla baseline (lambda=0) already done -- see commit c761c78.

set -e
cd "$(dirname "$0")"

LOG_DIR="logs/shaping_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run() {
    local gpu=$1 task=$2 lam=$3
    local tag="${task}_lam${lam}"
    echo "[GPU $gpu] $tag -> $LOG_DIR/${tag}.log"
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
        task=$task \
        cost_lambda=$lam \
        steps=500000 model_size=1 seed=1 \
        enable_wandb=false eval_freq=5000 eval_episodes=1 \
        exp_name=shaping_lam${lam} \
        > "$LOG_DIR/${tag}.log" 2>&1 &
    echo $! > "$LOG_DIR/${tag}.pid"
}

run 0 SafetyPointGoal1-v0 0.3
run 1 SafetyPointGoal1-v0 1.0
run 2 SafetyPointGoal2-v0 0.3
run 3 SafetyPointGoal2-v0 1.0

echo "All 4 runs launched. Logs in $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
wait
echo "All runs finished."
