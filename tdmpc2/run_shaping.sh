#!/bin/bash
# Reward-shaping control experiment for CILD motivation (1M-step run).
# 4 GPUs, one run each: PointGoal1/2 x lambda in {0.3, 1.0}.
# Vanilla baseline (lambda=0) already done -- see commit c761c78.
#
# Config picked from the throughput benchmark (run_benchmark.sh):
#   num_envs=8, num_samples=512, compile=true was fastest on 2080Ti without
#   sacrificing MPPI quality. model_size=5 matches the lambda=0 baseline so
#   the three shaping levels stay comparable.
#
# Stair-step pattern in train R is expected: 8 envs reset in sync, episodes
# finish in synchronous bursts every 500 ticks, deque-mean updates only on
# bursts. Use eval R / episode_raw_reward / episode_goal_reached_count for
# the actual learning signal.

set -e
cd "$(dirname "$0")"

# Timestamp goes into BOTH the host-side log dir AND exp_name, so re-running
# this script on the same day cannot overwrite a prior run's train.csv.
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/shaping_${TS}"
mkdir -p "$LOG_DIR"

# Optimal-throughput config from benchmark.
FAST="num_envs=8 num_samples=512 num_elites=64 num_pi_trajs=24 compile=true"

run() {
    local gpu=$1 task=$2 lam=$3
    local tag="${task}_lam${lam}"
    local exp="shaping_lam${lam}_${TS}"
    echo "[GPU $gpu] $tag -> $LOG_DIR/${tag}.log  (exp=${exp})"
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
        task=$task \
        cost_lambda=$lam \
        steps=1000000 model_size=5 seed=1 \
        ${FAST} \
        enable_wandb=false eval_freq=5000 eval_episodes=1 save_video=false \
        exp_name=${exp} \
        > "$LOG_DIR/${tag}.log" 2>&1 &
    echo $! > "$LOG_DIR/${tag}.pid"
}

run 0 SafetyPointGoal1-v0 0.3
run 1 SafetyPointGoal1-v0 1.0
run 2 SafetyPointGoal2-v0 0.3
run 3 SafetyPointGoal2-v0 1.0

echo "All 4 runs launched. Logs in $LOG_DIR/"
echo "Monitor:  tail -f $LOG_DIR/*.log"
echo "GPU util: watch -n 5 nvidia-smi"
wait
echo "All runs finished."
