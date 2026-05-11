#!/bin/bash
# Reward-shaping control experiment for CILD motivation (1M-step run).
# 4 GPUs, one run each: PointGoal1/2 x lambda in {0.3, 1.0}.
# Vanilla baseline (lambda=0) already done -- see commit c761c78.
#
# Config picked from the throughput benchmark (run_benchmark.sh):
#   num_envs=8, num_samples=512 was fastest on 2080Ti without sacrificing
#   MPPI quality. compile=true is benchmark-only — see FAST comment below.
#   model_size=5 matches the lambda=0 baseline (commit c761c78, single-env)
#   so the three shaping levels stay comparable.
#
# Stair-step pattern in train R is expected: 8 envs reset in sync, all
# episodes truncate together every 1000 ticks (= 8000 env_step_total), so
# the deque-mean only updates on those bursts. Use eval R /
# episode_raw_reward / episode_goal_reached_count for the actual learning
# signal between bursts.

set -e
cd "$(dirname "$0")"

# Force line-buffered stdout in all child python processes so episode log
# lines stream into the per-run *.log files in real time. Without this,
# nohup-redirected python defaults to block-buffered (4KB) and the train
# console output is invisible for tens of minutes at a stretch.
export PYTHONUNBUFFERED=1

# Timestamp goes into BOTH the host-side log dir AND exp_name, so re-running
# this script on the same day cannot overwrite a prior run's train.csv.
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/shaping_${TS}"
mkdir -p "$LOG_DIR"

# compile=false: torch.compile fast-paths churn the dynamo guard cache on
#   long runs (Adam.step + EMA target net), collapsing throughput from
#   ~10 step/s to ~0.9 step/s. benchmark numbers were a short-window
#   artifact; do not flip back to true for >10k-step runs.
FAST="num_envs=8 num_samples=512 num_elites=64 num_pi_trajs=24 compile=false"

run() {
    local gpu=$1 task=$2 lam=$3
    local tag="${task}_lam${lam}"
    local exp="shaping_lam${lam}_${TS}"
    echo "[GPU $gpu] $tag -> $LOG_DIR/${tag}.log  (exp=${exp})"
    # eval_freq must be >= num_envs * episode_length (8 * 1000 = 8000),
    # otherwise eval's env.reset() preempts envs before they can truncate
    # naturally, so the train deque never fills and train.csv stays NaN.
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u train.py \
        task=$task \
        cost_lambda=$lam \
        steps=1000000 model_size=5 seed=1 \
        ${FAST} \
        enable_wandb=false eval_freq=10000 eval_episodes=1 save_video=false \
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
