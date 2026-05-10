#!/bin/bash
# Throughput benchmark: 4 configs × 4 GPUs, parallel.
# Goal: pick the fastest "no-quality-loss" config before launching the real
# motivation runs. Each run is short (steps=20000) — long enough for compile
# warmup + cudnn autotune to settle, short enough to finish in ~10 min.
#
# Usage:
#   bash run_benchmark.sh                      # run all 4
#   bash run_benchmark.sh report <log_dir>     # just print FPS from existing logs

set -e
cd "$(dirname "$0")"

CONFIGS=(
    "0 bench_n1_s512_compile   num_envs=1 num_samples=512  num_elites=64  num_pi_trajs=24 compile=true"
    "1 bench_n1_s2048_compile  num_envs=1 num_samples=2048 num_elites=256 num_pi_trajs=64 compile=true"
    "2 bench_n8_s512_nocompile num_envs=8 num_samples=512  num_elites=64  num_pi_trajs=24 compile=false"
    "3 bench_n8_s512_compile   num_envs=8 num_samples=512  num_elites=64  num_pi_trajs=24 compile=true"
)

# steps small + eval_freq > steps to skip eval (pure throughput measurement)
TASK=SafetyPointGoal1-v0
COMMON="task=${TASK} seed=1 steps=20000 eval_freq=20001 log_freq=2000 model_size=1 enable_wandb=false save_video=false"

report() {
    local logdir="$1"
    echo
    echo "=== Benchmark FPS report ($logdir) ==="
    for row in "${CONFIGS[@]}"; do
        set -- $row
        local gpu=$1 exp=$2
        local log="${logdir}/${exp}.log"
        if [[ ! -f "$log" ]]; then
            printf "%-30s  (no log)\n" "$exp"
            continue
        fi
        local fps
        fps=$(grep -oE '(env_)?steps_per_second[^,}]*' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        printf "%-30s  GPU%s  FPS=%s\n" "$exp" "$gpu" "${fps:-N/A}"
    done
}

if [[ "${1:-}" == "report" ]]; then
    report "${2:-logs/bench_latest}"
    exit 0
fi

LOG_DIR="logs/bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_DIR")" logs/bench_latest

run() {
    local gpu=$1 exp=$2; shift 2
    local overrides="$*"
    local logfile="${LOG_DIR}/${exp}.log"
    echo "[GPU $gpu] $exp -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
        ${COMMON} exp_name=${exp} ${overrides} \
        > "$logfile" 2>&1 &
    echo $! > "${LOG_DIR}/${exp}.pid"
}

for row in "${CONFIGS[@]}"; do
    set -- $row
    gpu=$1 exp=$2; shift 2
    run "$gpu" "$exp" "$@"
done

echo
echo "All 4 runs launched. Logs in $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo "Waiting for all to finish..."

FAIL=0
for row in "${CONFIGS[@]}"; do
    set -- $row
    pid=$(cat "${LOG_DIR}/$2.pid")
    if ! wait "$pid"; then
        FAIL=$((FAIL+1))
    fi
done

echo "Done. Failed runs: ${FAIL}/${#CONFIGS[@]}"
report "$LOG_DIR"
