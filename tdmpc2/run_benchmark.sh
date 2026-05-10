#!/bin/bash
# Throughput benchmark: 4 configs × 4× 2080Ti, parallel.
# Goal: pick the fastest "no-quality-loss" config before launching the real
# motivation runs. Each run is short (steps=20000) — long enough for compile
# warmup + cudnn autotune to settle, short enough to finish in ~10 min.
#
# Output: tdmpc2/logs/bench_<tag>/SafetyPointGoal1-v0/1/<exp>/{train.csv,stdout.log}
# After all 4 finish, the script greps env_steps_per_second from each stdout.
#
# Usage:
#   bash run_benchmark.sh                   # run all 4
#   bash run_benchmark.sh report            # just print FPS from existing logs

set -u

PY=/home/ubuntu/miniforge3/envs/CILD/bin/python
TASK=SafetyPointGoal1-v0
SEED=1
STEPS=20000
EVAL_FREQ=20001          # skip eval during benchmark — pure throughput measurement
LOG_FREQ=2000
COMMON="task=${TASK} seed=${SEED} steps=${STEPS} eval_freq=${EVAL_FREQ} log_freq=${LOG_FREQ} enable_wandb=false save_video=false"

LOGROOT=$(pwd)/logs
mkdir -p "${LOGROOT}"

# Each row: GPU exp_name extra_overrides
CONFIGS=(
	"0 bench_n1_s512_compile   num_envs=1 num_samples=512  num_elites=64  num_pi_trajs=24 compile=true"
	"1 bench_n1_s2048_compile  num_envs=1 num_samples=2048 num_elites=256 num_pi_trajs=64 compile=true"
	"2 bench_n8_s512_nocompile num_envs=8 num_samples=512  num_elites=64  num_pi_trajs=24 compile=false"
	"3 bench_n8_s512_compile   num_envs=8 num_samples=512  num_elites=64  num_pi_trajs=24 compile=true"
)

report() {
	echo
	echo "=== Benchmark FPS report ==="
	for row in "${CONFIGS[@]}"; do
		set -- $row
		gpu=$1; exp=$2
		log="${LOGROOT}/${exp}.stdout"
		if [[ ! -f "$log" ]]; then
			printf "%-30s  (no log)\n" "$exp"
			continue
		fi
		# Grab the LAST env_steps_per_second / steps_per_second value printed.
		fps=$(grep -oE '(env_)?steps_per_second[^,}]*' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
		printf "%-30s  GPU%s  FPS=%s\n" "$exp" "$gpu" "${fps:-N/A}"
	done
}

if [[ "${1:-}" == "report" ]]; then
	report
	exit 0
fi

PIDS=()
for row in "${CONFIGS[@]}"; do
	set -- $row
	gpu=$1; exp=$2; shift 2
	overrides="$*"
	logfile="${LOGROOT}/${exp}.stdout"
	echo "[GPU${gpu}] launching: ${exp}  ->  ${logfile}"
	CUDA_VISIBLE_DEVICES=${gpu} ${PY} train.py \
		${COMMON} exp_name=${exp} ${overrides} \
		>"${logfile}" 2>&1 &
	PIDS+=($!)
done

echo "Launched PIDs: ${PIDS[*]}"
echo "Tail any log with:  tail -f ${LOGROOT}/<exp>.stdout"
echo "Waiting for all 4 to finish..."

FAIL=0
for pid in "${PIDS[@]}"; do
	if ! wait "$pid"; then
		FAIL=$((FAIL+1))
	fi
done

echo "Done. Failed runs: ${FAIL}/${#PIDS[@]}"
report
