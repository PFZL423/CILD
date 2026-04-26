#!/bin/bash

set -e

MODEL_SIZE=5
STEPS=10000
EVAL_FREQ=5000
TRAJECTORY_FREQ=1000

usage() {
  echo "Usage: $0 [MODEL_SIZE STEPS EVAL_FREQ TRAJECTORY_FREQ]"
  echo "   or: $0 --size MODEL_SIZE --steps STEPS --eval_freq EVAL_FREQ --traj_freq TRAJECTORY_FREQ"
}

if [[ $# -gt 0 ]]; then
  if [[ $1 == --help || $1 == -h ]]; then
    usage
    exit 0
  fi

  if [[ $1 == --* ]]; then
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --size)
          MODEL_SIZE="$2"
          shift 2
          ;;
        --steps)
          STEPS="$2"
          shift 2
          ;;
        --eval_freq)
          EVAL_FREQ="$2"
          shift 2
          ;;
        --traj_freq)
          TRAJECTORY_FREQ="$2"
          shift 2
          ;;
        *)
          echo "Unknown option: $1"
          usage
          exit 1
          ;;
      esac
    done
  else
    if [[ $# -gt 4 ]]; then
      echo "Too many positional arguments."
      usage
      exit 1
    fi

    [[ $# -ge 1 ]] && MODEL_SIZE="$1"
    [[ $# -ge 2 ]] && STEPS="$2"
    [[ $# -ge 3 ]] && EVAL_FREQ="$3"
    [[ $# -ge 4 ]] && TRAJECTORY_FREQ="$4"
  fi
fi

echo ""
echo "=== Training... ==="
python train.py \
  task=ugv-goal \
  episodic=true \
  steps=${STEPS} \
  model_size=${MODEL_SIZE} \
  seed=1 \
  enable_wandb=false \
  eval_freq=${EVAL_FREQ} \
  eval_episodes=5 \
  save_video=false \
  save_trajectory=true \
  trajectory_freq=${TRAJECTORY_FREQ} \
  compile=false

echo ""
echo "=== All tests passed! ==="
