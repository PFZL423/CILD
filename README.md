# CILD: Cost-Informed Latent Dynamics

A research codebase extending TD-MPC2 with **planner-coupled prediction heads** so latent representation learning and navigation cost design are co-designed rather than independent problems.

## Research path

The motivation experiments use **public navigation/safety benchmarks** (Safety Gymnasium) rather than custom environments — see `RESEARCH_NOTES.md` for why this matters.

- **Step 1 (current)**: Establish that latent world models (TD-MPC2) are systematically limited on navigation/avoidance tasks, using Safety Gymnasium as the benchmark.
- **Step 2**: Demonstrate that CILD — adding risk / progress / occupancy heads whose loss backpropagates through the latent dynamics chain — closes the gap.
- **Step 3 (later)**: Generalization to more complex sim2real-relevant scenes.

## Repository layout

```
tdmpc2/                  # TD-MPC2 core (vendored, with light modifications)
  envs/safety_gym.py     # Safety Gymnasium wrapper (entry point for benchmarks)
  ...
legacy/                  # Archived code, not on the main path
  mushr_nav/             # Custom MuJoCo MuSHR navigation env (Step 3 candidate)
RESEARCH_NOTES.md        # Research log and reasoning
CLAUDE.md                # Project instructions for Claude Code
```

## Setup

The project uses a single conda environment (`CILD`) with `safety-gymnasium` installed. See `docker/environment.yaml` for the full dependency list.

```bash
conda activate CILD
pip install safety-gymnasium
```

## Run

```bash
cd tdmpc2
python train.py task=SafetyPointGoal1-v0   # static
python train.py task=SafetyPointGoal2-v0   # dynamic
python train.py task=SafetyCarGoal1-v0     # different morphology
```

## Status

The repository is mid-refactor: switching from custom MuSHR environments to Safety Gymnasium benchmarks. See git log and `RESEARCH_NOTES.md` for context.
