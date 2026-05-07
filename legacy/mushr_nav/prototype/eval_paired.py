from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from env import DEFAULT_XML, DYNAMIC_HARD_XML, MuSHRNavConfig, MuSHRNavEnv
from eval_splits import make_seed_pairs, split_names
from test_env import heading_controller


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "paired_eval"

EVAL_MODES = (
    ("static", "none", "mushr-nav-static-hard"),
    ("frozen", "frozen", "mushr-nav-dynamic-frozen"),
    ("moving", "hard", "mushr-nav-dynamic-hard"),
)


def random_policy(rng: np.random.Generator, _obs: np.ndarray) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def policy_action(policy_name: str, rng: np.random.Generator, obs: np.ndarray) -> np.ndarray:
    if policy_name == "random":
        return random_policy(rng, obs)
    if policy_name == "heading":
        return heading_controller(obs)
    raise ValueError(f"Unknown policy: {policy_name}")


def run_episode(
    env: MuSHRNavEnv,
    *,
    split: str,
    mode_name: str,
    task: str,
    policy_name: str,
    policy_seed: int,
    layout_seed: int,
    dynamic_seed: int,
    max_steps: int,
) -> dict:
    rng = np.random.default_rng(policy_seed)
    options = {"layout_seed": layout_seed}
    if mode_name != "static":
        options["dynamic_seed"] = dynamic_seed

    obs, info = env.reset(options=options)
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = policy_action(policy_name, rng, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    timeout = bool(info["timeout"] or (not info["success"] and not info["collision"]))
    return {
        "split": split,
        "mode": mode_name,
        "task": task,
        "policy": policy_name,
        "layout_seed": int(layout_seed),
        "dynamic_seed": int(dynamic_seed),
        "realized_dynamic_seed": int(info["dynamic_seed"]),
        "template": info["layout_template"],
        "success": int(info["success"]),
        "collision": int(info["collision"]),
        "collision_type": info["collision_type"],
        "static_collision": int(info["static_collision"]),
        "dynamic_collision": int(info["dynamic_collision"]),
        "timeout": int(timeout),
        "spl": float(info["spl"]),
        "path_length": float(info["path_length"]),
        "geodesic_distance": float(info["episode_geodesic_distance"]),
        "near_miss": int(info["near_miss"]),
        "min_obstacle_distance": float(info["episode_min_obstacle_distance"]),
        "dynamic_min_distance": float(info["dynamic_min_distance"]),
        "dynamic_near_miss": int(info["dynamic_near_miss"]),
        "min_ttc": float(info["min_ttc"]),
        "ttc_violation": int(info["ttc_violation"]),
        "return": float(total_reward),
        "episode_steps": int(info["step"]),
        "final_distance": float(info["distance_to_goal"]),
        "terminated": int(terminated),
        "truncated": int(truncated),
    }


def mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if len(values) else float("nan")


def mean_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("inf")


def min_finite(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.min()) if len(finite) else float("inf")


def summarize_mode(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "success_rate": mean([r["success"] for r in rows]),
        "collision_rate": mean([r["collision"] for r in rows]),
        "static_collision_rate": mean([r["static_collision"] for r in rows]),
        "dynamic_collision_rate": mean([r["dynamic_collision"] for r in rows]),
        "timeout_rate": mean([r["timeout"] for r in rows]),
        "mean_spl": mean([r["spl"] for r in rows]),
        "mean_return": mean([r["return"] for r in rows]),
        "mean_episode_steps": mean([r["episode_steps"] for r in rows]),
        "mean_path_length": mean([r["path_length"] for r in rows]),
        "mean_geodesic_distance": mean([r["geodesic_distance"] for r in rows]),
        "near_miss_rate": mean([r["near_miss"] for r in rows]),
        "dynamic_near_miss_rate": mean([r["dynamic_near_miss"] for r in rows]),
        "ttc_violation_rate": mean([r["ttc_violation"] for r in rows]),
        "mean_min_obstacle_distance": mean([r["min_obstacle_distance"] for r in rows]),
        "mean_dynamic_min_distance": mean_finite([r["dynamic_min_distance"] for r in rows]),
        "min_dynamic_min_distance": min_finite([r["dynamic_min_distance"] for r in rows]),
        "mean_min_ttc": mean_finite([r["min_ttc"] for r in rows]),
        "min_min_ttc": min_finite([r["min_ttc"] for r in rows]),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["mode"])].append(row)

    summary_rows = []
    by_split = defaultdict(dict)
    for (split, mode), mode_rows in sorted(grouped.items()):
        stats = summarize_mode(mode_rows)
        summary_row = {"split": split, "mode": mode, "comparison": ""}
        summary_row.update(stats)
        summary_rows.append(summary_row)
        by_split[split][mode] = stats

    for split, stats_by_mode in sorted(by_split.items()):
        for left, right, name in (
            ("static", "frozen", "static_minus_frozen"),
            ("frozen", "moving", "frozen_minus_moving"),
        ):
            if left not in stats_by_mode or right not in stats_by_mode:
                continue
            summary_rows.append(
                {
                    "split": split,
                    "mode": "gap",
                    "comparison": name,
                    "episodes": min(stats_by_mode[left]["episodes"], stats_by_mode[right]["episodes"]),
                    "success_rate": stats_by_mode[left]["success_rate"] - stats_by_mode[right]["success_rate"],
                    "collision_rate": stats_by_mode[left]["collision_rate"] - stats_by_mode[right]["collision_rate"],
                    "timeout_rate": stats_by_mode[left]["timeout_rate"] - stats_by_mode[right]["timeout_rate"],
                    "mean_spl": stats_by_mode[left]["mean_spl"] - stats_by_mode[right]["mean_spl"],
                    "near_miss_rate": stats_by_mode[left]["near_miss_rate"] - stats_by_mode[right]["near_miss_rate"],
                    "dynamic_near_miss_rate": stats_by_mode[left]["dynamic_near_miss_rate"]
                    - stats_by_mode[right]["dynamic_near_miss_rate"],
                    "ttc_violation_rate": stats_by_mode[left]["ttc_violation_rate"]
                    - stats_by_mode[right]["ttc_violation_rate"],
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict]):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=split_names(), default="seen")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--policy", choices=["random", "heading"], default="random")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for stochastic policy action streams.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seed_pairs = make_seed_pairs(args.split, episodes=args.episodes)

    envs = {}
    for mode_name, dynamic_mode, _task in EVAL_MODES:
        xml_path = DYNAMIC_HARD_XML if dynamic_mode != "none" else DEFAULT_XML
        envs[mode_name] = MuSHRNavEnv(
            MuSHRNavConfig(
                xml_path=xml_path,
                max_episode_steps=args.max_steps,
                dynamic_mode=dynamic_mode,
                procedural_layout=True,
            )
        )

    rows = []
    for pair_idx, pair in enumerate(seed_pairs):
        policy_seed = args.seed + pair_idx
        for mode_name, _dynamic_mode, task in EVAL_MODES:
            rows.append(
                run_episode(
                    envs[mode_name],
                    split=pair.split,
                    mode_name=mode_name,
                    task=task,
                    policy_name=args.policy,
                    policy_seed=policy_seed,
                    layout_seed=pair.layout_seed,
                    dynamic_seed=pair.dynamic_seed,
                    max_steps=args.max_steps,
                )
            )

    eval_path = args.out / "paired_eval.csv"
    summary_path = args.out / "summary.csv"
    write_csv(eval_path, rows)
    write_csv(summary_path, summarize(rows))

    print(f"split: {args.split}")
    print(f"seed pairs: {len(seed_pairs)}")
    print(f"rows: {len(rows)}")
    print(f"wrote paired eval CSV to: {eval_path}")
    print(f"wrote summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
