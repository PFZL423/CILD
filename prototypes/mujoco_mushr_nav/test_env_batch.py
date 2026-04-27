import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import mujoco
import numpy as np

from env import DEFAULT_XML, DYNAMIC_EASY_XML, MuSHRNavConfig, MuSHRNavEnv
from test_env import heading_controller


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "batch_check"


def random_policy(rng: np.random.Generator, _obs: np.ndarray) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def draw_layout(env: MuSHRNavEnv, ax):
    for geom_id in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = env.model.geom_type[geom_id]
        pos = env.data.geom_xpos[geom_id]
        size = env.model.geom_size[geom_id]

        if name == "floor":
            ax.add_patch(Rectangle((-5, -5), 10, 10, facecolor="#eeeeee", edgecolor="none", zorder=0))
        elif "wall" in name:
            ax.add_patch(
                Rectangle(
                    (pos[0] - size[0], pos[1] - size[1]),
                    2 * size[0],
                    2 * size[1],
                    facecolor="#59616a",
                    edgecolor="#30353a",
                    linewidth=0.8,
                    zorder=2,
                )
            )
        elif "obs" in name:
            if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                ax.add_patch(Circle((pos[0], pos[1]), size[0], facecolor="#c7603d", edgecolor="#78351f", zorder=3))
            else:
                ax.add_patch(
                    Rectangle(
                        (pos[0] - size[0], pos[1] - size[1]),
                        2 * size[0],
                        2 * size[1],
                        facecolor="#c7603d",
                        edgecolor="#78351f",
                        zorder=3,
                    )
                )


def plot_trajectories(env: MuSHRNavEnv, trajectories, out_path: Path, max_trajs: int):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    draw_layout(env, ax)

    for idx, traj in enumerate(trajectories[:max_trajs]):
        pts = np.asarray([(p["x"], p["y"]) for p in traj])
        if len(pts) == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0, alpha=0.55, zorder=4)
        if idx == 0:
            ax.scatter(pts[0, 0], pts[0, 1], color="#1d4ed8", s=22, label="start", zorder=5)

    ax.scatter([3.8], [-3.6], color="#16a34a", label="goal", zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_episode(env: MuSHRNavEnv, policy_name: str, rng: np.random.Generator, seed: int, max_steps: int):
    obs, info = env.reset(seed=seed)
    traj = [info]
    total_reward = 0.0
    min_obstacle_distance = float("inf")

    for _ in range(max_steps):
        if policy_name == "random":
            action = random_policy(rng, obs)
        elif policy_name == "heading":
            action = heading_controller(obs)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        min_obstacle_distance = min(min_obstacle_distance, info["min_obstacle_distance"])
        traj.append(info)

        if terminated or truncated:
            break

    result = {
        "episode": seed,
        "steps": info["step"],
        "return": total_reward,
        "success": int(info["success"]),
        "collision": int(info["collision"]),
        "timeout": int(not info["success"] and not info["collision"]),
        "final_distance": info["distance_to_goal"],
        "min_obstacle_distance": min_obstacle_distance,
        "final_x": info["x"],
        "final_y": info["y"],
    }
    return result, traj


def summarize(results):
    returns = np.asarray([r["return"] for r in results], dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    final_dist = np.asarray([r["final_distance"] for r in results], dtype=np.float64)
    min_obs = np.asarray([r["min_obstacle_distance"] for r in results], dtype=np.float64)
    success = np.asarray([r["success"] for r in results], dtype=np.float64)
    collision = np.asarray([r["collision"] for r in results], dtype=np.float64)
    timeout = np.asarray([r["timeout"] for r in results], dtype=np.float64)
    return {
        "episodes": len(results),
        "success_rate": success.mean(),
        "collision_rate": collision.mean(),
        "timeout_rate": timeout.mean(),
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "mean_steps": steps.mean(),
        "mean_final_distance": final_dist.mean(),
        "mean_min_obstacle_distance": min_obs.mean(),
        "min_min_obstacle_distance": min_obs.min(),
    }


def write_csv(path: Path, results):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--policy", choices=["random", "heading"], default="random")
    parser.add_argument("--dynamic-mode", choices=["none", "easy"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plot-trajectories", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    xml_path = DYNAMIC_EASY_XML if args.dynamic_mode == "easy" else DEFAULT_XML
    cfg = MuSHRNavConfig(
        max_episode_steps=args.max_steps,
        xml_path=xml_path,
        dynamic_mode=args.dynamic_mode,
    )
    env = MuSHRNavEnv(cfg)

    results = []
    trajectories = []
    for ep in range(args.episodes):
        result, traj = run_episode(env, args.policy, rng, seed=args.seed + ep, max_steps=args.max_steps)
        results.append(result)
        trajectories.append(traj)

    summary = summarize(results)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    csv_path = args.out / f"{args.policy}_episodes.csv"
    write_csv(csv_path, results)
    plot_trajectories(env, trajectories, args.out / f"{args.policy}_trajectories.png", args.plot_trajectories)
    print(f"wrote per-episode CSV to: {csv_path}")
    print(f"wrote trajectory plot to: {args.out / f'{args.policy}_trajectories.png'}")


if __name__ == "__main__":
    main()
