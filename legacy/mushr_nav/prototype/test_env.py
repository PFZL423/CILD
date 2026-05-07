import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import mujoco
import numpy as np

from env import DEFAULT_XML, DYNAMIC_HARD_XML, MuSHRNavConfig, MuSHRNavEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "env_check"


def heading_controller(obs: np.ndarray) -> np.ndarray:
    """只用于检查环境闭环的简单非学习控制器。"""

    raw_obs_dim = 8 + 32
    latest_obs = obs[-raw_obs_dim:]
    goal_angle = float(latest_obs[7])
    steering = np.clip(1.8 * goal_angle, -1.0, 1.0)
    throttle = 0.55 if abs(goal_angle) < 1.2 else 0.25
    return np.array([steering, throttle], dtype=np.float32)


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


def plot_episode(env: MuSHRNavEnv, traj, out_path: Path):
    pts = np.asarray([(p["x"], p["y"]) for p in traj])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    draw_layout(env, ax)
    if len(pts) > 0:
        ax.plot(pts[:, 0], pts[:, 1], color="#2563eb", linewidth=1.5, zorder=4)
        ax.scatter(pts[0, 0], pts[0, 1], color="#1d4ed8", label="start", zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#0f172a", label="end", zorder=5)
    if traj:
        ax.scatter([traj[0]["goal_x"]], [traj[0]["goal_y"]], color="#16a34a", label="goal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dynamic-mode", choices=["none", "hard", "frozen"], default="none")
    parser.add_argument("--dynamic-seed", type=int, default=None)
    parser.add_argument("--fixed-layout", action="store_true")
    parser.add_argument("--layout-template", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    xml_path = DYNAMIC_HARD_XML if args.dynamic_mode in ("hard", "frozen") else DEFAULT_XML
    cfg = MuSHRNavConfig(
        max_episode_steps=args.max_steps,
        xml_path=xml_path,
        dynamic_mode=args.dynamic_mode,
        dynamic_seed=args.dynamic_seed,
        procedural_layout=not args.fixed_layout,
        layout_template=args.layout_template,
    )
    env = MuSHRNavEnv(cfg)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        traj = [info]
        total_reward = 0.0

        for _ in range(args.max_steps):
            action = heading_controller(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            traj.append(info)
            if terminated or truncated:
                break

        print(
            f"episode={ep} steps={info['step']} return={total_reward:.3f} "
            f"success={info['success']} collision={info['collision']} "
            f"distance={info['distance_to_goal']:.3f} path={info['path_length']:.3f} "
            f"spl={info['spl']:.3f} min_obs_dist={info['episode_min_obstacle_distance']:.3f} "
            f"min_ttc={info['min_ttc']:.3f}"
        )
        plot_episode(env, traj, args.out / f"episode_{ep:02d}.png")

    print(f"wrote env check plots to: {args.out}")


if __name__ == "__main__":
    main()
