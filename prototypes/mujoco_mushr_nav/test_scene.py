import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "assets" / "mushr_nav_static.xml"
DEFAULT_OUT = ROOT / "outputs" / "scene_check"


def save_rgb(renderer, model, data, camera_name, path):
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    imageio.imwrite(path, rgb)


def save_depth(renderer, model, data, camera_name, path):
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [2, 98])
        depth_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        depth_norm = np.zeros_like(depth)
    imageio.imwrite(path, (255 * depth_norm).astype(np.uint8))
    return depth


def print_model_summary(model):
    print(f"model: nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}, ncam={model.ncam}, nsite={model.nsite}")
    print("actuators:")
    for i in range(model.nu):
        print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
    print("cameras:")
    for i in range(model.ncam):
        print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}")
    print("sites:")
    for i in range(model.nsite):
        print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")


def draw_topdown(model, data, path, trail=None):
    mujoco.mj_forward(model, data)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.3)

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = model.geom_type[geom_id]
        pos = data.geom_xpos[geom_id]
        size = model.geom_size[geom_id]

        if name == "floor":
            ax.add_patch(Rectangle((-5, -5), 10, 10, facecolor="#ececec", edgecolor="none", zorder=0))
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

    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    if goal_id >= 0:
        goal = data.site_xpos[goal_id]
        ax.add_patch(Circle((goal[0], goal[1]), 0.18, facecolor="#21a652", edgecolor="#0d6b31", zorder=4))
        ax.text(goal[0] + 0.15, goal[1] + 0.15, "goal", fontsize=8)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "buddy")
    if trail:
        pts = np.asarray(trail)
        ax.plot(pts[:, 0], pts[:, 1], color="#2563eb", linewidth=1.5, alpha=0.85, zorder=4)
    if body_id >= 0:
        car = data.xpos[body_id]
        ax.add_patch(Circle((car[0], car[1]), 0.18, facecolor="#2563eb", edgecolor="#0f2f78", zorder=5))
        ax.text(car[0] + 0.15, car[1] + 0.15, "buddy", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def drive_action(step):
    if step < 120:
        return np.array([0.0, 8.0])
    if step < 240:
        return np.array([0.25, 8.0])
    return np.array([-0.25, 8.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mujoco-render", action="store_true", help="Try MuJoCo RGB/depth rendering if a GL backend is available.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    print_model_summary(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "buddy")
    if body_id < 0:
        raise RuntimeError("Could not find body 'buddy'")

    draw_topdown(model, data, args.out / "topdown_initial.png")

    renderer = None
    if args.mujoco_render:
        try:
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            save_rgb(renderer, model, data, "overview", args.out / "overview_initial.png")
            save_rgb(renderer, model, data, "buddy_third_person", args.out / "third_person_initial.png")
            depth = save_depth(renderer, model, data, "buddy_realsense_d435i", args.out / "realsense_depth_initial.png")
            print(f"initial depth: min={np.nanmin(depth):.3f}, max={np.nanmax(depth):.3f}, mean={np.nanmean(depth):.3f}")
        except Exception as exc:
            print(f"MuJoCo renderer unavailable: {type(exc).__name__}: {exc}")

    capture_steps = {0, 120, 240, args.steps - 1}
    trail = []
    for step in range(args.steps):
        data.ctrl[:] = drive_action(step)
        mujoco.mj_step(model, data)
        trail.append(data.xpos[body_id, :2].copy())
        if step in capture_steps:
            draw_topdown(model, data, args.out / f"topdown_step_{step:04d}.png", trail=trail)
            if renderer is not None:
                save_rgb(renderer, model, data, "overview", args.out / f"overview_step_{step:04d}.png")

    pos = data.xpos[body_id].copy()
    print(f"final buddy position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    draw_topdown(model, data, args.out / "topdown_final.png", trail=trail)
    if renderer is not None:
        save_rgb(renderer, model, data, "overview", args.out / "overview_final.png")
        save_rgb(renderer, model, data, "buddy_third_person", args.out / "third_person_final.png")
        depth = save_depth(renderer, model, data, "buddy_realsense_d435i", args.out / "realsense_depth_final.png")
        print(f"final depth: min={np.nanmin(depth):.3f}, max={np.nanmax(depth):.3f}, mean={np.nanmean(depth):.3f}")
    print(f"wrote scene check images to: {args.out}")


if __name__ == "__main__":
    main()
