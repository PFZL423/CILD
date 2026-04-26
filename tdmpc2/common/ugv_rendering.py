"""
Unified rendering utilities for UGV environments.
Supports both 2D kinematic and future MuJoCo physics versions.
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image


class UGVRenderer:
    """Renderer for UGV environments using matplotlib."""

    def __init__(self, map_size=8.0, robot_radius=0.25, goal_radius=0.35, dpi=50):
        self.map_size = map_size
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.dpi = dpi

    def render(
        self,
        robot_state,
        goal,
        trajectory=None,
        obstacles=None,
        map_size=None,
        robot_radius=None,
        goal_radius=None,
        step_count=0,
    ):
        """
        Render UGV environment to RGB array.

        Args:
            robot_state: [x, y, theta, v, omega]
            goal: [gx, gy]
            trajectory: List of [x, y] positions (optional)
            obstacles: List of [x, y, vx, vy, radius] (optional)
            map_size: Override map size
            robot_radius: Override robot radius
            goal_radius: Override goal radius
            step_count: Current step number

        Returns:
            RGB array (256, 256, 3) uint8
        """
        map_size = map_size or self.map_size
        robot_radius = robot_radius or self.robot_radius
        goal_radius = goal_radius or self.goal_radius

        fig, ax = plt.subplots(figsize=(5, 5))
        half_size = map_size / 2

        # Map boundaries
        boundary = Rectangle(
            (-half_size, -half_size),
            map_size,
            map_size,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(boundary)

        # Trajectory (if provided)
        if trajectory is not None and len(trajectory) > 0:
            traj_array = np.array(trajectory)
            ax.plot(traj_array[:, 0], traj_array[:, 1],
                   'b-', linewidth=1, alpha=0.3, label='Trajectory')

        # Obstacles (if provided)
        if obstacles is not None:
            for obs in obstacles:
                x, y = obs[:2]
                radius = obs[4] if len(obs) > 4 else 0.3

                # Draw obstacle
                obs_circle = Circle(
                    (x, y),
                    radius=radius,
                    color='red',
                    alpha=0.4
                )
                ax.add_patch(obs_circle)

                # Draw velocity arrow if dynamic
                if len(obs) > 3:
                    vx, vy = obs[2:4]
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.01:
                        ax.arrow(x, y, vx*0.5, vy*0.5,
                               head_width=0.15, head_length=0.1,
                               fc='darkred', ec='darkred', linewidth=1.5)

        # Goal
        goal_circle = Circle(
            goal[:2],
            radius=goal_radius,
            color='green',
            alpha=0.3
        )
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=15)

        # Robot
        x, y, theta = robot_state[:3]
        robot_circle = Circle(
            (x, y),
            radius=robot_radius,
            color='blue',
            alpha=0.6
        )
        ax.add_patch(robot_circle)

        # Robot orientation arrow
        arrow_len = 0.4
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15,
                fc='darkblue', ec='darkblue', linewidth=2)

        ax.set_xlim(-half_size - 0.5, half_size + 0.5)
        ax.set_ylim(-half_size - 0.5, half_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'UGV Step {step_count}')

        # Convert to RGB array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)[:, :, :3]  # Remove alpha channel
        plt.close(fig)

        # Resize to 256x256
        img = Image.fromarray(img_array)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)
