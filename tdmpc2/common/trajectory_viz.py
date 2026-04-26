from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle


def save_ugv_trajectory(states, goal, map_size, save_path, title="UGV Trajectory"):
	"""Save a 2D UGV trajectory plot to disk."""
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(10, 10))
	half_size = map_size / 2

	boundary = Rectangle(
		(-half_size, -half_size),
		map_size,
		map_size,
		fill=False,
		edgecolor="black",
		linewidth=2,
		label="Map boundary",
	)
	ax.add_patch(boundary)

	goal_circle = Circle(
		goal[:2],
		radius=0.35,
		color="green",
		alpha=0.3,
		label="Goal",
	)
	ax.add_patch(goal_circle)
	ax.plot(goal[0], goal[1], "g*", markersize=20)

	xs = [state[0] for state in states]
	ys = [state[1] for state in states]
	ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.6, label="Trajectory")

	if xs and ys:
		ax.plot(xs[0], ys[0], "ro", markersize=10, label="Start")
		x_end, y_end, theta_end = states[-1][:3]
		ax.plot(x_end, y_end, "bs", markersize=10, label="End")

		arrow_len = 0.5
		dx = arrow_len * np.cos(theta_end)
		dy = arrow_len * np.sin(theta_end)
		arrow = FancyArrow(
			x_end,
			y_end,
			dx,
			dy,
			width=0.1,
			head_width=0.3,
			head_length=0.2,
			fc="blue",
			ec="blue",
			alpha=0.7,
		)
		ax.add_patch(arrow)

		step_interval = max(1, len(states) // 20)
		for i in range(0, len(states), step_interval):
			x, y, theta = states[i][:3]
			dx = 0.3 * np.cos(theta)
			dy = 0.3 * np.sin(theta)
			ax.arrow(
				x,
				y,
				dx,
				dy,
				head_width=0.15,
				head_length=0.1,
				fc="gray",
				ec="gray",
				alpha=0.4,
				linewidth=0.5,
			)

	ax.set_xlim(-half_size - 1, half_size + 1)
	ax.set_ylim(-half_size - 1, half_size + 1)
	ax.set_aspect("equal")
	ax.grid(True, alpha=0.3)
	ax.legend(loc="upper right")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_title(title)

	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close(fig)
	return save_path