"""Live training plot for Safety Gymnasium TD-MPC2 baselines.

Reads `tdmpc2/logs/<task>/<seed>/<exp_name>/{train,eval}.csv` and plots
the four core metrics (reward / success / cost / goal_reached_count)
side by side. Refreshes periodically so you can watch a long run.

Usage (from repo root):
    python plot_results.py
    python plot_results.py --tasks SafetyPointGoal1-v0 SafetyPointGoal2-v0 SafetyCarGoal1-v0 \
                          --exp baseline_seed1
    python plot_results.py --no-loop                # one-shot static plot
    python plot_results.py --out /tmp/baseline.png  # save to file instead of showing

The script tolerates missing tasks (e.g. only one of three is currently
running) and tolerates an in-progress CSV being written to.
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / 'tdmpc2' / 'logs'

DEFAULT_TASKS = ['SafetyPointGoal1-v0', 'SafetyPointGoal2-v0', 'SafetyCarGoal1-v0']
TASK_COLORS = {
	'SafetyPointGoal1-v0': 'steelblue',
	'SafetyPointGoal2-v0': 'tomato',
	'SafetyCarGoal1-v0':   'seagreen',
}

# (csv_column, panel title, panel y-label)
PANELS = [
	('episode_reward',             'Episode Reward (R)',          'reward'),
	('episode_cost',               'Episode Cost (C)',            'cost'),
	('episode_goal_reached_count', 'Goals Reached / Episode (G)', 'count'),
	('episode_success',            'Per-Step Success Rate (S)',   'fraction'),
]


def load_csv(path):
	"""Robustly load a TD-MPC2 csv, handling in-progress writes and tensor strings."""
	if not path.exists():
		return None
	try:
		df = pd.read_csv(path)
	except Exception:
		return None
	# train.csv stores reward as 'tensor(-0.7580)'; extract the number.
	if 'episode_reward' in df.columns:
		df['episode_reward'] = pd.to_numeric(
			df['episode_reward'].astype(str).str.extract(r'([-\d.]+)')[0],
			errors='coerce',
		)
	for col in ('episode_cost', 'episode_goal_reached_count', 'episode_success'):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')
	return df


def plot_panel(ax, key, title, ylabel, runs, smooth_window=10):
	for label, color, df in runs:
		if df is None or key not in df.columns or df.empty:
			continue
		s = df[key]
		smoothed = s.rolling(smooth_window, min_periods=1).mean()
		ax.plot(df['step'], smoothed, label=label, color=color, linewidth=1.8)
		# Show the raw signal lightly in the background to reveal noise level.
		ax.plot(df['step'], s, color=color, alpha=0.15, linewidth=0.8)
	ax.set_title(title)
	ax.set_xlabel('step')
	ax.set_ylabel(ylabel)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', fontsize=8)


def render(fig, axes, tasks, exp, source, smooth_window):
	for ax in axes.flatten():
		ax.cla()

	runs = []
	for task in tasks:
		csv_path = LOG_ROOT / task / '1' / exp / f'{source}.csv'
		df = load_csv(csv_path)
		# Show shorter task name in legend to keep it readable.
		short = task.replace('-v0', '')
		runs.append((short, TASK_COLORS.get(task, 'gray'), df))

	for ax, (key, title, ylabel) in zip(axes.flatten(), PANELS):
		plot_panel(ax, key, title, ylabel, runs, smooth_window=smooth_window)

	fig.suptitle(f'TD-MPC2 baseline on Safety Gymnasium  |  exp={exp}  |  source={source}.csv',
	             fontsize=11)
	fig.tight_layout(rect=(0, 0, 1, 0.97))


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS,
	                    help='Task IDs to overlay (default: 3 motivation tasks).')
	parser.add_argument('--exp', default='baseline_seed1',
	                    help='Experiment name (default: baseline_seed1).')
	parser.add_argument('--source', choices=['eval', 'train'], default='eval',
	                    help='Read eval.csv (cleaner, sparser) or train.csv (denser).')
	parser.add_argument('--smooth', type=int, default=10,
	                    help='Rolling-mean window in csv rows (default: 10).')
	parser.add_argument('--refresh', type=float, default=15.0,
	                    help='Seconds between refreshes in live mode (default: 15).')
	parser.add_argument('--no-loop', action='store_true',
	                    help='Render once and exit (good for headless / file output).')
	parser.add_argument('--out', type=str, default=None,
	                    help='If set, save figure to this path and exit (implies --no-loop).')
	args = parser.parse_args()

	if args.out:
		args.no_loop = True

	if not args.no_loop:
		plt.ion()
	fig, axes = plt.subplots(2, 2, figsize=(13, 9))

	if args.no_loop:
		render(fig, axes, args.tasks, args.exp, args.source, args.smooth)
		if args.out:
			fig.savefig(args.out, dpi=120)
			print(f'Saved {args.out}')
		else:
			plt.show()
		return

	while True:
		render(fig, axes, args.tasks, args.exp, args.source, args.smooth)
		fig.canvas.draw_idle()
		plt.pause(args.refresh)


if __name__ == '__main__':
	main()
