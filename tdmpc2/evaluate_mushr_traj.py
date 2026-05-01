import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import hydra
import matplotlib
import mujoco
import numpy as np
import torch
from termcolor import colored

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from prototypes.mujoco_mushr_nav.env import DYNAMIC_HARD_XML, MuSHRNavConfig, MuSHRNavEnv
from prototypes.mujoco_mushr_nav.eval_splits import make_seed_pairs, split_names


EVAL_MODES = (
	('static', 'none', 'mushr-nav-static-hard'),
	('frozen', 'frozen', 'mushr-nav-dynamic-frozen'),
	('moving', 'hard', 'mushr-nav-dynamic-hard'),
)


def _to_float(value):
	if isinstance(value, torch.Tensor):
		value = value.detach().cpu().item()
	return float(value)


def _to_bool(value):
	if isinstance(value, torch.Tensor):
		value = value.detach().cpu().item()
	return bool(value)


def write_csv(path: Path, rows):
	fieldnames = []
	for row in rows:
		for key in row:
			if key not in fieldnames:
				fieldnames.append(key)
	with path.open('w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def draw_layout(env: MuSHRNavEnv, ax):
	for geom_id in range(env.model.ngeom):
		name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ''
		geom_type = env.model.geom_type[geom_id]
		pos = env.data.geom_xpos[geom_id]
		size = env.model.geom_size[geom_id]
		if name == 'floor':
			ax.add_patch(Rectangle((-5, -5), 10, 10, facecolor='#f3f4f6', edgecolor='none', zorder=0))
		elif 'wall' in name:
			ax.add_patch(
				Rectangle(
					(pos[0] - size[0], pos[1] - size[1]),
					2 * size[0],
					2 * size[1],
					facecolor='#4b5563',
					edgecolor='#1f2937',
					linewidth=0.7,
					zorder=2,
				)
			)
		elif 'obs' in name:
			if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
				ax.add_patch(Circle((pos[0], pos[1]), size[0], facecolor='#dc2626', edgecolor='#7f1d1d', zorder=3))
			else:
				ax.add_patch(
					Rectangle(
						(pos[0] - size[0], pos[1] - size[1]),
						2 * size[0],
						2 * size[1],
						facecolor='#dc2626',
						edgecolor='#7f1d1d',
						linewidth=0.7,
						zorder=3,
					)
				)


def plot_trajectories(env_by_mode, traj_rows, out_path: Path, max_pairs: int):
	fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160, sharex=True, sharey=True)
	rows_by_mode_pair = defaultdict(list)
	for row in traj_rows:
		pair_idx = int(row['pair_idx'])
		if pair_idx < max_pairs:
			rows_by_mode_pair[(row['mode'], pair_idx)].append(row)

	for ax, (mode, _dynamic_mode, _task) in zip(axes, EVAL_MODES):
		ax.set_title(mode)
		ax.set_aspect('equal')
		ax.set_xlim(-5.5, 5.5)
		ax.set_ylim(-5.5, 5.5)
		ax.grid(True, linewidth=0.35, alpha=0.35)
		draw_layout(env_by_mode[mode], ax)
		for pair_idx in range(max_pairs):
			rows = rows_by_mode_pair.get((mode, pair_idx), [])
			if not rows:
				continue
			pts = np.asarray([(float(r['x']), float(r['y'])) for r in rows], dtype=np.float64)
			ax.plot(pts[:, 0], pts[:, 1], linewidth=1.1, alpha=0.65, zorder=4)
			ax.scatter(pts[0, 0], pts[0, 1], color='#2563eb', s=14, zorder=5)
			ax.scatter(float(rows[0]['goal_x']), float(rows[0]['goal_y']), color='#16a34a', s=18, zorder=5)
		ax.set_xlabel('x [m]')
	axes[0].set_ylabel('y [m]')
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def make_proto_env(dynamic_mode: str, max_steps: int):
	return MuSHRNavEnv(
		MuSHRNavConfig(
			xml_path=DYNAMIC_HARD_XML,
			max_episode_steps=max_steps,
			dynamic_mode=dynamic_mode,
			procedural_layout=True,
		)
	)


def rollout(agent, env, *, cfg, pair_idx, split, mode, task, layout_seed, dynamic_seed, max_steps):
	options = {'layout_seed': int(layout_seed)}
	if mode != 'static':
		options['dynamic_seed'] = int(dynamic_seed)
	obs, info = env.reset(options=options)
	traj_rows = [
		trajectory_row(pair_idx, split, mode, task, layout_seed, dynamic_seed, info, reward=0.0, action=None)
	]
	total_reward = 0.0
	terminated = False
	truncated = False

	for step in range(max_steps):
		obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
		action = agent.act(obs_tensor, t0=(step == 0), eval_mode=True)
		obs, reward, terminated, truncated, info = env.step(action.numpy())
		total_reward += float(reward)
		traj_rows.append(
			trajectory_row(pair_idx, split, mode, task, layout_seed, dynamic_seed, info, reward, action)
		)
		if terminated or truncated:
			break

	timeout = bool(info['timeout'] or (not info['success'] and not info['collision']))
	result = {
		'pair_idx': pair_idx,
		'split': split,
		'mode': mode,
		'task': task,
		'layout_seed': int(layout_seed),
		'dynamic_seed': int(dynamic_seed),
		'realized_dynamic_seed': int(info['dynamic_seed']),
		'template': info['layout_template'],
		'success': int(info['success']),
		'collision': int(info['collision']),
		'collision_type': info['collision_type'],
		'static_collision': int(info['static_collision']),
		'dynamic_collision': int(info['dynamic_collision']),
		'timeout': int(timeout),
		'spl': float(info['spl']),
		'path_length': float(info['path_length']),
		'geodesic_distance': float(info['episode_geodesic_distance']),
		'near_miss': int(info['near_miss']),
		'dynamic_near_miss': int(info['dynamic_near_miss']),
		'min_obstacle_distance': float(info['episode_min_obstacle_distance']),
		'dynamic_min_distance': float(info['dynamic_min_distance']),
		'min_ttc': float(info['min_ttc']),
		'ttc_violation': int(info['ttc_violation']),
		'return': float(total_reward),
		'episode_steps': int(info['step']),
		'final_distance': float(info['distance_to_goal']),
		'terminated': int(terminated),
		'truncated': int(truncated),
	}
	return result, traj_rows


def trajectory_row(pair_idx, split, mode, task, layout_seed, dynamic_seed, info, reward, action):
	row = {
		'pair_idx': pair_idx,
		'split': split,
		'mode': mode,
		'task': task,
		'step': info['step'],
		'time': info['time'],
		'layout_seed': int(layout_seed),
		'dynamic_seed': int(dynamic_seed),
		'realized_dynamic_seed': int(info['dynamic_seed']),
		'template': info['layout_template'],
		'x': info['x'],
		'y': info['y'],
		'yaw': info['yaw'],
		'goal_x': info['goal_x'],
		'goal_y': info['goal_y'],
		'distance_to_goal': info['distance_to_goal'],
		'progress': info['progress'],
		'reward': float(reward),
		'action_steer': np.nan if action is None else _to_float(action[0]),
		'action_throttle': np.nan if action is None else _to_float(action[1]),
		'success': int(info['success']),
		'collision': int(info['collision']),
		'collision_type': info['collision_type'],
		'static_collision': int(info['static_collision']),
		'dynamic_collision': int(info['dynamic_collision']),
		'timeout': int(info['timeout']),
		'path_length': info['path_length'],
		'spl': info['spl'],
		'near_miss': int(info['near_miss']),
		'dynamic_near_miss': int(info['dynamic_near_miss']),
		'min_ttc': info['min_ttc'],
		'ttc_violation': int(info['ttc_violation']),
	}
	for key, value in info.items():
		if key.startswith('dyn_obs_'):
			row[key] = value
	return row


def _mean(values):
	values = np.asarray(values, dtype=np.float64)
	return float(values.mean()) if len(values) else float('nan')


def _mean_finite(values):
	values = np.asarray(values, dtype=np.float64)
	finite = values[np.isfinite(values)]
	return float(finite.mean()) if len(finite) else float('inf')


def summarize(results):
	rows = []
	grouped = defaultdict(list)
	for result in results:
		grouped[(result['split'], result['mode'])].append(result)
	for (split, mode), items in sorted(grouped.items()):
		rows.append(
			{
				'split': split,
				'mode': mode,
				'episodes': len(items),
				'success_rate': _mean([r['success'] for r in items]),
				'collision_rate': _mean([r['collision'] for r in items]),
				'static_collision_rate': _mean([r['static_collision'] for r in items]),
				'dynamic_collision_rate': _mean([r['dynamic_collision'] for r in items]),
				'timeout_rate': _mean([r['timeout'] for r in items]),
				'mean_spl': _mean([r['spl'] for r in items]),
				'mean_return': _mean([r['return'] for r in items]),
				'mean_episode_steps': _mean([r['episode_steps'] for r in items]),
				'near_miss_rate': _mean([r['near_miss'] for r in items]),
				'dynamic_near_miss_rate': _mean([r['dynamic_near_miss'] for r in items]),
				'ttc_violation_rate': _mean([r['ttc_violation'] for r in items]),
				'mean_min_ttc': _mean_finite([r['min_ttc'] for r in items]),
				'mean_final_distance': _mean([r['final_distance'] for r in items]),
			}
		)
	return rows


@hydra.main(config_name='config', config_path='.')
def main(cfg):
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)

	out_dir = Path(cfg.get('out_dir', cfg.work_dir / 'mushr_trajectories'))
	out_dir.mkdir(parents=True, exist_ok=True)
	episodes = int(cfg.get('traj_episodes', cfg.eval_episodes))
	max_steps = int(cfg.get('traj_max_steps', cfg.episode_length))
	plot_episodes = int(cfg.get('plot_episodes', min(episodes, 6)))
	split = cfg.get('eval_split', 'seen')
	if split not in split_names():
		raise ValueError(f'Unknown eval_split={split!r}; choose from {split_names()}')
	seed_pairs = make_seed_pairs(split, episodes=episodes)

	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Split: {split} episodes={len(seed_pairs)} max_steps={max_steps}', 'blue', attrs=['bold']))
	print(colored(f'Output: {out_dir}', 'blue', attrs=['bold']))

	make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	agent.model.eval()

	env_by_mode = {
		mode: make_proto_env(dynamic_mode, max_steps)
		for mode, dynamic_mode, _task in EVAL_MODES
	}

	results = []
	traj_rows = []
	for pair_idx, pair in enumerate(seed_pairs):
		for mode, _dynamic_mode, task in EVAL_MODES:
			result, traj = rollout(
				agent,
				env_by_mode[mode],
				cfg=cfg,
				pair_idx=pair_idx,
				split=pair.split,
				mode=mode,
				task=task,
				layout_seed=pair.layout_seed,
				dynamic_seed=pair.dynamic_seed,
				max_steps=max_steps,
			)
			results.append(result)
			traj_rows.extend(traj)

	write_csv(out_dir / 'learned_episodes.csv', results)
	write_csv(out_dir / 'learned_trajectories.csv', traj_rows)
	write_csv(out_dir / 'summary.csv', summarize(results))
	plot_trajectories(env_by_mode, traj_rows, out_dir / 'learned_trajectories.png', plot_episodes)
	print(colored(f'Wrote {out_dir / "learned_episodes.csv"}', 'yellow'))
	print(colored(f'Wrote {out_dir / "learned_trajectories.csv"}', 'yellow'))
	print(colored(f'Wrote {out_dir / "summary.csv"}', 'yellow'))
	print(colored(f'Wrote {out_dir / "learned_trajectories.png"}', 'yellow'))


if __name__ == '__main__':
	main()
