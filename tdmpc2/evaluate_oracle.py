import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import json
import math
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

# Oracle cost超参数：可调，但不需要精调
ORACLE_LAMBDA = 2.0   # 惩罚强度
ORACLE_SIGMA  = 0.5   # 距离衰减尺度（米）


def _get_dm_env(env):
	"""遍历wrapper链，找到有physics属性的dm_control env。"""
	e = env
	for _ in range(10):
		if hasattr(e, 'physics'):
			return e
		for attr in ('env', '_env'):
			if hasattr(e, attr):
				e = getattr(e, attr)
				break
		else:
			break
	raise RuntimeError(f'Cannot find dm_control env with physics in wrapper chain')


def get_obstacle_positions(env):
	dm_env = _get_dm_env(env)
	task = dm_env._task
	positions = []
	for i in range(task._n):
		x = float(dm_env.physics.named.model.geom_pos[f'obstacle{i}', 'x'])
		h = float(task._heights[i])
		positions.append([x, h])
	return np.array(positions, dtype=np.float32)


def get_walker_x(env):
	dm_env = _get_dm_env(env)
	return float(dm_env.physics.named.data.xpos['torso', 'x'])


def oracle_act(agent, obs, obstacle_positions, walker_x, t0=False, task=None):
	"""
	在CEM/MPPI规划时，对每条采样轨迹注入障碍物距离惩罚。
	只修改value估计，不改模型权重。
	"""
	# 把障碍物位置转成tensor
	obs_pos = torch.tensor(obstacle_positions[:, 0], dtype=torch.float32, device=agent.device)  # (n,)
	walker_x_t = torch.tensor(walker_x, dtype=torch.float32, device=agent.device)

	# 保存原始_estimate_value，用monkey-patch注入cost
	original_estimate_value = agent._estimate_value.__func__

	def patched_estimate_value(self, z, actions, task_inner):
		G = original_estimate_value(self, z, actions, task_inner)

		# 用当前obs里的walker x和障碍物x估算距离
		# 对每条轨迹用同一个当前距离作为惩罚代理（保守但简单）
		dists = torch.abs(obs_pos - walker_x_t)   # (n,)
		min_dist = dists.min()
		penalty = ORACLE_LAMBDA * torch.exp(-min_dist / ORACLE_SIGMA)

		# G shape: (num_samples, 1)
		return G - penalty

	# monkey-patch
	import types
	agent._estimate_value = types.MethodType(patched_estimate_value, agent)
	action = agent.act(obs, t0=t0, task=task)
	# 还原
	agent._estimate_value = types.MethodType(original_estimate_value, agent)
	return action


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Oracle评估：给planner注入障碍物位置的ground-truth cost，
	对比原版TD-MPC2，验证"cost信息缺失"是结构性问题。

	用法：
	  python evaluate_oracle.py task=walker-walk-dynamic-obstacle \\
	    model_size=1 checkpoint=<path> eval_episodes=30
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Oracle lambda={ORACLE_LAMBDA}, sigma={ORACLE_SIGMA}', 'cyan', attrs=['bold']))

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
	agent.load(cfg.checkpoint)

	print(colored(f'Evaluating Oracle agent on {cfg.task}:', 'yellow', attrs=['bold']))

	ep_rewards, ep_successes, ep_collisions, ep_passed = [], [], [], []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		while not done:
			obstacle_pos = get_obstacle_positions(env)
			walker_x = get_walker_x(env)
			action = oracle_act(agent, obs, obstacle_pos, walker_x, t0=(t == 0), task=None)
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			t += 1
		ep_rewards.append(float(ep_reward))
		ep_successes.append(float(info['success']))
		ep_collisions.append(float(info['collision_total']))
		ep_passed.append(float(info['obstacles_passed']))
		print(f'  ep {i+1:02d}  R={ep_reward:.1f}  S={info["success"]:.0f}'
			  f'  Coll={info["collision_total"]:.0f}  Passed={info["obstacles_passed"]:.0f}')

	mean_reward    = float(np.mean(ep_rewards))
	mean_success   = float(np.mean(ep_successes))
	mean_collision = float(np.mean(ep_collisions))
	mean_passed    = float(np.mean(ep_passed))

	print(colored(
		f'\n  {cfg.task:<22}'
		f'\tR: {mean_reward:.1f}'
		f'\tS: {mean_success:.2f}'
		f'\tCollision: {mean_collision:.1f}'
		f'\tPassed: {mean_passed:.1f}', 'yellow'))

	results = {
		'mode': 'oracle',
		'oracle_lambda': ORACLE_LAMBDA,
		'oracle_sigma': ORACLE_SIGMA,
		'task': cfg.task,
		'episodes': cfg.eval_episodes,
		'mean_reward': mean_reward,
		'mean_success': mean_success,
		'mean_collision_per_ep': mean_collision,
		'mean_obstacles_passed': mean_passed,
		'per_episode': {
			'rewards': ep_rewards,
			'successes': ep_successes,
			'collisions': ep_collisions,
			'obstacles_passed': ep_passed,
		}
	}
	results_path = os.path.join(cfg.work_dir, 'eval_oracle_results.json')
	with open(results_path, 'w') as f:
		json.dump(results, f, indent=2)
	print(colored(f'Results saved to {results_path}', 'green'))


if __name__ == '__main__':
	evaluate()
