import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.append(str(_REPO_ROOT))

from prototypes.mujoco_mushr_nav.env import DYNAMIC_HARD_XML, MuSHRNavConfig, MuSHRNavEnv


MUSHR_NAV_TASKS = {
	'mushr-nav-static-hard',
	'mushr-nav-static-fixed-geodesic',
	'mushr-nav-dynamic-hard',
	'mushr-nav-dynamic-frozen',
	'mushr-nav-dynamic-fixed-geodesic',
}


class MuSHRNavWrapper(gym.Wrapper):
	"""将 prototype MuSHRNavEnv 适配到 TD-MPC2 当前使用的环境接口。"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._collision_total = 0
		self._static_collision_total = 0
		self._dynamic_collision_total = 0

	def reset(self):
		self._collision_total = 0
		self._static_collision_total = 0
		self._dynamic_collision_total = 0
		obs, _ = self.env.reset()
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		done = terminated or truncated
		self._collision_total += int(info.get('collision', False))
		self._static_collision_total += int(info.get('static_collision', False))
		self._dynamic_collision_total += int(info.get('dynamic_collision', False))

		info['terminated'] = bool(terminated)
		info['success'] = float(info.get('success', False))
		info['collision'] = float(info.get('collision', False))
		info['static_collision'] = float(info.get('static_collision', False))
		info['dynamic_collision'] = float(info.get('dynamic_collision', False))
		info['timeout'] = float(info.get('timeout', False))
		info['collision_total'] = float(self._collision_total)
		info['static_collision_total'] = float(self._static_collision_total)
		info['dynamic_collision_total'] = float(self._dynamic_collision_total)
		for key in ['euclidean_distance_to_goal', 'geodesic_distance_to_goal', 'reward_distance_to_goal', 'progress']:
			info[key] = float(info.get(key, 0.0))
		for key in ['near_miss', 'dynamic_near_miss', 'ttc_violation']:
			info[key] = float(info.get(key, False))
		return obs, reward, done, info

	@property
	def max_episode_steps(self):
		return self.env.max_episode_steps

	def render(self, *args, **kwargs):
		# 当前 headless 环境没有可用 MuJoCo OpenGL 渲染后端。
		# 短训练请使用 save_video=false；后续可替换为 top-down RGB render。
		raise RuntimeError('MuSHRNavWrapper.render is not available; run with save_video=false.')


def make_env(cfg):
	"""创建 MuSHR 导航环境。"""
	if cfg.task not in MUSHR_NAV_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'MuSHR navigation currently supports state observations only.'

	if cfg.task == 'mushr-nav-static-fixed-geodesic':
		env_cfg = MuSHRNavConfig(
			reward_mode='geodesic',
			procedural_layout=True,
			layout_seed=0,
		)
	elif cfg.task == 'mushr-nav-dynamic-fixed-geodesic':
		env_cfg = MuSHRNavConfig(
			xml_path=DYNAMIC_HARD_XML,
			reward_mode='geodesic',
			procedural_layout=True,
			layout_seed=0,
			dynamic_mode='hard',
			dynamic_seed=0,
			dynamic_random_phase=False,
			dynamic_random_speed=False,
			dynamic_random_amplitude=False,
		)
	elif cfg.task == 'mushr-nav-dynamic-hard':
		env_cfg = MuSHRNavConfig(xml_path=DYNAMIC_HARD_XML, dynamic_mode='hard')
	elif cfg.task == 'mushr-nav-dynamic-frozen':
		env_cfg = MuSHRNavConfig(xml_path=DYNAMIC_HARD_XML, dynamic_mode='frozen')
	else:
		env_cfg = MuSHRNavConfig()
	env = MuSHRNavEnv(env_cfg)
	env = MuSHRNavWrapper(env, cfg)

	# MuSHR 导航会因为 success/collision 提前终止，TD-MPC2 需要 episodic=true。
	cfg.episodic = True

	# discount: 500步episode用0.99太高，bootstrapping噪声大；0.97更稳定
	cfg.discount_max = 0.97
	cfg.discount_min = 0.97

	# rho: 保持原版0.5，0.7会放大远端误差
	cfg.rho = 0.5

	# seed_steps 在 envs/__init__.py 里被覆盖，所以在那里处理

	# reward_coef: 原版0.1对progress≈0.1m/step的奖励信号太弱，调大让reward loss主导早期学习
	cfg.reward_coef = 1.0

	# termination_coef: 500步里只有1步是terminated=1，比例1:499，权重过高会压制其他loss
	cfg.termination_coef = 0.5

	# vmin/vmax: two_hot 在存储前会先做 symlog，所以这里要填 symlog 之后的范围。
	# 实际 reward 范围约 [-10.1, +10.2]，symlog(±10) ≈ ±2.40，symlog(0.1) ≈ 0.095。
	# 用 ±4 覆盖所有值，101个bin时 bin_size=0.08，稠密 progress 信号有足够分辨率。
	# 原来 ±10 导致 101 个 bin 里 progress 信号只占 <1 个 bin，梯度几乎为零。
	cfg.vmin = -4
	cfg.vmax = 4

	return env
