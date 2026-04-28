import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.append(str(_REPO_ROOT))

from prototypes.mujoco_mushr_nav.env import DYNAMIC_EASY_XML, MuSHRNavConfig, MuSHRNavEnv


MUSHR_NAV_TASKS = {
	'mushr-nav-static',
	'mushr-nav-dynamic-easy',
}


class MuSHRNavWrapper(gym.Wrapper):
	"""将 prototype MuSHRNavEnv 适配到 TD-MPC2 当前使用的环境接口。"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._collision_total = 0

	def reset(self):
		self._collision_total = 0
		obs, _ = self.env.reset()
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		done = terminated or truncated
		self._collision_total += int(info.get('collision', False))

		info['terminated'] = bool(terminated)
		info['success'] = float(info.get('success', False))
		info['collision'] = float(info.get('collision', False))
		info['collision_total'] = float(self._collision_total)
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

	if cfg.task == 'mushr-nav-dynamic-easy':
		env_cfg = MuSHRNavConfig(xml_path=DYNAMIC_EASY_XML, dynamic_mode='easy')
	else:
		env_cfg = MuSHRNavConfig()
	env = MuSHRNavEnv(env_cfg)
	env = MuSHRNavWrapper(env, cfg)

	# MuSHR 导航会因为 success/collision 提前终止，TD-MPC2 需要 episodic=true。
	cfg.episodic = True
	cfg.discount_max = 0.99
	cfg.rho = 0.7
	return env
