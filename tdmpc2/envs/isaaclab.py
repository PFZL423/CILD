"""
IsaacLab vectorized environment adapter for TD-MPC2.

IsaacLab DirectRLEnv instances are already vectorized GPU environments: reset()
returns a policy observation tensor for all sub-environments and step() returns
batched reward, terminated, and truncated tensors. TD-MPC2 expects a smaller vec
env surface that returns batched torch tensors directly, with single-env Gym
spaces for observation/action metadata. This module bridges those two contracts
without wrapping through TensorWrapper or Gymnasium vector wrappers.
"""

import gymnasium as gym
import numpy as np
import torch


class IsaacVecEnv:
	"""
	Thin adapter around an IsaacLab DirectRLEnv.

	The underlying environment owns vectorization, reset-on-done behavior, and GPU
	tensor buffers. This adapter only normalizes the observation key, done signal,
	info dictionary, and single-env space metadata expected by TD-MPC2.
	"""

	def __init__(self, env, device):
		self.env = env
		self.device = torch.device(device)
		self.num_envs = env.num_envs
		self.observation_space = self._box_space(env.single_observation_space['policy'])
		self.action_space = self._box_space(env.single_action_space)
		self.max_episode_steps = int(env.max_episode_length)

	def _box_space(self, space):
		if not isinstance(space, gym.spaces.Box):
			raise ValueError(f'IsaacVecEnv only supports Box spaces, got {type(space).__name__}.')
		return gym.spaces.Box(
			low=np.asarray(space.low, dtype=np.float32),
			high=np.asarray(space.high, dtype=np.float32),
			shape=space.shape,
			dtype=np.float32,
		)

	def _policy_obs(self, obs):
		if not isinstance(obs, dict) or 'policy' not in obs:
			raise ValueError('IsaacLab observation must be a dict containing the "policy" key.')
		return obs['policy'].to(device=self.device, dtype=torch.float32)

	def reset(self):
		obs, _ = self.env.reset()
		return self._policy_obs(obs)

	def step(self, action):
		action = action.to(device=self.device, dtype=torch.float32)
		obs, reward, terminated, truncated, info = self.env.step(action)
		terminated = terminated.to(device=self.device, dtype=torch.bool)
		truncated = truncated.to(device=self.device, dtype=torch.bool)
		done = terminated | truncated
		info = dict(info)
		info['terminated'] = terminated
		info['truncated'] = truncated
		if 'success' in info:
			info['success'] = torch.as_tensor(info['success'], device=self.device, dtype=torch.float32)
			if info['success'].ndim == 0:
				info['success'] = info['success'].repeat(self.num_envs)
		else:
			info['success'] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
		return (
			self._policy_obs(obs),
			reward.to(device=self.device, dtype=torch.float32),
			done,
			info,
		)

	def rand_act(self):
		return 2 * torch.rand((self.num_envs, *self.action_space.shape), device=self.device, dtype=torch.float32) - 1

	def close(self):
		self.env.close()


def _get_cfg(cfg, key, default):
	try:
		return cfg.get(key, default)
	except AttributeError:
		return getattr(cfg, key, default)


def make_env(cfg):
	"""
	Make an IsaacLab DirectRLEnv and adapt it to TD-MPC2's batched torch vec env API.

	Expected cfg fields:
		cfg.task: Gymnasium id, e.g. "Isaac-Cartpole-Direct-v0".
		cfg.num_envs: Number of IsaacLab sub-environments, default 512.
		cfg.device: Simulation/tensor device, default "cuda:0".
	"""
	if not cfg.task.startswith('Isaac-'):
		raise ValueError(f'Unknown task {cfg.task}')

	# Register the direct Cartpole task. This import must happen after Isaac Sim's
	# AppLauncher has started, which is guaranteed by train_isaac.py.
	import isaaclab_tasks.direct.cartpole  # noqa: F401
	from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

	num_envs = int(_get_cfg(cfg, 'num_envs', 512))
	device = _get_cfg(cfg, 'device', 'cuda:0')
	env_cfg = parse_env_cfg(cfg.task, device=device, num_envs=num_envs)
	env = gym.make(cfg.task, cfg=env_cfg).unwrapped
	return IsaacVecEnv(env, device)
