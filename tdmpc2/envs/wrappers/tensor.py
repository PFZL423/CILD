from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		result = self.env.reset()
		# Handle both old API (obs only) and new API (obs, info)
		if isinstance(result, tuple):
			obs, info = result
		else:
			obs = result
		return self._obs_to_tensor(obs)

	def step(self, action):
		result = self.env.step(action.numpy())
		# Handle both old API (4 values) and new API (5 values)
		if len(result) == 5:
			obs, reward, terminated, truncated, info = result
			done = terminated or truncated
		else:
			obs, reward, done, info = result
			terminated = done

		info = defaultdict(float, info)
		info['success'] = float(info.get('success', 0.0))
		info['terminated'] = torch.tensor(float(terminated))
		# Add collision_total if not present
		if 'collision_total' not in info:
			info['collision_total'] = float(info.get('collision', 0.0))

		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
