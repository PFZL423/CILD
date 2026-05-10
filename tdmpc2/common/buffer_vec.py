import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class VecBuffer():
	"""
	Vec-env replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available and large enough, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self._capacity = min(cfg.buffer_size, cfg.steps*cfg.num_envs)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		self._episode_ids = torch.arange(cfg.num_envs, dtype=torch.int64)
		if self._capacity < cfg.num_envs:
			raise ValueError(f'Buffer capacity ({self._capacity}) must be at least num_envs ({cfg.num_envs}).')
		base_capacity = self._capacity // cfg.num_envs
		remainder = self._capacity % cfg.num_envs
		self._env_capacities = torch.full((cfg.num_envs,), base_capacity, dtype=torch.int64)
		self._env_capacities[:remainder] += 1
		self._env_offsets = torch.zeros(cfg.num_envs, dtype=torch.int64)
		self._env_offsets[1:] = torch.cumsum(self._env_capacities[:-1], dim=0)
		self._env_positions = torch.zeros(cfg.num_envs, dtype=torch.int64)

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of completed episodes across all envs."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first env-step to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		if torch.cuda.is_available():
			mem_free, _ = torch.cuda.mem_get_info()
			storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		else:
			storage_device = 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		buffer = self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)
		self._init_storage(buffer, tds)
		return buffer

	def _init_storage(self, buffer, tds):
		"""
		Materialize storage and mark unwritten slots as length-1 dummy trajectories.
		"""
		dummy = tds[0].clone().expand(self._capacity).clone()
		dummy = dummy.zero_()
		dummy['episode'] = -torch.arange(1, self._capacity+1, dtype=torch.int64, device=dummy.device)
		index = torch.arange(self._capacity)
		buffer.storage.set(index, dummy)
		buffer.sampler.extend(index)

	def add_step(self, obs, action, reward, terminated, truncated):
		"""
		Add one vectorized env-step to the buffer.

		Args:
			obs (torch.Tensor): Observation before the step, shape (num_envs, *obs_shape).
			action (torch.Tensor): Action applied at the step, shape (num_envs, *action_shape).
			reward (torch.Tensor): Step reward, shape (num_envs,).
			terminated (torch.Tensor): Step termination signal, shape (num_envs,).
			truncated (torch.Tensor): Step truncation signal, shape (num_envs,).
		"""
		num_envs = self.cfg.num_envs
		assert obs.shape[0] == num_envs
		assert action.shape[0] == num_envs
		assert reward.shape[0] == num_envs
		assert terminated.shape[0] == num_envs
		assert truncated.shape[0] == num_envs

		episode = self._episode_ids.to(device=obs.device)
		td = TensorDict({
			'obs': obs,
			'action': action,
			'reward': reward,
			'terminated': terminated.to(dtype=torch.float32),
			'episode': episode,
		}, batch_size=(num_envs,))

		if not hasattr(self, '_buffer'):
			self._buffer = self._init(td)
		index = self._env_offsets + self._env_positions
		self._buffer.storage.set(index, td, set_cursor=False)
		self._buffer.mark_update(index)
		self._env_positions = (self._env_positions+1) % self._env_capacities

		done = (terminated | truncated).to(device=self._episode_ids.device, dtype=torch.bool)
		self._episode_ids[done] += num_envs
		self._num_eps += int(done.sum().item())
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		# NOTE: vec-buffer convention differs from single-env Buffer. We store
		# (obs_t, action_t, reward_t) at row t — i.e. action_t and reward_t are
		# the transition that LEAVES obs_t. Original Buffer stores
		# (obs_t, action_{t-1}, reward_{t-1}) (the transition that ARRIVED at
		# obs_t). So we drop the LAST action/reward of an (H+1)-row slice instead
		# of the FIRST. This keeps obs[0..H] paired with action[0..H-1] for the
		# H consecutive transitions in TDMPC2._update's latent rollout.
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[:-1].contiguous()
		reward = td.get('reward')[:-1].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[:-1].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)
