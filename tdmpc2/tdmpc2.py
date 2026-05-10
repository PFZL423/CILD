import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


# Compatibility shim: torch.compiler.cudagraph_mark_step_begin was added in
# pytorch 2.3. On older pytorch (e.g. 2.1) it doesn't exist; the call is only
# meaningful with torch.compile + cudagraphs anyway, so a no-op is fine.
if not hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
	torch.compiler.cudagraph_mark_step_begin = lambda: None


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self.cfg.planning_interval = getattr(self.cfg, 'planning_interval', 1)
		self.cfg.num_envs = getattr(self.cfg, 'num_envs', 1)
		num_envs = max(self.cfg.num_envs, 1)
		self.register_buffer('_prev_mean', torch.zeros(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self.register_buffer('_plan_step_counter', torch.zeros(num_envs, dtype=torch.long, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select actions for one or more environments.

		Args:
			obs (torch.Tensor): Observation tensor with shape [obs_dim] or [B, obs_dim].
			t0 (bool or torch.Tensor): Whether each observation starts an episode.
			eval_mode (bool): Whether to use deterministic actions.
			task (int or torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action tensor with shape [action_dim] for B=1, otherwise [B, action_dim].
		"""
		obs = obs.to(self.device, non_blocking=True)
		if obs.ndim == 1:
			obs = obs.unsqueeze(0)
		B = obs.shape[0]
		if isinstance(t0, torch.Tensor):
			t0 = t0.to(self.device, non_blocking=True).bool().flatten()
			if t0.numel() == 1:
				t0 = t0.expand(B)
		else:
			t0 = torch.full((B,), bool(t0), dtype=torch.bool, device=self.device)
		if task is not None:
			if isinstance(task, torch.Tensor):
				task = task.to(self.device, non_blocking=True).flatten()
			else:
				task = torch.tensor([task], device=self.device)
			if task.numel() == 1:
				task = task.expand(B)
		if self.cfg.mpc:
			if self._prev_mean.shape[0] < B:
				extra = B - self._prev_mean.shape[0]
				self._prev_mean = torch.cat([
					self._prev_mean,
					torch.zeros(extra, self.cfg.horizon, self.cfg.action_dim, device=self.device)
				], dim=0)
				self._plan_step_counter = torch.cat([
					self._plan_step_counter,
					torch.zeros(extra, dtype=torch.long, device=self.device)
				], dim=0)
			action = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
			return action[0].cpu() if B == 1 else action
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu() if B == 1 else action

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a batch of action sequences using the learned world model.

		Args:
			obs (torch.Tensor): Observation tensor with shape [B, obs_dim].
			t0 (torch.Tensor): Boolean tensor with shape [B].
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action tensor with shape [B, action_dim].
		"""
		B = obs.shape[0]
		prev_mean = self._prev_mean[:B]
		counter = self._plan_step_counter[:B]
		t0 = t0.to(self.device, non_blocking=True).bool().flatten()
		if t0.numel() == 1:
			t0 = t0.expand(B)
		plan_mask = t0 | (counter % self.cfg.planning_interval == 0)
		if not plan_mask.any():
			a = prev_mean[:, 0]
			if not eval_mode:
				a = a + self.cfg.min_std * torch.randn_like(a)
			self._plan_step_counter[:B] = torch.where(t0, torch.ones_like(counter), counter + 1)
			return a.clamp(-1, 1)

		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.multitask and task is not None:
			task_tiled = task.unsqueeze(1).expand(B, self.cfg.num_samples).reshape(B*self.cfg.num_samples)
			task_pi = task.unsqueeze(1).expand(B, self.cfg.num_pi_trajs).reshape(B*self.cfg.num_pi_trajs)
		else:
			task_tiled, task_pi = task, task
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, B, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(1).expand(B, self.cfg.num_pi_trajs, z.shape[-1]).reshape(B*self.cfg.num_pi_trajs, z.shape[-1])
			for t in range(self.cfg.horizon-1):
				_pi_action, _ = self.model.pi(_z, task_pi)
				pi_actions[t] = _pi_action.reshape(B, self.cfg.num_pi_trajs, self.cfg.action_dim)
				_z = self.model.next(_z, _pi_action, task_pi)
			_pi_action, _ = self.model.pi(_z, task_pi)
			pi_actions[-1] = _pi_action.reshape(B, self.cfg.num_pi_trajs, self.cfg.action_dim)

		# Initialize state and parameters
		z = z.unsqueeze(1).expand(B, self.cfg.num_samples, z.shape[-1]).reshape(B*self.cfg.num_samples, z.shape[-1])
		mean = torch.zeros(B, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((B, self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		mean[:, :-1] = torch.where(t0.view(B, 1, 1), mean[:, :-1], prev_mean[:, 1:])
		actions = torch.empty(self.cfg.horizon, B, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(B, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, :, self.cfg.num_pi_trajs:] = actions_sample.permute(1, 0, 2, 3)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task].view(1, B, 1, self.cfg.action_dim)

			# Compute elite actions
			flat_actions = actions.reshape(self.cfg.horizon, B*self.cfg.num_samples, self.cfg.action_dim)
			value = self._estimate_value(z, flat_actions, task_tiled).reshape(B, self.cfg.num_samples, 1).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(-1), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(-1))
			elite_idxs = elite_idxs.view(1, B, self.cfg.num_elites, 1).expand(self.cfg.horizon, B, self.cfg.num_elites, self.cfg.action_dim)
			elite_actions = torch.gather(actions, 2, elite_idxs)

			# Update parameters
			max_value = elite_value.max(1, keepdim=True).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(1, keepdim=True)
			elite_actions = elite_actions.permute(1, 0, 2, 3)
			mean = (score.view(B, 1, self.cfg.num_elites, 1) * elite_actions).sum(dim=2) / (score.sum(1).view(B, 1, 1) + 1e-9)
			std = ((score.view(B, 1, self.cfg.num_elites, 1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2) / (score.sum(1).view(B, 1, 1) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task].view(B, 1, self.cfg.action_dim)
				std = std * self.model._action_masks[task].view(B, 1, self.cfg.action_dim)
			elite_actions = elite_actions.permute(1, 0, 2, 3)

		# Select action
		probs = score.squeeze(-1).clamp_min(1e-9)
		gumbel = -torch.log(-torch.log(torch.rand_like(probs).clamp_min(1e-9)).clamp_min(1e-9))
		rand_idx = torch.argmax(probs.log() + gumbel, dim=1)
		elite_actions = elite_actions.permute(1, 0, 2, 3)
		actions = torch.gather(
			elite_actions,
			2,
			rand_idx.view(B, 1, 1, 1).expand(B, self.cfg.horizon, 1, self.cfg.action_dim)
		).squeeze(2)
		a = actions[:, 0]
		if not eval_mode:
			a = a + std[:, 0] * torch.randn(B, self.cfg.action_dim, device=std.device)
		if not plan_mask.all():
			cached_a = prev_mean[:, 0]
			if not eval_mode:
				cached_a = cached_a + self.cfg.min_std * torch.randn_like(cached_a)
			a = torch.where(plan_mask.view(B, 1), a, cached_a)
			mean = torch.where(plan_mask.view(B, 1, 1), mean, prev_mean)
		self._prev_mean[:B].copy_(mean)
		self._plan_step_counter[:B] = torch.where(t0, torch.ones_like(counter), counter + 1)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
