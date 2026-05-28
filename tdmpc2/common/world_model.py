from copy import deepcopy

import torch
import torch.nn as nn

try:
	from common import layers, math, init
except ModuleNotFoundError:
	from tdmpc2.common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()
		if getattr(cfg, 'use_cild_heads', False):
			try:
				from common.cild_heads import RiskHead, ProgressHead, OccupancyHead
			except ModuleNotFoundError:
				from tdmpc2.common.cild_heads import RiskHead, ProgressHead, OccupancyHead
			hidden = getattr(cfg, 'cild_head_hidden', 256)
			self._risk_head = RiskHead(cfg.latent_dim, cfg.action_dim, hidden)
			self._progress_head = ProgressHead(cfg.latent_dim, cfg.action_dim, hidden)
			# Occupancy loss intentionally deferred to Phase C: env does not yet expose K-bin occupancy GT.
			self._occupancy_head = OccupancyHead(cfg.latent_dim, getattr(cfg, 'occupancy_dim', 16), hidden)
			self._cild_log_var_risk = nn.Parameter(torch.zeros(()))
			self._cild_log_var_prog = nn.Parameter(torch.zeros(()))

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

	def cild_loss(self, zs, actions, labels):
		"""CILD auxiliary head loss: risk BCE + proximity auxiliary + progress.

		Args:
			zs: latent rollout, shape (H+1, B, latent_dim)
			actions: action sequence, shape (H, B, action_dim)
			labels: dict with keys 'collision_flag', 'min_lidar_dist', 'goal_dist',
				each tensor shape (H+1, B, 1), or None values if not yet wired.

		Returns:
			Scalar loss tensor with gradient through dynamics chain.
		"""
		if not getattr(self.cfg, 'use_cild_heads', False):
			return torch.zeros((), device=zs.device, dtype=zs.dtype)

		if labels is None:
			return torch.zeros((), device=zs.device, dtype=zs.dtype)

		collision_flag = labels.get('collision_flag')
		min_lidar_dist = labels.get('min_lidar_dist')
		goal_dist = labels.get('goal_dist')

		if collision_flag is None or min_lidar_dist is None or goal_dist is None:
			return torch.zeros((), device=zs.device, dtype=zs.dtype)

		H = actions.shape[0]
		device = zs.device

		# ---- detect reset rows (raw NaN in goal_dist) BEFORE sanitize ----
		# A buffer slice that straddles an episode reset contains a reset
		# observation whose label fields are NaN placeholders. We mask those
		# rows out of progress loss because per-step progress is undefined
		# across an episode boundary (the previous step belongs to a
		# different episode). Risk labels are still valid (no collision at
		# reset), so we don't mask them.
		goal_nan_mask = torch.isnan(goal_dist).squeeze(-1)  # (H+1, B), True = reset row

		# ---- sanitize NaN labels (first step of an episode lacks env info) ----
		# Replace NaNs with safe defaults: no collision, far obstacle, no progress.
		coll_clean = torch.nan_to_num(collision_flag, nan=0.0).clamp(0.0, 1.0)
		lidar_clean = torch.nan_to_num(min_lidar_dist, nan=10.0)
		goal_clean = torch.nan_to_num(goal_dist, nan=0.0)

		# ---- (b) horizon-window risk label ----
		k = getattr(self.cfg, 'risk_horizon_k', 5)
		coll = coll_clean.squeeze(-1)  # (H+1, B)
		pad = coll[-1:].expand(min(k, 10), -1)
		coll_padded = torch.cat([coll, pad], dim=0)
		window_label = torch.stack(
			[coll_padded[t:t+k+1].max(dim=0).values for t in range(H+1)], dim=0
		).clamp(0.0, 1.0)  # (H+1, B)

		# ---- risk head prediction ----
		risk_pred = self._risk_head(zs[:-1], actions).squeeze(-1)  # (H, B), in [0,1]
		risk_target = window_label[:-1]  # (H, B)
		bce = torch.nn.functional.binary_cross_entropy(
			risk_pred.clamp(1e-6, 1 - 1e-6), risk_target, reduction='none'
		)

		# ---- (c) min_lidar_dist auxiliary regression ----
		lidar = lidar_clean[:-1].squeeze(-1)  # (H, B), meters
		proximity = torch.exp(-lidar / 0.5).clamp(0, 1)
		mlse = torch.nn.functional.smooth_l1_loss(risk_pred, proximity, reduction='none')

		risk_loss_per_step = bce + 0.3 * mlse  # (H, B)

		# ---- progress head ----
		p_pred = self._progress_head(zs[:-1], actions).squeeze(-1)  # (H, B)
		goal_d = goal_clean.squeeze(-1)  # (H+1, B)
		p_target = goal_d[:-1] - goal_d[1:]  # per-step progress (meters)
		progress_loss_per_step = torch.nn.functional.smooth_l1_loss(
			p_pred, p_target, reduction='none'
		)

		# ---- horizon decay gamma^h ----
		gamma = 0.5
		discount = torch.pow(
			torch.tensor(gamma, device=device),
			torch.arange(H, device=device).float()
		).unsqueeze(-1)  # (H, 1)
		risk_loss = (risk_loss_per_step * discount).mean()

		# Progress label is undefined when either endpoint of the difference
		# (goal_d[t] or goal_d[t+1]) comes from a reset placeholder.
		# Mask those steps out of the progress loss; if every step in the
		# batch is masked (rare), skip the progress term entirely so the
		# log-variance parameter does not receive a one-sided gradient.
		progress_valid = (~(goal_nan_mask[:-1] | goal_nan_mask[1:])).float()  # (H, B)
		progress_weighted = progress_loss_per_step * progress_valid * discount
		valid_weight = (progress_valid * discount).sum()
		# Threshold: at least one valid weighted row required.
		progress_has_data = valid_weight > 1e-8

		# ---- Kendall 2018 uncertainty weighting ----
		total = (torch.exp(-self._cild_log_var_risk) * risk_loss
				 + 0.5 * self._cild_log_var_risk)
		if progress_has_data:
			progress_loss = progress_weighted.sum() / valid_weight
			total = (total
					 + torch.exp(-self._cild_log_var_prog) * progress_loss
					 + 0.5 * self._cild_log_var_prog)
		return total
