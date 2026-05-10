from collections import deque
from math import ceil
from time import time

import numpy as np
import torch
from trainer.base import Trainer


class VecOnlineTrainer(Trainer):
	"""Step-based online trainer for IsaacLab vectorized environments."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._env_step_total = 0
		self._update_credit = 0.0
		self._update_count = 0
		self._start_time = time()
		self._episode_rewards = deque(maxlen=100)
		self._episode_lengths = deque(maxlen=100)
		self._episode_successes = deque(maxlen=100)
		# Per-episode metric deques (parity with single-env OnlineTrainer).
		self._episode_costs = deque(maxlen=100)
		self._episode_cost_hazards = deque(maxlen=100)
		self._episode_cost_vases_c = deque(maxlen=100)
		self._episode_cost_vases_v = deque(maxlen=100)
		self._episode_in_hazard = deque(maxlen=100)
		self._episode_goal_reached = deque(maxlen=100)
		self._episode_final_goal_dist = deque(maxlen=100)

	def _to_float(self, value):
		"""Convert scalar-like values to plain Python floats for logging."""
		if isinstance(value, torch.Tensor):
			return value.detach().float().mean().item()
		return float(value)

	def common_metrics(self):
		"""Return common step and throughput metrics."""
		elapsed_time = time() - self._start_time
		elapsed_time = max(elapsed_time, 1e-9)
		return dict(
			step=self._env_step_total,
			env_step=self._env_step_total,
			elapsed_time=elapsed_time,
			env_steps_per_second=self._env_step_total / elapsed_time,
			throughput=self._env_step_total / elapsed_time,
			updates_per_second=self._update_count / elapsed_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent in full-vector rollout batches."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		ep_costs, ep_ch, ep_cvc, ep_cvv = [], [], [], []
		ep_ih, ep_gr, ep_fgd = [], [], []
		num_rollouts = ceil(self.cfg.eval_episodes / self.cfg.num_envs)
		for _ in range(num_rollouts):
			obs = self.env.reset()
			t0_mask = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
			done_once = torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
			episode_reward = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
			episode_length = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
			episode_success = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
			# Latest metric snapshot per env (overwritten each step until that
			# env's first 'done' stops updating it via the active mask).
			last_metrics = {k: torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
				for k in ('cost_total', 'cost_hazards_total',
					'cost_vases_contact_total', 'cost_vases_velocity_total',
					'in_hazard_steps', 'goal_reached_count', 'final_goal_distance')}
			for _ in range(self.cfg.episode_length):
				action = self.agent.act(obs, t0=t0_mask, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				active = ~done_once
				episode_reward[active] += reward[active].float()
				episode_length[active] += 1
				if 'success' in info:
					episode_success[active] = torch.maximum(
						episode_success[active],
						info['success'][active].float(),
					)
				for k, buf in last_metrics.items():
					if k in info:
						buf[active] = info[k][active].float()
				new_done = done & active
				done_once |= new_done
				t0_mask = done
				if bool(done_once.all()):
					break
			ep_rewards.extend(episode_reward.detach().cpu().tolist())
			ep_successes.extend(episode_success.detach().cpu().tolist())
			ep_lengths.extend(episode_length.detach().cpu().tolist())
			ep_costs.extend(last_metrics['cost_total'].detach().cpu().tolist())
			ep_ch.extend(last_metrics['cost_hazards_total'].detach().cpu().tolist())
			ep_cvc.extend(last_metrics['cost_vases_contact_total'].detach().cpu().tolist())
			ep_cvv.extend(last_metrics['cost_vases_velocity_total'].detach().cpu().tolist())
			ep_ih.extend(last_metrics['in_hazard_steps'].detach().cpu().tolist())
			ep_gr.extend(last_metrics['goal_reached_count'].detach().cpu().tolist())
			ep_fgd.extend(last_metrics['final_goal_distance'].detach().cpu().tolist())
		limit = self.cfg.eval_episodes
		return dict(
			episode_reward=np.nanmean(ep_rewards[:limit]),
			episode_success=np.nanmean(ep_successes[:limit]),
			episode_length=np.nanmean(ep_lengths[:limit]),
			episode_cost=np.nanmean(ep_costs[:limit]),
			episode_cost_hazards=np.nanmean(ep_ch[:limit]),
			episode_cost_vases_contact=np.nanmean(ep_cvc[:limit]),
			episode_cost_vases_velocity=np.nanmean(ep_cvv[:limit]),
			episode_in_hazard_steps=np.nanmean(ep_ih[:limit]),
			episode_goal_reached_count=np.nanmean(ep_gr[:limit]),
			episode_final_goal_distance=np.nanmean(ep_fgd[:limit]),
		)

	def train(self):
		"""Train a TD-MPC2 agent with step-based collection and UTD-driven updates.

		cfg.steps / eval_freq / log_freq / checkpoint_freq are interpreted as
		ENVIRONMENT steps (transitions), matching single-env conventions. Each
		vec tick contributes num_envs to env_step_total.
		"""
		obs = self.env.reset()
		t0_mask = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
		episode_reward = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
		episode_length = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
		train_metrics = {}
		next_log = self.cfg.log_freq
		next_eval = self.cfg.eval_freq
		next_ckpt = self.cfg.checkpoint_freq
		while self._env_step_total < self.cfg.steps:
			if self._env_step_total < self.cfg.seed_steps:
				action = self.env.rand_act()
			else:
				action = self.agent.act(obs, t0=t0_mask, eval_mode=False)

			next_obs, reward, done, info = self.env.step(action)
			self.buffer.add_step(obs, action, reward, info['terminated'], info['truncated'])

			episode_reward += reward.float()
			episode_length += 1
			if bool(done.any()):
				done_cpu = done.detach().cpu()
				reward_cpu = episode_reward.detach().cpu()
				length_cpu = episode_length.detach().cpu()
				zeros = torch.zeros_like(reward)
				# Per-episode totals are emitted by _SafetyGymShim on every
				# step; their value at the done step is the episode total.
				success_cpu = info.get('success', zeros).detach().cpu()
				cost_cpu = info.get('cost_total', zeros).detach().cpu()
				ch_cpu = info.get('cost_hazards_total', zeros).detach().cpu()
				cvc_cpu = info.get('cost_vases_contact_total', zeros).detach().cpu()
				cvv_cpu = info.get('cost_vases_velocity_total', zeros).detach().cpu()
				ih_cpu = info.get('in_hazard_steps', zeros).detach().cpu()
				gr_cpu = info.get('goal_reached_count', zeros).detach().cpu()
				fgd_cpu = info.get('final_goal_distance', zeros).detach().cpu()
				for idx in done_cpu.nonzero(as_tuple=False).flatten().tolist():
					self._episode_rewards.append(float(reward_cpu[idx]))
					self._episode_lengths.append(float(length_cpu[idx]))
					self._episode_successes.append(float(success_cpu[idx]))
					self._episode_costs.append(float(cost_cpu[idx]))
					self._episode_cost_hazards.append(float(ch_cpu[idx]))
					self._episode_cost_vases_c.append(float(cvc_cpu[idx]))
					self._episode_cost_vases_v.append(float(cvv_cpu[idx]))
					self._episode_in_hazard.append(float(ih_cpu[idx]))
					self._episode_goal_reached.append(float(gr_cpu[idx]))
					self._episode_final_goal_dist.append(float(fgd_cpu[idx]))
				episode_reward[done] = 0
				episode_length[done] = 0

			self._update_credit += self.cfg.num_envs * self.cfg.utd
			if self._env_step_total >= self.cfg.seed_steps:
				num_updates = int(self._update_credit)
				self._update_credit -= num_updates
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
					train_metrics.update({k: self._to_float(v) for k, v in _train_metrics.items()})
				self._update_count += num_updates

			obs = next_obs
			t0_mask = done
			self._step += 1
			self._env_step_total += self.cfg.num_envs

			if self._env_step_total >= next_log:
				_nanmean = lambda d: np.nanmean(d) if d else float('nan')
				metrics = dict(
					episode_reward=_nanmean(self._episode_rewards),
					episode_success=_nanmean(self._episode_successes),
					episode_length=_nanmean(self._episode_lengths),
					episode_cost=_nanmean(self._episode_costs),
					episode_cost_hazards=_nanmean(self._episode_cost_hazards),
					episode_cost_vases_contact=_nanmean(self._episode_cost_vases_c),
					episode_cost_vases_velocity=_nanmean(self._episode_cost_vases_v),
					episode_in_hazard_steps=_nanmean(self._episode_in_hazard),
					episode_goal_reached_count=_nanmean(self._episode_goal_reached),
					episode_final_goal_distance=_nanmean(self._episode_final_goal_dist),
				)
				metrics.update(train_metrics)
				metrics.update(self.common_metrics())
				self.logger.log(metrics, 'train')
				next_log = ((self._env_step_total // self.cfg.log_freq) + 1) * self.cfg.log_freq

			if self._env_step_total >= next_eval:
				eval_metrics = self.eval()
				eval_metrics.update(self.common_metrics())
				self.logger.log(eval_metrics, 'eval')
				obs = self.env.reset()
				t0_mask = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
				episode_reward.zero_()
				episode_length.zero_()
				next_eval = ((self._env_step_total // self.cfg.eval_freq) + 1) * self.cfg.eval_freq

			if self._env_step_total >= next_ckpt:
				self.logger.save_agent(self.agent, identifier=self._env_step_total)
				next_ckpt = ((self._env_step_total // self.cfg.checkpoint_freq) + 1) * self.cfg.checkpoint_freq

		self.logger.finish(self.agent)
