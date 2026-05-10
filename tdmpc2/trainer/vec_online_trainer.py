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
		num_rollouts = ceil(self.cfg.eval_episodes / self.cfg.num_envs)
		for _ in range(num_rollouts):
			obs = self.env.reset()
			t0_mask = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
			done_once = torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=obs.device)
			episode_reward = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
			episode_length = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
			episode_success = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=obs.device)
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
				new_done = done & active
				done_once |= new_done
				t0_mask = done
				if bool(done_once.all()):
					break
			ep_rewards.extend(episode_reward.detach().cpu().tolist())
			ep_successes.extend(episode_success.detach().cpu().tolist())
			ep_lengths.extend(episode_length.detach().cpu().tolist())
		limit = self.cfg.eval_episodes
		return dict(
			episode_reward=np.nanmean(ep_rewards[:limit]),
			episode_success=np.nanmean(ep_successes[:limit]),
			episode_length=np.nanmean(ep_lengths[:limit]),
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
				success = info.get('success', torch.zeros_like(reward))
				success_cpu = success.detach().cpu()
				for idx in done_cpu.nonzero(as_tuple=False).flatten().tolist():
					self._episode_rewards.append(float(reward_cpu[idx]))
					self._episode_lengths.append(float(length_cpu[idx]))
					self._episode_successes.append(float(success_cpu[idx]))
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
				metrics = dict(
					episode_reward=np.nanmean(self._episode_rewards) if self._episode_rewards else float('nan'),
					episode_success=np.nanmean(self._episode_successes) if self._episode_successes else float('nan'),
					episode_length=np.nanmean(self._episode_lengths) if self._episode_lengths else float('nan'),
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
