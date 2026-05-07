from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		ep_costs, ep_goal_reached_counts = [], []
		ep_cost_hazards, ep_cost_vases_c, ep_cost_vases_v = [], [], []
		ep_in_hazard_steps, ep_final_goal_dist = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			# `success` here is a per-step flag from the wrapper; on Safety
			# Gymnasium goal tasks the goal respawns, so the per-episode
			# meaningful summary is goal_reached_count, not success.
			ep_successes.append(info.get('success', 0.0))
			ep_lengths.append(t)
			ep_costs.append(info.get('cost_total', 0.0))
			ep_goal_reached_counts.append(info.get('goal_reached_count', 0.0))
			ep_cost_hazards.append(info.get('cost_hazards_total', 0.0))
			ep_cost_vases_c.append(info.get('cost_vases_contact_total', 0.0))
			ep_cost_vases_v.append(info.get('cost_vases_velocity_total', 0.0))
			ep_in_hazard_steps.append(info.get('in_hazard_steps', 0.0))
			ep_final_goal_dist.append(info.get('final_goal_distance', float('nan')))
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lengths),
			episode_cost=np.nanmean(ep_costs),
			episode_cost_hazards=np.nanmean(ep_cost_hazards),
			episode_cost_vases_contact=np.nanmean(ep_cost_vases_c),
			episode_cost_vases_velocity=np.nanmean(ep_cost_vases_v),
			episode_in_hazard_steps=np.nanmean(ep_in_hazard_steps),
			episode_goal_reached_count=np.nanmean(ep_goal_reached_counts),
			episode_final_goal_distance=np.nanmean(ep_final_goal_dist),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info.get('success', 0.0),
						episode_length=len(self._tds),
						episode_terminated=info['terminated'],
						episode_cost=info.get('cost_total', 0.0),
						episode_cost_hazards=info.get('cost_hazards_total', 0.0),
						episode_cost_vases_contact=info.get('cost_vases_contact_total', 0.0),
						episode_cost_vases_velocity=info.get('cost_vases_velocity_total', 0.0),
						episode_in_hazard_steps=info.get('in_hazard_steps', 0.0),
						episode_goal_reached_count=info.get('goal_reached_count', 0.0),
						episode_final_goal_distance=info.get('final_goal_distance', float('nan')),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			if self._step % self.cfg.checkpoint_freq == 0:
				self.logger.save_agent(self.agent, identifier=self._step)

			self._step += 1

		self.logger.finish(self.agent)
