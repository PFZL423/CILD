from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
try:
	from trainer.base import Trainer
except ModuleNotFoundError:
	from .base import Trainer


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
		ep_rewards, ep_raw_rewards, ep_successes, ep_lengths = [], [], [], []
		ep_costs, ep_goal_reached_counts = [], []
		ep_cost_hazards, ep_cost_vases_c, ep_cost_vases_v = [], [], []
		ep_in_hazard_steps, ep_final_goal_dist = [], []
		total_goals_reached, total_env_steps = 0.0, 0
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			ep_goal_reached_count = 0.0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				ep_goal_reached_count = info.get('goal_reached_count', ep_goal_reached_count)
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_raw_rewards.append(info.get('raw_reward_total', float('nan')))
			# Legacy metric: `success` is only the final step's flag. For
			# respawning-goal tasks, use throughput metrics below instead.
			ep_successes.append(info.get('success', 0.0))
			ep_lengths.append(t)
			ep_costs.append(info.get('cost_total', 0.0))
			ep_goal_reached_count = info.get('goal_reached_count', ep_goal_reached_count)
			ep_goal_reached_counts.append(ep_goal_reached_count)
			total_goals_reached += ep_goal_reached_count
			total_env_steps += t
			ep_cost_hazards.append(info.get('cost_hazards_total', 0.0))
			ep_cost_vases_c.append(info.get('cost_vases_contact_total', 0.0))
			ep_cost_vases_v.append(info.get('cost_vases_velocity_total', 0.0))
			ep_in_hazard_steps.append(info.get('in_hazard_steps', 0.0))
			ep_final_goal_dist.append(info.get('final_goal_distance', float('nan')))
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_raw_reward=np.nanmean(ep_raw_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lengths),
			episode_cost=np.nanmean(ep_costs),
			episode_cost_hazards=np.nanmean(ep_cost_hazards),
			episode_cost_vases_contact=np.nanmean(ep_cost_vases_c),
			episode_cost_vases_velocity=np.nanmean(ep_cost_vases_v),
			episode_in_hazard_steps=np.nanmean(ep_in_hazard_steps),
			episode_goal_reached_count=np.nanmean(ep_goal_reached_counts),
			episode_final_goal_distance=np.nanmean(ep_final_goal_dist),
			goal_throughput_per_step=total_goals_reached / max(total_env_steps, 1),
			mean_steps_per_goal=total_env_steps / max(total_goals_reached, 1),
			total_goals_reached=total_goals_reached,
			total_env_steps=total_env_steps,
		)

	def to_td(self, obs, action=None, reward=None, terminated=None,
		  collision_flag=None, min_lidar_dist=None, goal_dist=None,
		  occupancy_gt=None):
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
		if collision_flag is None:
			collision_flag = torch.tensor(float('nan'))
		elif not isinstance(collision_flag, torch.Tensor):
			collision_flag = torch.tensor(collision_flag)
		if min_lidar_dist is None:
			min_lidar_dist = torch.tensor(float('nan'))
		elif not isinstance(min_lidar_dist, torch.Tensor):
			min_lidar_dist = torch.tensor(min_lidar_dist)
		if goal_dist is None:
			goal_dist = torch.tensor(float('nan'))
		elif not isinstance(goal_dist, torch.Tensor):
			goal_dist = torch.tensor(goal_dist)
		if occupancy_gt is None:
			occupancy_gt = torch.full((int(getattr(self.cfg, 'occupancy_dim', 16)),), float('nan'))
		elif not isinstance(occupancy_gt, torch.Tensor):
			occupancy_gt = torch.tensor(occupancy_gt)
		data = dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		)
		if getattr(getattr(self, 'cfg', None), 'use_cild_heads', False):
			data.update(
				collision_flag=collision_flag.unsqueeze(0),
				min_lidar_dist=min_lidar_dist.unsqueeze(0),
				goal_dist=goal_dist.unsqueeze(0),
				occupancy_gt=occupancy_gt.unsqueeze(0),
			)
		td = TensorDict(data, batch_size=(1,))
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
						episode_raw_reward=info.get('raw_reward_total', float('nan')),
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
			self._tds.append(self.to_td(
				obs, action, reward, info['terminated'],
				collision_flag=info.get('collision_flag', float('nan')),
				min_lidar_dist=info.get('min_lidar_dist', float('nan')),
				goal_dist=info.get('goal_dist', float('nan')),
				occupancy_gt=info.get('occupancy_gt', np.full(int(getattr(self.cfg, 'occupancy_dim', 16)), np.nan, dtype=np.float32)),
			))

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
