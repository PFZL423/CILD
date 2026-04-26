from time import time
from pathlib import Path

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from common.trajectory_viz import save_ugv_trajectory
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._episode_states = []
		self._next_trajectory_step = self.cfg.get('trajectory_freq', None) if self.cfg.get('save_trajectory', False) else None

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
		ep_rewards, ep_successes, ep_collisions, ep_lengths = [], [], [], []
		ep_goal_distances, ep_path_lengths, ep_out_of_bounds = [], [], []
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
			ep_successes.append(info['success'])
			ep_collisions.append(info['collision_total'])
			ep_lengths.append(t)
			ep_goal_distances.append(info.get('goal_distance', 0.0))
			ep_path_lengths.append(info.get('path_length', 0.0))
			ep_out_of_bounds.append(info.get('out_of_bounds', 0.0))
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_collision=np.nanmean(ep_collisions),
			episode_length= np.nanmean(ep_lengths),
			episode_goal_distance=np.nanmean(ep_goal_distances),
			episode_path_length=np.nanmean(ep_path_lengths),
			episode_out_of_bounds=np.nanmean(ep_out_of_bounds),
		)

	def _maybe_save_trajectory(self):
		"""Save the latest UGV training episode trajectory at a fixed step interval."""
		if not self.cfg.get('save_trajectory', False):
			return
		if self._next_trajectory_step is None or self._step < self._next_trajectory_step:
			return
		base_env = self.env.unwrapped
		if not all(hasattr(base_env, attr) for attr in ('robot_state', 'goal', 'map_size')):
			return
		if len(self._episode_states) == 0:
			return
		traj_dir = Path(self.cfg.work_dir) / 'trajectories'
		save_path = traj_dir / f'step_{self._step:09d}.png'
		save_ugv_trajectory(
			states=self._episode_states,
			goal=base_env.goal,
			map_size=base_env.map_size,
			save_path=save_path,
			title=f'UGV Trajectory @ step {self._step}',
		)
		while self._next_trajectory_step is not None and self._step >= self._next_trajectory_step:
			self._next_trajectory_step += self.cfg.trajectory_freq

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
					self._maybe_save_trajectory()
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_collision=info['collision_total'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]
				self._episode_states = []

			# Collect experience
			base_env = self.env.unwrapped
			if hasattr(base_env, 'robot_state'):
				self._episode_states.append(base_env.robot_state.copy())
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))
			if done and hasattr(base_env, 'robot_state'):
				self._episode_states.append(base_env.robot_state.copy())

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
			self._maybe_save_trajectory()

		self.logger.finish(self.agent)
