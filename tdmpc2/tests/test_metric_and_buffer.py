from types import SimpleNamespace

import pytest
import torch
from tensordict.tensordict import TensorDict

from tdmpc2.common.buffer import Buffer
from tdmpc2.trainer.online_trainer import OnlineTrainer


class DummyEvalEnv:
	def __init__(self):
		self._episode = -1
		self._t = 0
		self._lengths = [3, 2]
		self._goals = [2.0, 1.0]

	def reset(self):
		self._episode += 1
		self._t = 0
		return torch.zeros(1)

	def step(self, action):
		self._t += 1
		done = self._t == self._lengths[self._episode]
		info = dict(
			success=1.0 if done else 0.0,
			goal_reached_count=self._goals[self._episode] if done else 0.0,
			final_goal_distance=0.5,
		)
		return torch.zeros(1), 1.0, done, info


class DummyAgent:
	model = None

	def act(self, obs, t0=False, eval_mode=False):
		return torch.zeros(1)


class DummyBufferStorage:
	def __init__(self, td):
		self._td = td

	def sample(self):
		return self._td.clone()


def make_buffer(use_cild_heads=False):
	cfg = SimpleNamespace(
		buffer_size=100,
		steps=100,
		batch_size=2,
		horizon=3,
		multitask=False,
		use_cild_heads=use_cild_heads,
		occupancy_dim=16,
	)
	buffer = Buffer(cfg)
	buffer._device = torch.device('cpu')
	return buffer


def make_sample_td(cfg, include_labels=False):
	n = cfg.batch_size * (cfg.horizon + 1)
	data = dict(
		obs=torch.randn(n, 4),
		action=torch.randn(n, 2),
		reward=torch.randn(n),
		terminated=torch.zeros(n),
	)
	if include_labels:
		data.update(
			collision_flag=torch.zeros(n),
			min_lidar_dist=torch.ones(n),
			goal_dist=torch.arange(n, dtype=torch.float32),
			occupancy_gt=torch.zeros(n, cfg.occupancy_dim),
		)
	return TensorDict(data, batch_size=(n,))


def test_eval_metric_keys(monkeypatch):
	monkeypatch.setattr(torch.compiler, 'cudagraph_mark_step_begin', lambda: None)
	trainer = OnlineTrainer.__new__(OnlineTrainer)
	trainer.cfg = SimpleNamespace(eval_episodes=2, save_video=False)
	trainer.env = DummyEvalEnv()
	trainer.agent = DummyAgent()

	metrics = trainer.eval()

	assert 'goal_throughput_per_step' in metrics
	assert 'mean_steps_per_goal' in metrics
	assert 'total_goals_reached' in metrics
	assert 'total_env_steps' in metrics
	assert metrics['total_goals_reached'] == pytest.approx(3.0)
	assert metrics['total_env_steps'] == 5
	assert metrics['goal_throughput_per_step'] == pytest.approx(3.0 / 5.0)
	assert metrics['mean_steps_per_goal'] == pytest.approx(5.0 / 3.0)
	assert 'episode_success' in metrics


def test_buffer_default_path():
	buffer = make_buffer(use_cild_heads=False)
	buffer._buffer = DummyBufferStorage(make_sample_td(buffer.cfg))

	batch = buffer.sample()

	assert len(batch) == 5


def test_buffer_with_labels():
	buffer = make_buffer(use_cild_heads=True)
	buffer._buffer = DummyBufferStorage(make_sample_td(buffer.cfg, include_labels=True))

	batch = buffer.sample_with_labels()

	assert len(batch) == 9
	collision_flag, min_lidar_dist, goal_dist, occupancy_gt = batch[-4:]
	expected_shape = (buffer.cfg.horizon + 1, buffer.cfg.batch_size, 1)
	assert collision_flag.shape == expected_shape
	assert min_lidar_dist.shape == expected_shape
	assert goal_dist.shape == expected_shape
	assert occupancy_gt.shape == (buffer.cfg.horizon + 1, buffer.cfg.batch_size, buffer.cfg.occupancy_dim)
