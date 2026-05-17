from types import SimpleNamespace

import torch
from tensordict.tensordict import TensorDict

from tdmpc2.common.buffer import Buffer
from tdmpc2.common.world_model import WorldModel
from tdmpc2.trainer.online_trainer import OnlineTrainer


class FakeEnv:
	def __init__(self):
		self.t = 0

	def reset(self):
		self.t = 0
		return torch.zeros(4)

	def rand_act(self):
		return torch.zeros(2)

	def step(self, action):
		self.t += 1
		info = {
			'terminated': torch.tensor(False),
			'collision_flag': float(self.t % 2),
			'min_lidar_dist': float(10 - self.t),
			'goal_dist': float(5 - self.t),
		}
		return torch.full((4,), float(self.t)), torch.tensor(1.0), False, info


class DummyBufferStorage:
	def __init__(self, td):
		self._td = td

	def sample(self):
		return self._td.clone()


def _make_buffer(use_cild_heads=True):
	cfg = SimpleNamespace(
		buffer_size=100,
		steps=100,
		batch_size=2,
		horizon=3,
		multitask=False,
		use_cild_heads=use_cild_heads,
	)
	buffer = Buffer(cfg)
	buffer._device = torch.device('cpu')
	return buffer


def _make_sample_td(cfg):
	n = cfg.batch_size * (cfg.horizon + 1)
	return TensorDict(
		{
			'obs': torch.randn(n, 4),
			'action': torch.randn(n, 2),
			'reward': torch.randn(n),
			'terminated': torch.zeros(n),
			'collision_flag': torch.zeros(n),
			'min_lidar_dist': torch.ones(n),
			'goal_dist': torch.arange(n, dtype=torch.float32),
		},
		batch_size=(n,),
	)


def _world_model_cfg():
	return SimpleNamespace(
		multitask=False,
		obs='state',
		obs_shape={'state': (4,)},
		task_dim=0,
		num_enc_layers=1,
		enc_dim=8,
		latent_dim=8,
		simnorm_dim=4,
		action_dim=2,
		mlp_dim=8,
		num_bins=1,
		episodic=False,
		num_q=2,
		dropout=0.0,
		log_std_min=-10,
		log_std_max=2,
		use_cild_heads=False,
	)


def test_online_trainer_to_td_writes_cild_labels():
	trainer = OnlineTrainer.__new__(OnlineTrainer)
	trainer.cfg = SimpleNamespace(use_cild_heads=True)
	trainer.env = FakeEnv()
	obs = trainer.env.reset()
	trainer._tds = [trainer.to_td(obs)]

	for _ in range(5):
		action = trainer.env.rand_act()
		obs, reward, _done, info = trainer.env.step(action)
		trainer._tds.append(trainer.to_td(
			obs, action, reward, info['terminated'],
			collision_flag=info.get('collision_flag', float('nan')),
			min_lidar_dist=info.get('min_lidar_dist', float('nan')),
			goal_dist=info.get('goal_dist', float('nan')),
		))

	for td in trainer._tds:
		for key in ('collision_flag', 'min_lidar_dist', 'goal_dist'):
			assert key in td
			assert td[key].shape == (1,)


def test_buffer_sample_with_labels_returns_label_tensors():
	buffer = _make_buffer(use_cild_heads=True)
	buffer._buffer = DummyBufferStorage(_make_sample_td(buffer.cfg))

	batch = buffer.sample_with_labels()

	assert len(batch) == 8
	for label in batch[-3:]:
		assert label is not None
		assert label.shape == (buffer.cfg.horizon + 1, buffer.cfg.batch_size, 1)


def test_world_model_cild_loss_is_zero_scalar_and_graph_safe():
	model = WorldModel(_world_model_cfg())
	zs = torch.randn(4, 2, 8, requires_grad=True)
	actions = torch.randn(3, 2, 2)
	labels = {
		'collision_flag': torch.zeros(4, 2, 1),
		'min_lidar_dist': torch.ones(4, 2, 1),
		'goal_dist': torch.ones(4, 2, 1),
	}

	loss = model.cild_loss(zs, actions, labels)

	assert loss.shape == ()
	assert loss.item() == 0.0
	assert not loss.requires_grad
	((zs * 0).sum() + loss).backward()
	assert zs.grad is not None
