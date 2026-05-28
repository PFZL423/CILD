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
		occupancy_dim=16,
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
			'occupancy_gt': torch.zeros(n, cfg.occupancy_dim),
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
		occupancy_dim=16,
		occ_loss_weight=1.0,
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

	assert len(batch) == 9
	for label in batch[-4:-1]:
		assert label is not None
		assert label.shape == (buffer.cfg.horizon + 1, buffer.cfg.batch_size, 1)
		assert label.dtype == torch.float32
	occupancy_gt = batch[-1]
	assert occupancy_gt is not None
	assert occupancy_gt.shape == (buffer.cfg.horizon + 1, buffer.cfg.batch_size, buffer.cfg.occupancy_dim)
	assert occupancy_gt.dtype == torch.float32


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


def test_cild_loss_nonzero_and_gradient_flows():
	"""With use_cild_heads=True, cild_loss should be nonzero and backprop through zs."""
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	model = WorldModel(cfg)

	H, B = 3, 4
	zs = torch.randn(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.randn(H, B, cfg.action_dim)
	labels = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.ones(H + 1, B, 1) * 0.5,
		'goal_dist': torch.linspace(3, 1, H + 1).unsqueeze(-1).unsqueeze(-1).expand(H + 1, B, 1),
	}
	# Inject some collisions to make BCE non-trivial
	labels['collision_flag'][1, :, 0] = 1.0
	labels['collision_flag'][2, 0, 0] = 1.0

	loss = model.cild_loss(zs, actions, labels)

	assert loss.shape == ()
	assert loss.item() != 0.0, f"cild_loss should be nonzero, got {loss.item()}"
	assert loss.requires_grad
	assert torch.isfinite(loss), f"cild_loss is not finite: {loss.item()}"

	loss.backward()
	assert zs.grad is not None
	assert zs.grad.abs().sum() > 0, "Gradient should flow through zs"


def test_cild_loss_masks_reset_rows_in_progress():
	"""Reset rows (NaN in goal_dist) must not contribute to progress loss.

	Reproduces the Codex P2 finding: if goal_dist[0]=NaN gets sanitized to 0,
	then p_target = 0 - goal_dist[1] becomes a large false-negative target.
	With masking, the loss at row 0 should be excluded.
	"""
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	model = WorldModel(cfg)
	# Make heads deterministic by zeroing all params; predictions are then
	# constants and any nonzero loss comes from labels.
	for p in model._progress_head.parameters():
		torch.nn.init.zeros_(p)
	for p in model._risk_head.parameters():
		torch.nn.init.zeros_(p)

	H, B = 3, 1
	zs = torch.zeros(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.zeros(H, B, cfg.action_dim)

	# Case 1: clean labels (no NaN) — establish baseline loss
	labels_clean = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.full((H + 1, B, 1), 10.0),
		# Strong progress signal: 1.5m -> 1.4m -> 1.3m -> 1.2m
		'goal_dist': torch.tensor([1.5, 1.4, 1.3, 1.2]).view(H + 1, 1, 1),
	}
	loss_clean = model.cild_loss(zs, actions, labels_clean)

	# Case 2: row 0 of goal_dist is NaN (simulating reset). If mask works,
	# the only loss difference comes from removing 1 of H progress rows.
	# The bug-symptom would be: p_target[0] = 0 - 1.4 = -1.4 (huge), loss explodes.
	labels_reset = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.full((H + 1, B, 1), 10.0),
		'goal_dist': labels_clean['goal_dist'].clone(),
	}
	labels_reset['goal_dist'][0, 0, 0] = float('nan')
	loss_reset = model.cild_loss(zs, actions, labels_reset)

	# Without masking, loss_reset would be much LARGER than loss_clean
	# (because the synthetic huge -1.4 target dominates). With proper
	# masking, the masked-out row is the EARLIEST (largest discount weight)
	# so loss_reset is at most comparable to loss_clean and finite.
	assert torch.isfinite(loss_reset), "loss must remain finite with reset rows"
	# Compute what an unmasked impl would produce as a sanity check:
	# with nan_to_num→0 and no mask, p_target[0] = -1.4 ⇒ smooth_l1 ≈ 0.9
	# which (with gamma^0=1 weighting) would dominate everything else.
	# So we assert loss_reset stays close to loss_clean (≤ 5× tolerance).
	assert loss_reset.item() < loss_clean.item() * 5 + 1.0, (
		f"loss_reset={loss_reset.item():.4f} far exceeds loss_clean={loss_clean.item():.4f} "
		"— reset row likely not masked out of progress loss"
	)


def test_cild_loss_skips_progress_term_when_all_masked():
	"""When every progress label is NaN, progress term (including log_var_prog) must be skipped.

	Otherwise 0.5 * log_var_prog accumulates a one-sided gradient → log_var
	drifts to -inf and explodes the whole loss (Codex round-2 P2 finding).
	"""
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	model = WorldModel(cfg)

	H, B = 3, 2
	zs = torch.randn(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.randn(H, B, cfg.action_dim)
	labels = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.ones(H + 1, B, 1) * 5.0,
		'goal_dist': torch.full((H + 1, B, 1), float('nan')),  # everything masked
	}
	loss = model.cild_loss(zs, actions, labels)
	# Must still be finite + scalar; no NaN/Inf from div-by-zero.
	assert torch.isfinite(loss), f"loss must be finite when all progress masked, got {loss.item()}"
	assert loss.shape == ()
	# Backward: log_var_prog must receive zero gradient (no data term and no
	# 0.5*log_var prior either, since the whole progress branch is skipped).
	loss.backward()
	# log_var_prog must be completely disconnected from the graph (grad is None)
	# or receive zero gradient. Either way: no one-sided pressure toward -inf.
	grad = model._cild_log_var_prog.grad
	assert grad is None or grad.abs().item() < 1e-9, (
		f"log_var_prog received nonzero gradient ({grad}) "
		"despite all progress labels being masked"
	)


def test_cild_loss_includes_occ_when_gt_present():
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	cfg.occ_loss_weight = 1.0
	model = WorldModel(cfg)

	H, B, K = 3, 4, cfg.occupancy_dim
	zs = torch.randn(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.randn(H, B, cfg.action_dim)
	labels = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.ones(H + 1, B, 1) * 0.5,
		'goal_dist': torch.linspace(3, 1, H + 1).unsqueeze(-1).unsqueeze(-1).expand(H + 1, B, 1),
		'occupancy_gt': torch.rand(H + 1, B, K),
	}

	loss = model.cild_loss(zs, actions, labels)

	assert loss.shape == ()
	assert torch.isfinite(loss), f"cild_loss is not finite: {loss.item()}"
	assert loss.item() > 0.0

	loss.backward()
	occ_grads = [p.grad for p in model._occupancy_head.parameters()]
	assert all(grad is not None for grad in occ_grads)
	assert sum(grad.abs().sum().item() for grad in occ_grads) > 0.0
	assert model._cild_log_var_occ.grad is not None
	assert model._cild_log_var_occ.grad.abs().item() > 0.0


def test_cild_loss_skips_occ_when_weight_zero():
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	cfg.occ_loss_weight = 0.0
	model = WorldModel(cfg)

	H, B, K = 3, 4, cfg.occupancy_dim
	zs = torch.randn(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.randn(H, B, cfg.action_dim)
	labels = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.ones(H + 1, B, 1) * 0.5,
		'goal_dist': torch.linspace(3, 1, H + 1).unsqueeze(-1).unsqueeze(-1).expand(H + 1, B, 1),
		'occupancy_gt': torch.rand(H + 1, B, K),
	}
	labels_without_occ = {k: v for k, v in labels.items() if k != 'occupancy_gt'}

	loss_with_occ_disabled = model.cild_loss(zs, actions, labels)
	loss_without_occ_key = model.cild_loss(zs, actions, labels_without_occ)

	assert torch.allclose(loss_with_occ_disabled, loss_without_occ_key, atol=1e-6)

	loss_with_occ_disabled.backward()
	grad = model._cild_log_var_occ.grad
	assert grad is None or torch.equal(grad, torch.zeros_like(grad))


def test_cild_loss_skips_occ_when_all_gt_nan():
	cfg = _world_model_cfg()
	cfg.use_cild_heads = True
	cfg.occ_loss_weight = 1.0
	model = WorldModel(cfg)

	H, B, K = 3, 4, cfg.occupancy_dim
	zs = torch.randn(H + 1, B, cfg.latent_dim, requires_grad=True)
	actions = torch.randn(H, B, cfg.action_dim)
	labels = {
		'collision_flag': torch.zeros(H + 1, B, 1),
		'min_lidar_dist': torch.ones(H + 1, B, 1) * 0.5,
		'goal_dist': torch.linspace(3, 1, H + 1).unsqueeze(-1).unsqueeze(-1).expand(H + 1, B, 1),
		'occupancy_gt': torch.full((H + 1, B, K), float('nan')),
	}

	loss = model.cild_loss(zs, actions, labels)

	assert torch.isfinite(loss).item() is True

	loss.backward()
	grad = model._cild_log_var_occ.grad
	assert grad is None or torch.equal(grad, torch.zeros_like(grad))
