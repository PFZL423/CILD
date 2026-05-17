import torch

from tdmpc2.common.cild_heads import OccupancyHead, ProgressHead, RiskHead


def test_risk_head_shape():
	head = RiskHead(latent_dim=512, action_dim=2)
	z = torch.randn(8, 512)
	a = torch.randn(8, 2)

	out = head(z, a)

	assert out.shape == (8, 1)
	assert torch.all((out >= 0) & (out <= 1))


def test_progress_head_shape():
	head = ProgressHead(latent_dim=512, action_dim=2)
	z = torch.randn(8, 512)
	a = torch.randn(8, 2)

	out = head(z, a)

	assert out.shape == (8, 1)


def test_occupancy_head_shape():
	head = OccupancyHead(latent_dim=512, k=16)
	z = torch.randn(8, 512)

	out = head(z)

	assert out.shape == (8, 16)
	assert torch.all((out >= 0) & (out <= 1))


def test_risk_head_is_input_dependent():
	"""Regression: previously LayerNorm on the scalar output collapsed rho to 0.5."""
	torch.manual_seed(0)
	head = RiskHead(latent_dim=64, action_dim=2)
	z = torch.randn(32, 64)
	a = torch.randn(32, 2)
	out = head(z, a).squeeze(-1)
	assert out.std().item() > 1e-3, f"Risk head output appears constant (std={out.std().item()})"


def test_occupancy_head_is_input_dependent():
	torch.manual_seed(0)
	head = OccupancyHead(latent_dim=64, k=16)
	z = torch.randn(32, 64)
	out = head(z)
	assert out.std(dim=0).mean().item() > 1e-3, "Occupancy head appears constant across batch"
