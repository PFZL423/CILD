import torch
import torch.nn as nn

try:
	from common import layers
except ModuleNotFoundError:
	from tdmpc2.common import layers


class RiskHead(nn.Module):
	"""
	Predicts the CILD navigation risk label from a latent state-action pair.

	Input:
		z: latent state tensor with shape (..., latent_dim)
		a: action tensor with shape (..., action_dim)

	Output:
		rho: risk probability with shape (..., 1)
	"""

	def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256):
		super().__init__()
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.hidden = hidden
		# Note: do NOT pass act=Sigmoid into mlp — the last layer would become
		# NormedLinear with LayerNorm(1), collapsing the scalar output to 0.
		# Use a plain Linear tail and apply Sigmoid externally.
		self.net = layers.mlp(
			latent_dim + action_dim,
			2 * [hidden],
			1,
		)
		self.act = nn.Sigmoid()

	def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
		"""
		Returns rho in [0, 1] with shape (..., 1).
		"""
		x = torch.cat([z, a], dim=-1)
		return self.act(self.net(x))


class ProgressHead(nn.Module):
	"""
	Predicts scalar CILD navigation progress from a latent state-action pair.

	Input:
		z: latent state tensor with shape (..., latent_dim)
		a: action tensor with shape (..., action_dim)

	Output:
		p: scalar progress prediction with shape (..., 1)
	"""

	def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256):
		super().__init__()
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.hidden = hidden
		self.net = layers.mlp(
			latent_dim + action_dim,
			2 * [hidden],
			1,
		)

	def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
		"""
		Returns scalar progress with shape (..., 1).
		"""
		x = torch.cat([z, a], dim=-1)
		return self.net(x)


class OccupancyHead(nn.Module):
	"""
	Predicts a K-dimensional CILD occupancy vector from a latent state.

	Input:
		z: latent state tensor with shape (..., latent_dim)

	Output:
		occupancy: probabilities with shape (..., K)
	"""

	def __init__(self, latent_dim: int, k: int = 16, hidden: int = 256):
		super().__init__()
		self.latent_dim = latent_dim
		self.k = k
		self.hidden = hidden
		# Plain Linear tail + external Sigmoid (see RiskHead note).
		self.net = layers.mlp(
			latent_dim,
			2 * [hidden],
			k,
		)
		self.act = nn.Sigmoid()

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Returns occupancy probabilities in [0, 1] with shape (..., K).
		"""
		return self.act(self.net(z))
