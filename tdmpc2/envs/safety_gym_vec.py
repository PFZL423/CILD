"""Vectorized Safety Gymnasium environment for TD-MPC2.

Uses gymnasium AsyncVectorEnv to run N safety-gymnasium instances in parallel
sub-processes. Presents the IsaacVecEnv interface expected by VecOnlineTrainer:

    reset()       → torch.Tensor[num_envs, obs_dim]
    step(action)  → (obs, reward, done, info) with torch.Tensor values
    rand_act()    → torch.Tensor[num_envs, action_dim]
    .num_envs     → int
    .observation_space, .action_space → single-env gymnasium spaces
    .max_episode_steps → int

The 6-tuple step (obs, reward, cost, terminated, truncated, info) from
safety-gymnasium is adapted to the standard gymnasium 5-tuple by _SafetyGymShim
so that AsyncVectorEnv can collect results uniformly.
"""

from functools import partial

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

import safety_gymnasium
from envs.safety_gym import SAFETY_GYM_TASKS


class _SafetyGymShim(gym.Env):
    """
    Converts safety-gymnasium's 6-tuple step() to standard gymnasium 5-tuple.
    Must be at module level (not nested) to remain picklable for AsyncVectorEnv.
    """

    def __init__(self, env, cost_lambda: float = 0.0):
        super().__init__()
        self._env = env
        self._cost_lambda = cost_lambda
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=env.action_space.low.astype(np.float32),
            high=env.action_space.high.astype(np.float32),
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float64)
        )
        if self._cost_lambda != 0.0:
            reward = float(reward) - self._cost_lambda * float(cost)
        info = dict(info) if info else {}
        info['cost'] = float(cost)
        info['success'] = float(info.get('goal_met', False))
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._env.close()


def _make_shim(task: str, cost_lambda: float) -> _SafetyGymShim:
    """Module-level factory — picklable for AsyncVectorEnv subprocesses."""
    return _SafetyGymShim(safety_gymnasium.make(task), cost_lambda)


class SafetyGymVecEnv:
    """AsyncVectorEnv wrapper matching the IsaacVecEnv interface."""

    def __init__(self, cfg, device: str = 'cuda:0'):
        if cfg.task not in SAFETY_GYM_TASKS:
            raise ValueError(f'Unknown Safety Gym task: {cfg.task}')
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(cfg.num_envs)
        cost_lambda = float(getattr(cfg, 'cost_lambda', 0.0))

        env_fns = [
            partial(_make_shim, cfg.task, cost_lambda)
            for _ in range(self.num_envs)
        ]
        self._vec = AsyncVectorEnv(env_fns)

        # Single-env metadata: spaces + max_episode_steps
        _probe = safety_gymnasium.make(cfg.task)
        self.observation_space = gym.spaces.Box(
            low=_probe.observation_space.low.astype(np.float32),
            high=_probe.observation_space.high.astype(np.float32),
            shape=_probe.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=_probe.action_space.low.astype(np.float32),
            high=_probe.action_space.high.astype(np.float32),
            shape=_probe.action_space.shape,
            dtype=np.float32,
        )
        self.max_episode_steps = int(_probe.spec.max_episode_steps)
        _probe.close()

    def reset(self) -> torch.Tensor:
        obs, _ = self._vec.reset()
        return torch.from_numpy(obs.astype(np.float32)).to(self.device, non_blocking=True)

    def step(self, action: torch.Tensor):
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, _ = self._vec.step(action_np)

        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device, non_blocking=True)
        reward_t = torch.from_numpy(reward.astype(np.float32)).to(self.device, non_blocking=True)
        terminated_t = torch.from_numpy(np.asarray(terminated, dtype=bool)).to(self.device)
        truncated_t = torch.from_numpy(np.asarray(truncated, dtype=bool)).to(self.device)
        done_t = terminated_t | truncated_t

        info = {
            'terminated': terminated_t,
            'truncated': truncated_t,
            # success/cost tracking omitted for initial impl; trainer falls back to zeros
            'success': torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
        }
        return obs_t, reward_t, done_t, info

    def rand_act(self) -> torch.Tensor:
        low = torch.from_numpy(self.action_space.low)
        high = torch.from_numpy(self.action_space.high)
        return (
            torch.rand(self.num_envs, *self.action_space.shape) * (high - low) + low
        ).to(self.device)

    def close(self):
        self._vec.close()
