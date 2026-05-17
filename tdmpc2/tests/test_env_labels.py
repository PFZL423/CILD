from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


TD_MPC2_ROOT = Path(__file__).resolve().parents[1]
if str(TD_MPC2_ROOT) not in sys.path:
	sys.path.insert(0, str(TD_MPC2_ROOT))

LABEL_KEYS = ("collision_flag", "min_lidar_dist", "goal_dist")


def _assert_scalar_labels(info):
	for key in LABEL_KEYS:
		assert key in info
		assert isinstance(info[key], float)
	assert info["collision_flag"] in (0.0, 1.0)
	assert info["min_lidar_dist"] > 0.0
	assert info["goal_dist"] > 0.0


def _assert_vector_labels(info, num_envs):
	for key in LABEL_KEYS:
		assert key in info
		value = info[key]
		assert isinstance(value, torch.Tensor)
		assert value.shape == (num_envs,)
		assert torch.is_floating_point(value)
	assert torch.all((info["collision_flag"] == 0.0) | (info["collision_flag"] == 1.0))
	assert torch.all(info["min_lidar_dist"] > 0.0)
	assert torch.all(info["goal_dist"] > 0.0)


def test_safety_gymnasium_wrapper_labels():
	gym = pytest.importorskip("gymnasium")
	safety_gymnasium = pytest.importorskip("safety_gymnasium")
	from tdmpc2.envs.safety_gym import SafetyGymnasiumWrapper

	cfg = SimpleNamespace(cost_lambda=0.0)
	try:
		env = SafetyGymnasiumWrapper(safety_gymnasium.make("SafetyPointGoal1-v0"), cfg)
	except Exception as exc:
		pytest.skip(f"Safety Gymnasium env unavailable: {exc}")
	try:
		obs = env.reset()
		shape = np.asarray(obs).shape
		for _ in range(5):
			action = env.action_space.sample()
			obs, _reward, done, info = env.step(action)
			_assert_scalar_labels(info)
			assert np.asarray(obs).shape == shape
			if done:
				obs = env.reset()
				assert np.asarray(obs).shape == shape
	finally:
		env.close()


def test_safety_gym_vec_labels():
	pytest.importorskip("gymnasium")
	pytest.importorskip("safety_gymnasium")
	from tdmpc2.envs.safety_gym_vec import SafetyGymVecEnv

	cfg = SimpleNamespace(task="SafetyPointGoal1-v0", num_envs=2, cost_lambda=0.0)
	try:
		env = SafetyGymVecEnv(cfg, device="cpu")
	except Exception as exc:
		pytest.skip(f"Safety Gymnasium vector env unavailable: {exc}")
	try:
		obs = env.reset()
		shape = tuple(obs.shape)
		for _ in range(5):
			obs, _reward, _done, info = env.step(env.rand_act())
			_assert_vector_labels(info, cfg.num_envs)
			assert tuple(obs.shape) == shape
	finally:
		env.close()


def test_cild_nav_labels():
	pytest.importorskip("isaaclab")
	pytest.importorskip("isaaclab_tasks")
	import gymnasium as gym
	import tdmpc2.envs.tasks.cild_nav  # noqa: F401
	from tdmpc2.envs.tasks.cild_nav.cild_nav_cfg import CILDNavEnvCfg

	cfg = CILDNavEnvCfg()
	cfg.scene.num_envs = 2
	cfg.sim.device = "cpu"
	cfg.collision_lidar_threshold = 0.3
	try:
		env = gym.make("Isaac-CILDNav-v0", cfg=cfg).unwrapped
	except Exception as exc:
		pytest.skip(f"CILD nav env unavailable: {exc}")
	try:
		obs, _info = env.reset()
		shape = tuple(obs["policy"].shape)
		for _ in range(5):
			action = 2 * torch.rand((env.num_envs, *env.single_action_space.shape), device=env.device) - 1
			obs, _reward, _terminated, _truncated, info = env.step(action)
			_assert_vector_labels(info, env.num_envs)
			assert tuple(obs["policy"].shape) == shape
	finally:
		env.close()
