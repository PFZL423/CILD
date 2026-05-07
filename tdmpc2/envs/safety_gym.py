"""Safety Gymnasium wrapper for TD-MPC2.

Adapts safety-gymnasium 1.0 environments to the 4-tuple step API expected by
TD-MPC2's TensorWrapper:

    safety-gymnasium step:  (obs, reward, cost, terminated, truncated, info)
    TD-MPC2 expects:        (obs, reward, done, info)

The cost signal and goal-reached flag are passed through `info`. CILD will
later consume `info['cost']` (or its decomposition) when training the risk
prediction head.

Per RESEARCH_NOTES.md:
- No reward shaping. Use safety-gymnasium defaults.
- No cfg overrides. Run TD-MPC2 with stock hyperparameters.
- Baseline performance is an observation, not a tunable.
"""

import gymnasium as gym
import numpy as np

import safety_gymnasium


# Tasks targeted by the motivation experiments. Static / dynamic / different
# robot morphology — three points strengthen the "broadly limited" story.
SAFETY_GYM_TASKS = {
	'SafetyPointGoal1-v0',   # static obstacles
	'SafetyPointGoal2-v0',   # dynamic obstacles (vases)
	'SafetyCarGoal1-v0',     # different robot morphology, static
}


class SafetyGymnasiumWrapper(gym.Wrapper):
	"""Bridge between safety-gymnasium 1.0 and TD-MPC2's TensorWrapper.

	- Collapses the 6-tuple step into a 4-tuple `(obs, reward, done, info)`.
	- `reset()` returns obs only (TensorWrapper does not pass info through).
	- Surfaces cost and goal-reached flags via `info`.
	"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		# Used to populate info['success'] without requiring a 'success' key
		# from the underlying env. Reset on each reset().
		self._goal_reached_count = 0
		self._cost_total = 0.0

	def reset(self):
		self._goal_reached_count = 0
		self._cost_total = 0.0
		obs, _info = self.env.reset()
		# safety-gymnasium returns float64; let TensorWrapper handle the cast
		return obs

	def step(self, action):
		# Safety-Gymnasium's underlying space is float64; TD-MPC2 actions are
		# float32 numpy by the time they reach the wrapper. Cast to satisfy
		# the action space dtype check.
		obs, reward, cost, terminated, truncated, info = self.env.step(
			np.asarray(action, dtype=np.float64)
		)
		done = bool(terminated or truncated)

		# Read goal_achieved from the underlying task object. This is the
		# per-step "the agent reached the goal this step" flag — a goal env
		# can be reached multiple times per episode (the goal respawns).
		goal_reached = bool(getattr(self.env.unwrapped.task, 'goal_achieved', False))
		if goal_reached:
			self._goal_reached_count += 1
		self._cost_total += float(cost)

		# Fields TensorWrapper requires:
		#   info['success']    — per-step bool/float
		#   info['terminated'] — distinguishes natural termination vs truncation
		# Plus CILD-relevant signals.
		info = dict(info) if info else {}
		info['success'] = float(goal_reached)
		info['terminated'] = bool(terminated)
		info['cost'] = float(cost)
		info['cost_total'] = float(self._cost_total)
		info['goal_reached_count'] = float(self._goal_reached_count)

		return obs, reward, done, info

	@property
	def max_episode_steps(self):
		# safety-gymnasium exposes this via env.spec.max_episode_steps
		return self.env.spec.max_episode_steps

	def render(self, *args, **kwargs):
		# Headless training. Safety-Gymnasium has its own render path but
		# motivation runs go with save_video=false to keep things simple.
		raise RuntimeError(
			'SafetyGymnasiumWrapper.render is not enabled; run with save_video=false.'
		)


def make_env(cfg):
	"""Create a Safety Gymnasium environment for TD-MPC2."""
	if cfg.task not in SAFETY_GYM_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'Safety Gymnasium wrapper supports state observations only.'

	env = safety_gymnasium.make(cfg.task)
	env = SafetyGymnasiumWrapper(env, cfg)
	return env
