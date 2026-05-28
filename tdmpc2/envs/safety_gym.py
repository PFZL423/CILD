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


def _min_obstacle_surface_dist(task):
	"""Agent center-to-obstacle surface distance (meters) via task object.

	Returns the minimum over all hazards and vases. Positive = safe gap,
	zero = touching, negative = penetrating.
	"""
	agent_xy = task.agent.pos[:2]
	dists = []
	if task.hazards.num > 0:
		hz_xy = np.array([h[:2] for h in task.hazards.pos])
		hz_surface = np.linalg.norm(hz_xy - agent_xy, axis=1) - task.hazards.size
		dists.append(float(hz_surface.min()))
	if task.vases.num > 0:
		vs_xy = np.array([v[:2] for v in task.vases.pos])
		vs_surface = np.linalg.norm(vs_xy - agent_xy, axis=1) - task.vases.size
		dists.append(float(vs_surface.min()))
	return min(dists) if dists else float('inf')


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
		# Per-episode accumulators. Reset on each reset().
		self._goal_reached_count = 0
		self._cost_total = 0.0
		self._cost_hazards_total = 0.0
		self._cost_vases_contact_total = 0.0
		self._cost_vases_velocity_total = 0.0
		# Steps where the agent's center sits inside *any* hazard radius.
		# Different from cost_hazards (which is a continuous distance penalty).
		self._in_hazard_steps = 0
		# Updated each step; the trainer reads it at episode end.
		self._last_dist_goal = float('nan')
		# Sum of UNSHAPED rewards over the episode. Lets us compare runs with
		# different cost_lambda on the same y-axis.
		self._raw_reward_total = 0.0

	def reset(self):
		self._goal_reached_count = 0
		self._cost_total = 0.0
		self._cost_hazards_total = 0.0
		self._cost_vases_contact_total = 0.0
		self._cost_vases_velocity_total = 0.0
		self._in_hazard_steps = 0
		self._last_dist_goal = float('nan')
		self._raw_reward_total = 0.0
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

		# Optional reward shaping: reward <- reward - lambda * cost.
		# lambda=0 (default) recovers the vanilla baseline. Non-zero values
		# are the control experiment for CILD's motivation -- showing that
		# naive scalar shaping is not a structural fix for cost-aware planning.
		raw_reward = float(reward)
		self._raw_reward_total += raw_reward
		cost_lambda = float(getattr(self.cfg, 'cost_lambda', 0.0))
		if cost_lambda != 0.0:
			reward = raw_reward - cost_lambda * float(cost)
		else:
			reward = raw_reward

		task = self.env.unwrapped.task

		# Whether the agent reached the goal this step. The underlying builder
		# sets info['goal_met']=True at the moment of achievement, then
		# immediately respawns the goal — so reading task.goal_achieved AFTER
		# env.step() has already returned is too late (always False). Use the
		# info flag the builder leaves behind.
		goal_reached = bool(info.get('goal_met', False))
		if goal_reached:
			self._goal_reached_count += 1
		self._cost_total += float(cost)

		# Decompose cost by source. Different tasks expose different keys:
		#   PointGoal1 / CarGoal1 → only cost_hazards
		#   PointGoal2            → cost_hazards + cost_vases_contact + cost_vases_velocity
		# Use .get with default 0.0 so missing keys are treated as zero.
		c_hazards = float(info.get('cost_hazards', 0.0))
		c_vases_contact = float(info.get('cost_vases_contact', 0.0))
		c_vases_velocity = float(info.get('cost_vases_velocity', 0.0))
		self._cost_hazards_total += c_hazards
		self._cost_vases_contact_total += c_vases_contact
		self._cost_vases_velocity_total += c_vases_velocity

		# Whether the agent center is inside *any* hazard radius this step.
		# This is a binary event count, complementary to cost_hazards (which
		# weights by how deep into the hazard the agent is).
		if c_hazards > 0.0:
			self._in_hazard_steps += 1

		# Distance to the (possibly respawned) goal at the end of this step.
		# `dist_goal()` is a method on the task object.
		# Note: Goal worlds are open (no walls), so an unconstrained random
		# policy can drift far from the placement region. Don't be alarmed
		# if early-training final_goal_distance is in the tens of meters.
		try:
			self._last_dist_goal = float(task.dist_goal())
		except Exception:
			self._last_dist_goal = float('nan')

		# Fields TensorWrapper requires:
		#   info['success']    — per-step bool/float
		#   info['terminated'] — distinguishes natural termination vs truncation
		# Plus CILD-relevant signals.
		info = dict(info) if info else {}
		info['success'] = float(goal_reached)
		info['terminated'] = bool(terminated)
		info['raw_reward'] = raw_reward
		info['raw_reward_total'] = float(self._raw_reward_total)
		info['cost'] = float(cost)
		info['cost_total'] = float(self._cost_total)
		info['cost_hazards_total'] = float(self._cost_hazards_total)
		info['cost_vases_contact_total'] = float(self._cost_vases_contact_total)
		info['cost_vases_velocity_total'] = float(self._cost_vases_velocity_total)
		info['in_hazard_steps'] = float(self._in_hazard_steps)
		info['goal_reached_count'] = float(self._goal_reached_count)
		info['final_goal_distance'] = float(self._last_dist_goal)
		info['collision_flag'] = float((c_hazards + c_vases_contact + c_vases_velocity) > 0.0)
		info['min_lidar_dist'] = _min_obstacle_surface_dist(self.env.unwrapped.task)
		info['goal_dist'] = float(self._last_dist_goal)

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
