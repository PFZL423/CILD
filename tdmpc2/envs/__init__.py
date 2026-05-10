from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.safety_gym import make_env as make_safety_gym_env
	from envs.safety_gym_vec import SafetyGymVecEnv
except:
	make_safety_gym_env = missing_dependencies
	SafetyGymVecEnv = None

try:
	from envs.isaaclab import make_env as make_isaaclab_env
except:
	make_isaaclab_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env


def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	try:
		gym.logger.set_level(40)
	except AttributeError:
		pass  # newer gymnasium dropped this API; silencing its logger is cosmetic
	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		if cfg.task.startswith('Isaac-'):
			env = make_isaaclab_env(cfg)
			cfg.obs_shape = {'state': env.observation_space.shape}
			cfg.action_dim = env.action_space.shape[0]
			cfg.episode_length = env.max_episode_steps
			cfg.seed_steps = max(1000, 5 * cfg.episode_length)
			cfg.num_envs = env.num_envs
			return env
		num_envs = int(getattr(cfg, 'num_envs', 1))
		if num_envs > 1:
			if SafetyGymVecEnv is None:
				raise RuntimeError('SafetyGymVecEnv could not be imported; check safety-gymnasium installation.')
			env = SafetyGymVecEnv(cfg, device='cuda:0')
			cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
			cfg.action_dim = env.action_space.shape[0]
			cfg.episode_length = env.max_episode_steps
			cfg.seed_steps = max(1000, 5 * cfg.episode_length)
			cfg.num_envs = env.num_envs
			return env
		env = None
		for fn in [make_safety_gym_env]:
			try:
				env = fn(cfg)
			except ValueError:
				pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5 * cfg.episode_length)
	return env
