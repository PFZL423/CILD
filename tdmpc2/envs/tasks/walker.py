import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import walker
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_YOGA_STAND_HEIGHT = 1.0
_YOGA_LIE_DOWN_HEIGHT = 0.08
_YOGA_LEGS_UP_HEIGHT = 1.1
_NUM_OBSTACLES = 5
_OBSTACLE_SPACING = 3.0
_OBSTACLE_START = 3.0
_OBSTACLE_X_JITTER = 0.5   # 每episode随机偏移范围
_OBSTACLE_HEIGHT_MIN = 0.14
_OBSTACLE_HEIGHT_MAX = 0.18
_OBSTACLE_HEIGHT = 0.12  # xml初始值，运行时被随机覆盖
_OBSTACLE_CLEARANCE = 0.22
_OBSTACLE_COLLISION = 0.16
_DYNAMIC_AMP_MIN = 0.3
_DYNAMIC_AMP_MAX = 0.8
_DYNAMIC_SPEED_MIN = 0.04
_DYNAMIC_SPEED_MAX = 0.10


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'walker.xml')), common.ASSETS


@walker.SUITE.add('custom')
def walk_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def run_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def walk_static_obstacle(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task with a static hurdle obstacle."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = ObstacleWalker(move_speed=walker._WALK_SPEED, dynamic=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def walk_dynamic_obstacle(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task with a moving hurdle obstacle."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = ObstacleWalker(move_speed=walker._WALK_SPEED, dynamic=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def arabesque(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Arabesque task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='arabesque', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def lie_down(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Lie Down task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='lie_down', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def legs_up(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Legs Up task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='legs_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def headstand(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Headstand task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def flip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def backflip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Backflip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=-walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


class BackwardsPlanarWalker(walker.PlanarWalker):
    """Backwards PlanarWalker task."""
    def __init__(self, move_speed, random=None):
        super().__init__(move_speed, random)
    
    def get_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(walker._STAND_HEIGHT, float('inf')),
                                 margin=walker._STAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(-float('inf'), -self._move_speed),
                                            margin=self._move_speed/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5*move_reward + 1) / 6


class ObstacleWalker(walker.PlanarWalker):
    """Planar walker with multiple hurdle obstacles for static/dynamic avoidance tests."""

    def __init__(self, move_speed, dynamic=False, random=None):
        super().__init__(move_speed, random)
        self._dynamic = dynamic
        self._n = _NUM_OBSTACLES
        self._centers = np.zeros(self._n)
        self._heights = np.full(self._n, _OBSTACLE_HEIGHT)
        self._phases = np.zeros(self._n)
        self._amplitudes = np.zeros(self._n)
        self._speeds = np.zeros(self._n)
        self._velocities = np.zeros(self._n)
        self._has_collision = False

    def _obs_name(self, i):
        return f'obstacle{i}'

    def _set_obstacle_x(self, physics, i, x):
        physics.named.model.geom_pos[self._obs_name(i), 'x'] = x

    def _obstacle_x(self, physics, i):
        return float(physics.named.data.geom_xpos[self._obs_name(i), 'x'])

    def _walker_points(self, physics):
        return [
            np.array([physics.named.data.xpos['torso', 'x'], physics.named.data.xpos['torso', 'z']], dtype=np.float32),
            np.array([physics.named.data.xpos['left_foot', 'x'], physics.named.data.xpos['left_foot', 'z']], dtype=np.float32),
            np.array([physics.named.data.xpos['right_foot', 'x'], physics.named.data.xpos['right_foot', 'z']], dtype=np.float32),
        ]

    def _obstacle_distance(self, physics, i):
        center = np.array([self._obstacle_x(physics, i), self._heights[i]], dtype=np.float32)
        return min(np.linalg.norm(p - center) for p in self._walker_points(physics))

    def _min_obstacle_distance(self, physics):
        return min(self._obstacle_distance(physics, i) for i in range(self._n))

    def _obstacle_clearance_reward(self, physics):
        d = self._min_obstacle_distance(physics)
        return rewards.tolerance(
            d,
            bounds=(_OBSTACLE_CLEARANCE, float('inf')),
            margin=_OBSTACLE_CLEARANCE,
            value_at_margin=0.0,
            sigmoid='linear')

    def _collision(self, physics):
        return self._min_obstacle_distance(physics) < _OBSTACLE_COLLISION

    def _obstacles_passed(self, physics):
        torso_x = physics.named.data.xpos['torso', 'x']
        return sum(1 for i in range(self._n) if torso_x > self._centers[i] + 0.25)

    def initialize_episode(self, physics):
        for i in range(self._n):
            base = _OBSTACLE_START + i * _OBSTACLE_SPACING
            jitter = self.random.uniform(-_OBSTACLE_X_JITTER, _OBSTACLE_X_JITTER)
            self._centers[i] = base + jitter
            self._heights[i] = self.random.uniform(_OBSTACLE_HEIGHT_MIN, _OBSTACLE_HEIGHT_MAX)
            self._phases[i] = self.random.uniform(0.0, 2 * np.pi)
            self._amplitudes[i] = self.random.uniform(_DYNAMIC_AMP_MIN, _DYNAMIC_AMP_MAX)
            self._speeds[i] = self.random.uniform(_DYNAMIC_SPEED_MIN, _DYNAMIC_SPEED_MAX)
            self._velocities[i] = 0.0
            self._set_obstacle_x(physics, i, self._centers[i])
            physics.named.model.geom_size[self._obs_name(i), 2] = self._heights[i]
            physics.named.model.geom_pos[self._obs_name(i), 'z'] = self._heights[i]
        self._has_collision = False
        self._collision_count = 0
        self._max_passed = 0
        super().initialize_episode(physics)

    def before_step(self, action, physics):
        if self._dynamic:
            for i in range(self._n):
                self._phases[i] += self._speeds[i]
                x = self._centers[i] + self._amplitudes[i] * np.sin(self._phases[i])
                self._velocities[i] = self._amplitudes[i] * self._speeds[i] * np.cos(self._phases[i])
                self._set_obstacle_x(physics, i, x)
        else:
            self._velocities[:] = 0.0
        super().before_step(action, physics)

    def after_step(self, physics):
        if self._collision(physics):
            self._has_collision = True
            self._collision_count += 1
        self._max_passed = max(self._max_passed, self._obstacles_passed(physics))
        super().after_step(physics)

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['obstacle_state'] = np.array(
            [[self._obstacle_x(physics, i), self._heights[i], self._velocities[i]]
             for i in range(self._n)],
            dtype=np.float32).flatten()
        return obs

    def get_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(walker._STAND_HEIGHT, float('inf')),
                                 margin=walker._STAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                        bounds=(self._move_speed, float('inf')),
                                        margin=self._move_speed/2,
                                        value_at_margin=0,
                                        sigmoid='linear')
        base_reward = stand_reward * (5*move_reward + 1) / 6
        clearance_reward = self._obstacle_clearance_reward(physics)
        return base_reward * (0.2 + 0.8*clearance_reward)

    def get_info(self, physics):
        return {
            'success': float(self._max_passed >= 4 and not self._has_collision),
            'collision': float(self._collision(physics)),
            'collision_total': float(self._collision_count),
            'obstacles_passed': float(self._max_passed),
            'terminated': False,
        }


class YogaPlanarWalker(walker.PlanarWalker):
    """Yoga PlanarWalker tasks."""
    
    def __init__(self, goal='arabesque', move_speed=0, random=None):
        super().__init__(0, random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _arabesque_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        left_foot_down = rewards.tolerance(left_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_STAND_HEIGHT/2)
        right_foot_up = rewards.tolerance(right_foot_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        arabesque_reward = (3*standing + left_foot_down + right_foot_up + upright) / 6
        return arabesque_reward
    
    def _lie_down_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        lie_down_reward = (3*torso_down + thigh_down + upright) / 5
        return lie_down_reward
    
    def _legs_up_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        legs_up_reward = (3*torso_down + 2*legs_up + thigh_down + upright) / 7
        return legs_up_reward
    
    def _flip_reward(self, physics):
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_up = rewards.tolerance(thigh_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upside_down_reward = (3*legs_up + 2*thigh_up) / 5
        if self._move_speed == 0:
            return upside_down_reward
        move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._move_speed, float('inf')) if self._move_speed > 0 else (-float('inf'), self._move_speed),
                                    margin=abs(self._move_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return upside_down_reward * (5*move_reward + 1) / 6
    
    def get_reward(self, physics):
        if self._goal == 'arabesque':
            return self._arabesque_reward(physics)
        elif self._goal == 'lie_down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs_up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')


if __name__ == '__main__':
    env = legs_up()
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(6))
