from gymnasium.wrappers import TimeLimit

from envs.ugv.ugv_2d_env import UGV2DEnv


_TASKS = {
    "ugv-goal": {
        "task": "goal",
        "episode_length": 300,
        "num_obstacles": 0,
    },
}


def make_env(cfg):
    """
    Factory used by tdmpc2/envs/__init__.py.

    Supported tasks:
    - ugv-goal
    """
    task_name = str(cfg.task)

    if task_name not in _TASKS:
        raise ValueError(f"Unknown UGV task: {task_name}")

    task_cfg = _TASKS[task_name]

    env = UGV2DEnv(
        task=task_cfg["task"],
        seed=int(cfg.seed),
        episode_length=int(task_cfg["episode_length"]),
    )

    env = TimeLimit(env, max_episode_steps=int(task_cfg["episode_length"]))
    env.max_episode_steps = int(task_cfg["episode_length"])
    return env
