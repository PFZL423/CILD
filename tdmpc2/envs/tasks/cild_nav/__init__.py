import gymnasium as gym

from .cild_nav_env import CILDNavEnv
from .cild_nav_cfg import CILDNavEnvCfg

gym.register(
    id="Isaac-CILDNav-v0",
    entry_point=CILDNavEnv,
    disable_env_checker=True,
    kwargs={"cfg": CILDNavEnvCfg()},
)
