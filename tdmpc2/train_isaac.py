import argparse
import sys
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description='Train TD-MPC2 with IsaacLab vectorized environments.')
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--num_envs', type=int, default=None)
parser.add_argument('--steps', type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import hydra

from common.buffer_vec import VecBuffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.vec_online_trainer import VecOnlineTrainer


@hydra.main(config_name='config_isaac', config_path='.')
def train(cfg: dict):
	"""Training entrypoint for IsaacLab vectorized environments."""
	if args_cli.task is not None:
		cfg.task = args_cli.task
	if args_cli.num_envs is not None:
		cfg.num_envs = args_cli.num_envs
	if args_cli.steps is not None:
		cfg.steps = args_cli.steps
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	buffer = VecBuffer(cfg)
	logger = Logger(cfg)
	trainer = VecOnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger)
	try:
		trainer.train()
	finally:
		env.close()


if __name__ == '__main__':
	try:
		train()
	finally:
		simulation_app.close()
