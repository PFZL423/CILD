import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.buffer_vec import VecBuffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from trainer.vec_online_trainer import VecOnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=SafetyPointGoal1-v0 steps=500000
		$ python train.py task=SafetyPointGoal2-v0 steps=500000 seed=2
		$ python train.py task=SafetyCarGoal1-v0 model_size=5
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	env = make_env(cfg)
	num_envs = int(getattr(cfg, 'num_envs', 1))
	if num_envs > 1:
		trainer_cls = VecOnlineTrainer
		buffer = VecBuffer(cfg)
	elif cfg.multitask:
		trainer_cls = OfflineTrainer
		buffer = Buffer(cfg)
	else:
		trainer_cls = OnlineTrainer
		buffer = Buffer(cfg)
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=TDMPC2(cfg),
		buffer=buffer,
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
