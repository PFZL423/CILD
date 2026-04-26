"""
Record UGV rollouts with trained agent or random policy.

Usage:
    # Random policy
    python tools/record_ugv_rollout.py +episodes=5

    # Trained agent
    python tools/record_ugv_rollout.py \
        +checkpoint=logs/ugv-goal/1/default/model.pt \
        +episodes=10 \
        +save_video=true \
        +save_traj=true
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from pathlib import Path

from common.rollout_recorder import RolloutRecorder
from envs import make_env


@hydra.main(config_name='config', config_path='..', version_base='1.1')
def main(cfg):
    """Record UGV rollouts."""
    print("\n" + "="*60)
    print("TD-MPC2 UGV Rollout Recorder")
    print("="*60 + "\n")
    # Allow new keys
    OmegaConf.set_struct(cfg, False)

    # Handle mandatory values from config
    if cfg.get('checkpoint') == '???':
        cfg.checkpoint = None

    # Set defaults
    if not hasattr(cfg, 'episodes'):
        cfg.episodes = 5
    if cfg.checkpoint is None or cfg.checkpoint == '???':
        cfg.checkpoint = None
    if not hasattr(cfg, 'save_video'):
        cfg.save_video = True
    if not hasattr(cfg, 'save_traj'):
        cfg.save_traj = True
    if not hasattr(cfg, 'save_data'):
        cfg.save_data = True

    # Override task if not specified
    if cfg.task == 'dog-run':  # Default from config.yaml
        cfg.task = 'ugv-goal'

    cfg.multitask = False
    cfg.obs = 'state'

    # Create environment (unwrapped to access internal state)
    from envs.ugv import make_env as make_ugv_env
    env_unwrapped = make_ugv_env(cfg)

    # Load agent if checkpoint provided
    agent = None
    if cfg.checkpoint:
        print(f"Loading checkpoint: {cfg.checkpoint}")
        from tdmpc2 import TDMPC2
        from common.parser import parse_cfg
        cfg = parse_cfg(cfg)
        agent = TDMPC2(cfg)
        agent.load(cfg.checkpoint)
        agent.eval()
        print("Agent loaded successfully")
    else:
        print("Using random policy")

    # Setup recorder
    if cfg.checkpoint:
        save_dir = Path(cfg.checkpoint).parent / 'rollouts'
    else:
        save_dir = Path(f'logs/{cfg.task}/random_rollouts')

    recorder = RolloutRecorder(
        save_dir=save_dir,
        save_video=cfg.save_video,
        save_traj=cfg.save_traj,
        save_data=cfg.save_data
    )

    print(f"\nRecording {cfg.episodes} episodes to {save_dir}")
    print(f"  save_video: {cfg.save_video}")
    print(f"  save_traj: {cfg.save_traj}")
    print(f"  save_data: {cfg.save_data}\n")

    # Record episodes
    episode_stats = []

    for ep in range(cfg.episodes):
        obs, info = env_unwrapped.reset()
        done = False
        recorder.reset()
        recorder.set_episode_info(
            goal=env_unwrapped.goal,
            map_size=env_unwrapped.map_size
        )

        print(f"Episode {ep + 1}/{cfg.episodes}:")
        print(f"  Start: x={info['x']:.2f}, y={info['y']:.2f}, theta={info['theta']:.2f}")
        print(f"  Goal: x={env_unwrapped.goal[0]:.2f}, y={env_unwrapped.goal[1]:.2f}")
        print(f"  Initial distance: {info['goal_distance']:.2f}")

        step = 0
        ep_reward = 0.0

        while not done:
            # Get action
            if agent is not None:
                # Use trained agent
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action = agent.act(obs_tensor, t0=(step==0), eval_mode=True)
                action = action.cpu().numpy()
            else:
                # Random action
                action = env_unwrapped.action_space.sample()

            # Render frame if saving video
            frame = None
            if cfg.save_video:
                frame = env_unwrapped.render()

            # Record step
            recorder.record_step(
                state=env_unwrapped.robot_state,
                action=action,
                reward=0.0,  # Will be filled in next step
                frame=frame,
                obstacle_states=getattr(env_unwrapped, 'obstacles', None)
            )

            # Step environment
            obs, reward, terminated, truncated, info = env_unwrapped.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            # Update last reward
            if len(recorder.rewards) > 0:
                recorder.rewards[-1] = reward

            if step >= 300:
                break

        # Save episode
        paths = recorder.save_episode(ep, info)

        # Print stats
        print(f"  End: x={info['x']:.2f}, y={info['y']:.2f}, theta={info['theta']:.2f}")
        print(f"  Final distance: {info['goal_distance']:.2f}")
        print(f"  Path length: {info['path_length']:.2f}")
        print(f"  Total reward: {ep_reward:.2f}")
        print(f"  Success: {bool(info['success'])}")
        print(f"  Collision: {bool(info.get('collision', False))}")
        print(f"  Out of bounds: {info.get('out_of_bounds', False)}")
        print(f"  Steps: {step}")
        if cfg.save_traj:
            print(f"  Saved: {paths['traj_path']}")
        if cfg.save_video:
            print(f"  Saved: {paths['video_path']}")
        if cfg.save_data:
            print(f"  Saved: {paths['data_path']}")
        print()

        episode_stats.append({
            'success': bool(info['success']),
            'collision': bool(info.get('collision', False)),
            'goal_distance': float(info['goal_distance']),
            'path_length': float(info['path_length']),
            'reward': ep_reward,
            'steps': step,
        })

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    success_rate = np.mean([s['success'] for s in episode_stats])
    collision_rate = np.mean([s['collision'] for s in episode_stats])
    avg_distance = np.mean([s['goal_distance'] for s in episode_stats])
    avg_path = np.mean([s['path_length'] for s in episode_stats])
    avg_reward = np.mean([s['reward'] for s in episode_stats])
    avg_steps = np.mean([s['steps'] for s in episode_stats])

    print(f"Success rate:     {success_rate*100:.1f}%")
    print(f"Collision rate:   {collision_rate*100:.1f}%")
    print(f"Avg goal dist:    {avg_distance:.2f} m")
    print(f"Avg path length:  {avg_path:.2f} m")
    print(f"Avg reward:       {avg_reward:.2f}")
    print(f"Avg steps:        {avg_steps:.1f}")
    print(f"\nAll files saved to: {save_dir}")


if __name__ == "__main__":
    main()
