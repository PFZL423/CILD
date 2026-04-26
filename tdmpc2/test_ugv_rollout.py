"""
Rollout UGV environment with random actions and visualize trajectory.
"""
import numpy as np
from omegaconf import OmegaConf

from common.trajectory_viz import save_ugv_trajectory


def plot_trajectory(states, goal, map_size, save_path="trajectory.png"):
    save_ugv_trajectory(states, goal, map_size, save_path)
    print(f"Saved trajectory plot to {save_path}")


def rollout_random(num_episodes=3, save_dir="./"):
    """Rollout random policy and save trajectories."""
    cfg = OmegaConf.create({
        "task": "ugv-goal",
        "seed": 42,
        "multitask": False,
        "obs": "state",
    })

    # Create unwrapped env to access internal state
    from envs.ugv import make_env as make_ugv_env
    env_unwrapped = make_ugv_env(cfg)

    print(f"Running {num_episodes} random rollouts...")

    for ep in range(num_episodes):
        obs, info = env_unwrapped.reset()
        done = False
        states = []

        print(f"\nEpisode {ep + 1}:")
        print(f"  Start: x={info['x']:.2f}, y={info['y']:.2f}, theta={info['theta']:.2f}")
        print(f"  Goal distance: {info['goal_distance']:.2f}")

        step = 0
        while not done:
            # Random action
            action = env_unwrapped.action_space.sample()

            # Store state
            states.append(env_unwrapped.robot_state.copy())

            # Step
            obs, reward, terminated, truncated, info = env_unwrapped.step(action)
            done = terminated or truncated
            step += 1

            if step >= 300:  # Safety limit
                break

        # Final state
        states.append(env_unwrapped.robot_state.copy())

        print(f"  End: x={info['x']:.2f}, y={info['y']:.2f}, theta={info['theta']:.2f}")
        print(f"  Final goal distance: {info['goal_distance']:.2f}")
        print(f"  Path length: {info['path_length']:.2f}")
        print(f"  Success: {bool(info['success'])}")
        print(f"  Out of bounds: {info.get('out_of_bounds', False)}")
        print(f"  Steps: {step}")

        # Plot
        save_path = f"{save_dir}/trajectory_ep{ep + 1}.png"
        plot_trajectory(
            states,
            env_unwrapped.goal,
            env_unwrapped.map_size,
            save_path
        )


if __name__ == "__main__":
    import sys
    num_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "./"

    rollout_random(num_episodes=num_eps, save_dir=save_dir)
    print("\nDone!")
