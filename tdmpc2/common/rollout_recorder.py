"""
Unified rollout recorder for UGV environments.
Records trajectories, videos, and episode data.
"""
import numpy as np
from pathlib import Path
import imageio
from common.trajectory_viz import save_ugv_trajectory


class RolloutRecorder:
    """Records UGV rollouts with trajectory, video, and data."""

    def __init__(self, save_dir, save_video=True, save_traj=True, save_data=True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = save_video
        self.save_traj = save_traj
        self.save_data = save_data
        self.reset()

    def reset(self):
        """Reset recorder for new episode."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.frames = []
        self.goal = None
        self.obstacles = []
        self.map_size = None

    def record_step(self, state, action, reward, frame=None, obstacle_states=None):
        """Record a single step."""
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        if frame is not None:
            self.frames.append(frame)
        if obstacle_states is not None:
            self.obstacles.append(obstacle_states.copy())

    def set_episode_info(self, goal, map_size):
        """Set episode-level information."""
        self.goal = goal
        self.map_size = map_size

    def save_episode(self, episode_idx, info):
        """
        Save episode data to disk.

        Args:
            episode_idx: Episode number
            info: Final info dict from environment
        """
        prefix = self.save_dir / f"episode_{episode_idx:03d}"

        # Save trajectory plot
        if self.save_traj and len(self.states) > 0:
            traj_path = f"{prefix}.png"
            save_ugv_trajectory(
                self.states,
                self.goal,
                self.map_size,
                traj_path,
                title=f"Episode {episode_idx} - Success: {bool(info.get('success', False))}"
            )

        # Save video
        if self.save_video and len(self.frames) > 0:
            video_path = f"{prefix}.mp4"
            imageio.mimsave(video_path, self.frames, fps=20)

        # Save data
        if self.save_data and len(self.states) > 0:
            data_path = f"{prefix}.npz"
            data = {
                "states": np.array(self.states),  # [T, 5]
                "actions": np.array(self.actions),  # [T, 2]
                "rewards": np.array(self.rewards),  # [T]
                "success": bool(info.get('success', False)),
                "collision": bool(info.get('collision', False)),
                "goal": self.goal,
                "map_size": self.map_size,
                "goal_distance": float(info.get('goal_distance', 0.0)),
                "path_length": float(info.get('path_length', 0.0)),
                "out_of_bounds": bool(info.get('out_of_bounds', False)),
            }
            if len(self.obstacles) > 0:
                data["obstacles"] = np.array(self.obstacles)  # [T, K, 5]
            np.savez_compressed(data_path, **data)

        return {
            "traj_path": f"{prefix}.png" if self.save_traj else None,
            "video_path": f"{prefix}.mp4" if self.save_video else None,
            "data_path": f"{prefix}.npz" if self.save_data else None,
        }
