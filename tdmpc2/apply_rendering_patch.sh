#!/bin/bash
# Apply rendering updates to UGV2DEnv
# Run as root inside Docker

set -e

FILE="envs/ugv/ugv_2d_env.py"

echo "Updating $FILE..."

# Backup
cp "$FILE" "${FILE}.backup"

# Create Python script to do the modifications
python3 << 'PYTHON_SCRIPT'
import re

file_path = "envs/ugv/ugv_2d_env.py"

with open(file_path, 'r') as f:
    content = f.read()

# 1. Add imports at top (after existing imports)
if "from common.ugv_rendering import UGVRenderer" not in content:
    # Find the last import line
    import_pattern = r'(from envs\.ugv\.dynamics import.*\n)'
    content = re.sub(
        import_pattern,
        r'\1from common.ugv_rendering import UGVRenderer\n',
        content
    )

# 2. Add trajectory and renderer to __init__
if "self.trajectory = []" not in content:
    # Find after self.path_length = 0.0
    init_pattern = r'(self\.path_length = 0\.0\n)'
    content = re.sub(
        init_pattern,
        r'\1        self.trajectory = []\n\n        # Renderer\n        self._renderer = UGVRenderer(\n            map_size=self.map_size,\n            robot_radius=self.robot_radius,\n            goal_radius=self.goal_radius\n        )\n',
        content
    )

# 3. Initialize trajectory in reset()
if "self.trajectory = [self.robot_state[:2].copy()]" not in content:
    reset_pattern = r'(self\.prev_goal_dist = self\._goal_distance\(\)\n)'
    content = re.sub(
        reset_pattern,
        r'\1\n        self.trajectory = [self.robot_state[:2].copy()]\n',
        content
    )

# 4. Append to trajectory in step()
if "self.trajectory.append" not in content:
    step_pattern = r'(self\.step_count \+= 1\n)'
    content = re.sub(
        step_pattern,
        r'\1        self.trajectory.append(self.robot_state[:2].copy())\n',
        content
    )

# 5. Replace render() method
old_render = r'    def render\(self\):.*?return img'
new_render = '''    def render(self):
        """
        Render the environment using the unified renderer.
        Returns RGB array for video recording.
        """
        return self._renderer.render(
            robot_state=self.robot_state,
            goal=self.goal,
            trajectory=self.trajectory,
            obstacles=getattr(self, "obstacles", None),
            map_size=self.map_size,
            robot_radius=self.robot_radius,
            goal_radius=self.goal_radius,
            step_count=self.step_count,
        )'''

content = re.sub(old_render, new_render, content, flags=re.DOTALL)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✓ Updated successfully")
PYTHON_SCRIPT

echo "Done! Backup saved to ${FILE}.backup"
echo "Test with: python tools/record_ugv_rollout.py +episodes=1 checkpoint=null"
