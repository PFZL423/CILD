# MuSHR Nav Prototype

这是 CILD/TD-MPC2 导航任务的 MuJoCo 最小原型。当前目标不是追求完整传感器栈，而是先固定一个可运行、可测试的静态障碍 point-goal navigation 任务。

## 当前任务

小车从固定起点出发，在封闭平面场景中避开墙和静态障碍，到达固定目标点。

当前阶段用于验证：

- MuSHR 小车 XML 能否加载和控制
- navigation reward / done / collision 是否合理
- 后续接入 TD-MPC2 前，任务语义是否清楚

## 文件结构

```text
prototypes/mujoco_mushr_nav/
  assets/
    mushr_nav_static.xml        # 当前静态导航场景
    cars/base_car/buddy.xml     # 从 MuSHR 拷贝的最小车模
    cars/meshes/*.stl           # 车模 mesh
    MUSHR_LICENSE.md            # MuSHR BSD-3-Clause license
  env.py                        # Gymnasium 风格 MuJoCo 环境
  test_scene.py                 # XML/场景布局检查
  test_env.py                   # 单 episode 环境闭环检查
  test_env_batch.py             # 多 episode 统计测试
tdmpc2/envs/mushr_nav.py        # TD-MPC2 环境适配层
```

`third_party/mushr_mujoco_ros/` 是本地外部仓库，已在 `.gitignore` 中忽略，不应直接提交。

## 场景

`assets/mushr_nav_static.xml` 包含：

- MuSHR `buddy` 小车
- 10m x 10m 边界墙
- 3 个内部墙
- 4 个静态 box/cylinder 障碍物
- 1 个固定目标点 `goal`
- 1 个固定总览相机 `overview`
- 小车自带 `buddy_realsense_d435i` 相机位

注意：当前环境没有接入真实 RGB/depth observation；相机位只是资产中已存在。目前障碍感知使用轻量 2D lidar/raycast proxy。

## Action Space

当前动作空间为 2 维连续动作：

```text
action[0] = steering command
action[1] = throttle command
```

Gym space:

```text
Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
```

内部映射：

```text
steering = action[0] * max_steering
throttle = action[1] * max_throttle
```

默认参数：

```text
max_steering = 0.38 rad
max_throttle = 5.0
```

对应 MuJoCo actuator：

```text
buddy_steering_pos
buddy_throttle_velocity
```

## Observation Space

当前 observation 是低维向量，不包含全局 `x, y, yaw`。

```text
obs[0] = ego_vx          车体坐标系 x 方向速度
obs[1] = ego_vy          车体坐标系 y 方向速度
obs[2] = yaw_rate        车体 yaw 角速度
obs[3] = steering_angle  当前转向关节角
obs[4] = goal_dx_body    目标在车体坐标系下的 x
obs[5] = goal_dy_body    目标在车体坐标系下的 y
obs[6] = goal_distance   到目标的欧氏距离
obs[7] = goal_angle      目标相对车头方向角
obs[8:] = lidar_ranges   车体坐标系下的归一化 2D lidar 距离
```

Gym space:

```text
Box(low=-inf, high=inf, shape=(8 + lidar_num_rays,), dtype=float32)
```

默认 lidar 参数：

```text
lidar_num_rays = 32
lidar_range = 5.0 m
lidar_fov = 360 deg
```

`lidar_ranges` 归一化到 `[0, 1]`：

```text
1.0 表示最大探测距离内没有障碍
0.2 表示约 1m 处存在障碍，默认 range=5m
```

全局位姿只放在 `info` 中用于 debug、画图、reward 和指标统计，不直接给 agent。

## Reward

当前 reward 定义在 `env.py` 中：

```text
reward =
  progress_scale * progress
  - time_penalty
  - control_penalty * ||action||^2
  - near_obstacle_penalty * proximity_cost
  + success_bonus if success
  - collision_penalty if collision
```

其中：

```text
progress = previous_distance_to_goal - current_distance_to_goal
```

默认参数：

```text
progress_scale = 1.0
time_penalty = 0.01
control_penalty = 0.01
near_obstacle_penalty = 0.2
near_obstacle_margin = 0.8
success_bonus = 10.0
collision_penalty = 10.0
```

## Termination

```text
success:
  distance_to_goal <= goal_radius

collision:
  MuJoCo contact between car subtree and wall/obstacle geoms

timeout:
  step_count >= max_episode_steps
```

默认：

```text
goal_radius = 0.35
max_episode_steps = 500
```

## Info Fields

`env.step()` 的 `info` 当前包含：

```text
x
y
yaw
distance_to_goal
progress
success
collision
min_obstacle_distance
step
```

这些字段主要用于调试、统计和后续 CILD label 设计。

## 运行方式

从 repo 根目录运行：

```bash
cd /home/ubuntu/self_project/tdmpc2
conda activate tdmpc2
```

检查 XML 和场景布局：

```bash
python prototypes/mujoco_mushr_nav/test_scene.py
```

检查单 episode 环境闭环：

```bash
python prototypes/mujoco_mushr_nav/test_env.py --episodes 1 --max-steps 500
```

批量统计测试：

```bash
python prototypes/mujoco_mushr_nav/test_env_batch.py --episodes 50 --policy random
python prototypes/mujoco_mushr_nav/test_env_batch.py --episodes 50 --policy heading
```

输出文件写入：

```text
prototypes/mujoco_mushr_nav/outputs/
```

该目录已在 `.gitignore` 中忽略。

## 当前限制

- 固定起点、固定目标、固定静态障碍
- 没有随机化 reset
- 没有动态障碍
- 没有 RGB/depth observation
- 没有 CILD heads 或 CILD labels

当前环境的 MuJoCo 物理仿真可运行；本机 headless OpenGL 渲染不可用，因此 `test_scene.py` 默认生成 top-down 示意图，而不是 MuJoCo camera RGB/depth 图。

## 下一步

推荐推进顺序：

1. 先跑 `mushr-nav-static` low-dim+lidar TD-MPC2 baseline。
2. 定义 CILD labels：
   - `progress`
   - `risk`
   - `sector occupancy`
3. 再考虑 depth image observation 和视觉 encoder。
4. 任务稳定后迁移到 Isaac Lab 重建并行训练基座。

## TD-MPC2 接入

当前已通过 `tdmpc2/envs/mushr_nav.py` 接入 TD-MPC2 的 `make_env` 流程，任务名：

```text
mushr-nav-static
```

短训练建议先关闭视频、wandb 和 compile：

```bash
cd /home/ubuntu/self_project/tdmpc2/tdmpc2
conda activate tdmpc2
python train.py task=mushr-nav-static model_size=1 steps=5000 seed_steps=1000 eval_freq=2500 save_video=false enable_wandb=false compile=false
```

该任务会提前终止，因此适配层会设置：

```text
cfg.episodic = True
```
