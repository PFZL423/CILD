# MuSHR Nav Prototype

这是 CILD/TD-MPC2 导航任务的 MuJoCo 最小原型。当前目标不是追求完整传感器栈，而是先固定一个可运行、可测试的 point-goal navigation 任务，并提供 static-hard / dynamic-hard 对照实验。

## 当前任务

小车在封闭平面场景中从 reset 生成的起点出发，避开墙、静态障碍以及可选动态障碍，到达目标点。

当前阶段用于验证：

- MuSHR 小车 XML 能否加载和控制
- navigation reward / done / collision 是否合理
- 后续接入 TD-MPC2 前，任务语义是否清楚
- static-hard、dynamic-hard moving、dynamic-hard frozen 是否能在同一布局种子上做公平对照

## 文件结构

```text
prototypes/mujoco_mushr_nav/
  assets/
    mushr_nav_hard_template.xml # hard 任务的 MuJoCo slot 模板
    mushr_nav_static.xml        # 旧版固定静态导航场景
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

`assets/mushr_nav_hard_template.xml` 包含一个固定 MuJoCo 模型和一组可由 Python 在 reset 时配置的 slot：

- MuSHR `buddy` 小车
- 10m x 10m 边界墙
- wall / box / cylinder 静态障碍 slot
- dynamic obstacle slot
- 1 个固定目标点 `goal`
- 1 个固定总览相机 `overview`
- 小车自带 `buddy_realsense_d435i` 相机位

当前 hard 任务使用 slot 系统在 reset 时生成随机起终点、静态障碍和可达布局。dynamic-hard 在同一个静态布局上额外生成动态障碍。

实现细节：

- reset 首次会把所有未使用的静态 slot variant 移到远离场地的位置，并同步更新这些 free joint 的 `qpos0`，避免后续 `mj_resetData()` 恢复到模板默认密集位置。
- 后续 reset 只清理上一轮 active 静态障碍，并局部清理本轮会使用的 slot variants；每个 physics substep 只 pin 当前 active 静态障碍。
- 碰撞分类会扫描当前所有 MuJoCo contacts；如果同一帧同时接触 active static 和 active dynamic，`collision_type` 优先返回 `dynamic`，避免 dynamic collision 被 contact 顺序遮蔽。

注意：当前环境没有接入真实 RGB/depth observation；相机位只是资产中已存在。目前障碍感知使用轻量 2D lidar/raycast proxy。

## 任务模式

Group 5 使用三个主要实验模式：

```text
static-hard:
  dynamic_mode = none
  只使用 procedural static-hard 布局、起点和目标。
  没有动态障碍；dynamic_min_distance = inf，min_ttc = inf。

dynamic-hard moving:
  dynamic_mode = hard 或 moving
  与 static-hard 使用同一 layout_seed 生成静态布局、起点和目标。
  额外按 dynamic_seed 生成动态障碍，障碍会随仿真时间移动。

dynamic-hard frozen:
  dynamic_mode = frozen
  与 moving 模式使用同一 layout_seed 和 dynamic_seed 生成同一组动态障碍初始状态。
  障碍位置保持冻结，速度报告为 0，用于分离“额外障碍占位”和“时间变化交互风险”。
```

`dynamic-hard` 生成器会根据当前 static-hard `LayoutSpec` 放置动态障碍，当前 pattern 包括 crossing-bottleneck、intersection-crossing、same-direction-blocking、occluded-emergence 和 goal-area-interference。`hard` 是 moving 模式的兼容别名。

## 种子和对照逻辑

reset 的布局种子解析顺序是：

```text
options["layout_seed"] > cfg.layout_seed > reset(seed=...) > episode counter
```

动态障碍种子解析顺序是：

```text
options["dynamic_seed"] > cfg.dynamic_seed > current layout_seed > 0
```

默认情况下，`dynamic_seed` 会跟随当前 `layout_seed`。因此同一个 episode seed 可以配对比较：

```text
static-hard(seed=N)          -> 静态布局、起点、目标
dynamic-hard moving(seed=N)  -> 同一静态布局 + moving 动态障碍
dynamic-hard frozen(seed=N)  -> 同一静态布局 + 同一动态障碍初始状态，但冻结
```

如果需要固定静态布局但改变动态障碍采样，可以显式传入相同 `layout_seed` 和不同 `dynamic_seed`。如果需要 moving / frozen 成对比较，应保持二者的 `layout_seed` 和 `dynamic_seed` 完全一致。

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
time
layout_seed
layout_template
dynamic_mode
dynamic_seed
x
y
yaw
goal_x
goal_y
distance_to_goal
progress
success
collision
collision_type
static_collision
dynamic_collision
min_obstacle_distance
episode_min_obstacle_distance
near_miss
dynamic_min_distance
dynamic_near_miss
min_ttc
ttc_violation
path_length
episode_geodesic_distance
spl
step
```

TD-MPC2 wrapper 会额外累计并写入训练/评估日志：

```text
collision_total
static_collision_total
dynamic_collision_total
```

`tdmpc2/common/logger.py` 会将它们导出为：

```text
episode_collision
episode_static_collision
episode_dynamic_collision
```

动态模式还会为每个活动动态障碍导出 `dyn_obs_N_x/y/vx/vy`、`dyn_obs_N_pattern` 和 `dyn_obs_N_trajectory_type`。这些字段主要用于调试、统计、TTC / near-miss 指标和后续 CILD label 设计。

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

检查 moving dynamic-hard：

```bash
python prototypes/mujoco_mushr_nav/test_env.py --dynamic-mode hard --episodes 1 --max-steps 500
```

检查 frozen dynamic-hard：

```bash
python prototypes/mujoco_mushr_nav/test_env.py --dynamic-mode frozen --episodes 1 --max-steps 500
```

批量统计测试：

```bash
python prototypes/mujoco_mushr_nav/test_env_batch.py --dynamic-mode none --episodes 50 --policy random --out /tmp/mushr_static
python prototypes/mujoco_mushr_nav/test_env_batch.py --dynamic-mode hard --episodes 50 --policy random --out /tmp/mushr_dynamic_moving
python prototypes/mujoco_mushr_nav/test_env_batch.py --dynamic-mode frozen --episodes 50 --policy random --out /tmp/mushr_dynamic_frozen
```

成对公平评估：

```bash
python prototypes/mujoco_mushr_nav/eval_paired.py --split seen --episodes 100 --policy random --out /tmp/mushr_eval_seen
```

`eval_paired.py` 对每个 seed pair 依次评估 `static`、`frozen` 和 `moving`，并写出 `paired_eval.csv` 与 `summary.csv`。当前固定 split 包括 `train-smoke`、`seen`、`unseen-layout`、`unseen-dynamic` 和 `combo`。动态模式依赖 procedural `LayoutSpec`，因此 `--fixed-layout` 不能与 `--dynamic-mode hard/frozen` 同时使用。

检查 TD-MPC2 checkpoint 轨迹：

```bash
cd tdmpc2
python evaluate_mushr_traj.py \
  task=mushr-nav-dynamic-hard \
  model_size=1 \
  checkpoint=logs/mushr-nav-dynamic-hard/1/debug_dynamic_50k/models/50000.pt \
  save_video=false \
  enable_wandb=false \
  compile=false \
  +eval_split=seen \
  +traj_episodes=10 \
  +traj_max_steps=500 \
  +plot_episodes=6 \
  +out_dir=/tmp/mushr_learned_traj_50k
```

该脚本会在同一组 seed pair 上 rollout learned policy 的 `static`、`frozen` 和 `moving`，并导出 `learned_episodes.csv`、`learned_trajectories.csv`、`summary.csv` 和 `learned_trajectories.png`。

输出文件写入：

```text
prototypes/mujoco_mushr_nav/outputs/
```

该目录已在 `.gitignore` 中忽略。batch check 会写出 per-episode CSV、trajectory CSV 和 trajectory plot。trajectory CSV 包含每步位姿、动作、reward、layout_seed、TTC / near-miss 指标，以及动态障碍位置、速度、pattern 和 trajectory type；moving dynamic 中动态障碍位置/速度应随时间变化，frozen dynamic 中动态障碍位置应保持不变且速度为 0。`env.step()` 的 `info` 中同时保留 dynamic_mode 和 dynamic_seed，便于后续评估脚本导出。

## 当前限制

- 没有 RGB/depth observation
- 没有 CILD heads 或 CILD labels

当前环境的 MuJoCo 物理仿真可运行；本机 headless OpenGL 渲染不可用，因此 `test_scene.py` 默认生成 top-down 示意图，而不是 MuJoCo camera RGB/depth 图。

## 下一步

推荐推进顺序：

1. 先跑 `mushr-nav-static-hard` low-dim+lidar TD-MPC2 baseline。
2. 定义 CILD labels：
   - `progress`
   - `risk`
   - `sector occupancy`
3. 再考虑 depth image observation 和视觉 encoder。
4. 任务稳定后迁移到 Isaac Lab 重建并行训练基座。

## TD-MPC2 接入

当前已通过 `tdmpc2/envs/mushr_nav.py` 接入 TD-MPC2 的 `make_env` 流程，任务名：

```text
mushr-nav-static-hard
mushr-nav-dynamic-hard
mushr-nav-dynamic-frozen
```

短训练建议先关闭视频、wandb 和 compile：

```bash
cd /home/ubuntu/self_project/tdmpc2/tdmpc2
conda activate tdmpc2
python train.py task=mushr-nav-static-hard model_size=1 steps=5000 seed_steps=1000 eval_freq=2500 save_video=false enable_wandb=false compile=false
```

该任务会提前终止，因此适配层会设置：

```text
cfg.episodic = True
```
