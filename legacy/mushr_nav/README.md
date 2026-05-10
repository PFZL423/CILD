# mushr_nav (legacy)

这是一个早期阶段使用的**自定义 MuSHR 导航环境**，基于 MuJoCo + 32-ray lidar + 程序化场景生成。

## 当前状态

**已停用**。Motivation 实验路线已切换到公开 benchmark（Safety Gymnasium），原因见 `../../RESEARCH_NOTES.md`（2026-05-06 认知突破日）。

核心结论：自造环境不能用作 motivation（baseline 表现是观测对象，不是可控变量），但可以作为后续阶段的"复杂场景泛化"展示。

## 文件来源

```
prototypes/mujoco_mushr_nav/      → legacy/mushr_nav/prototype/
tdmpc2/envs/mushr_nav.py          → legacy/mushr_nav/env_wrapper.py
tdmpc2/evaluate_mushr_traj.py     → legacy/mushr_nav/evaluate_traj.py
```

## 何时重新启用

**实验3（泛化展示）阶段**：在 Safety Gymnasium 上完成 motivation + CILD 方法验证后，把 CILD 也跑在 mushr-nav 上，展示方法在更复杂、更接近 sim2real 的环境也 work。

那时需要：
1. 把 `env_wrapper.py` 的 import 路径恢复（`prototypes.mujoco_mushr_nav.env` → `legacy.mushr_nav.prototype.env`）
2. 重新接入 `tdmpc2/envs/__init__.py`
3. 注意：原 wrapper 包含一组为 mushr 调出来的 cfg 覆盖（discount/rho/reward_coef/vmin/vmax 等），重启用时需要重新评估这些是否还合理

## 不要做的事

- ❌ 把 mushr 调参经验"应用"到 Safety Gymnasium baseline — benchmark 必须用默认 cfg
- ❌ 重新陷入 reward 工程 — 自造环境的所有调参问题不要带到 benchmark 路线
