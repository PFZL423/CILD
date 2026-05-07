# CILD 研究笔记

## 2026-05-06 认知突破日

---

## 核心反思：我之前的错误研究逻辑

### ❌ 错误的逻辑链
```
目标：动态避障（CILD）
   ↓
论证：动态比静态难（motivation）
   ↓
策略：让静态学得好，让动态学不好
   ↓
矛盾陷阱：
   - 静态调不好 → "我的环境/reward有bug"（焦虑）
   - 静态调太好 → "动态会不会也行了？"（焦虑）
   ↓
结果：陷入永无止境的 reward 工程
```

**根本错误**：把 baseline 表现当成"自变量"（可控制的），实际上它应该是"因变量"（只能观测的）。

---

### ✅ 正确的逻辑链
```
目标：开发新的 latent dynamics 方法（CILD）
   ↓
观察事实：现有 latent world model 在 navigation/avoidance 上能力受限
   ↓
诚实记录：在 X/Y/Z 三个环境上，TD-MPC2 表现是 A/B/C
        （不论 A/B/C 高低，都是数据）
   ↓
提出方法：CILD 通过 cost-informed latent dynamics 改进
   ↓
对比验证：在同样环境上 CILD 表现是 A'/B'/C'，比 baseline 好
   ↓
结论：CILD 是 latent dynamics 在 navigation 的更好选择
```

---

## 关键认知原则（要永远记住）

### 1. baseline 表现是观测对象，不是可调参数
不要为了让 motivation 成立而控制 baseline 表现。
你的工作是：诚实测量 baseline → 提出方法 → 展示相对提升。

### 2. "baseline 跑不好"本身就是数据
| 错误解读 | 正确解读 |
|---------|---------|
| "TD-MPC2 静态学不会 → 我的环境有bug" | "TD-MPC2 在这类任务上能力受限" |
| "调了7个参数还不行 → 我调得不够好" | "这不是调参能解决的问题" |
| "成功率永远0% → 我的设置有问题" | "需要新方法（CILD）" |

### 3. epistemic honesty（认知诚实）
真正的研究人员：**"我赌方法 A 比 B 好，但如果实验证明 B 好，我就接受"**。
判断标准：**如果你的方法 CILD 反而比 baseline 差，你愿意在论文里诚实报告吗？**

### 4. "普遍失败"比"特定失败"更强的故事
- 旧故事：TD-MPC2 在动态环境特别差
- 新故事：**TD-MPC2 在导航/避障任务普遍受限，CILD 系统性解决这个方法论缺陷**

---

## 今晚学到的技术内容

### TD-MPC2 内部机制
- **two-hot encoding**：reward/value 通过 symlog → bin 离散化做 soft cross-entropy
- **vmin/vmax 必须填 symlog 后的范围**（这是个隐藏的坑）
- **world model loss = consistency + reward + value + termination**，各项系数的层次很重要
- **discount, rho, horizon** 三者协同决定 bootstrapping 深度

### Reward shaping 的常见陷阱
1. **Reward hacking**：模型找到"快速结束"或"原地不动"的局部最优
2. **反向激励**：探索失败比不动更亏 → 模型选择不动
3. **数值范围与 vmin/vmax 不匹配**：reward 落在单一 bin，梯度消失
4. **类别不平衡的 BCE loss**：terminated=1 极少，干扰主 loss

### 导航文献的 reward 设计哲学
- 主信号是 progress / 距离势函数
- collision 通常**不作为大 penalty**，靠 episode 终止的"机会成本"自然惩罚
- success_bonus 不需要很大
- 文献：Habitat, DD-PPO, DreamerV3-Crafter

---

## Latent Dynamics 在导航/避障领域的研究地图

| 方向 | 代表工作 | 特点 | 与 CILD 的关系 |
|------|---------|------|---------------|
| 通用 latent world model | DreamerV3, TD-MPC2, IRIS | 不专注动态避障 | baseline |
| 自动驾驶 latent model | MILE, TrafficBots | 强但限于 driving | 邻域参考 |
| Safe RL + world model | SafeDreamer, LAMBDA | safety constraint，但不挖 latent 结构 | 邻域参考 |
| **动态避障 + latent dynamics** | **几乎空白** | **CILD 的位置** | **本研究** |

### CILD 的研究空白点
"专门为动态避障设计的 latent dynamics" 是真实存在的研究空白。
方法论新角度：**让 latent representation 和 navigation cost 协同设计**。

---

## 修正后的研究路径

### 实验1（Motivation）：用公开 benchmark
- **环境**：SafetyPointGoal1（静态）vs SafetyPointGoal2（动态）
- **目的**：展示 TD-MPC2 / Dreamer 等 latent world model 在 navigation 上的局限
- **可信度**：公开 benchmark + 默认 reward + 文献认可

### 实验2（Method validation）
- 在同样的 SafetyPointGoal1/2 上跑 CILD
- 对比 baseline 的相对提升
- **重点：相对提升明显，绝对值不必完美**

### 实验3（Generalization / 应用展示）
- 在 mushr-nav 上跑 CILD vs baseline
- 展示方法在更复杂、更接近 sim2real 的环境也 work
- **mushr-nav 不需要 baseline 跑得好，只需要 CILD 比 baseline 好**

### 实验4（Ablation）
- 拆解 CILD 的三个 cost heads（risk / progress / occupancy）
- 证明每个组件的贡献

---

## 下一步具体计划

### 优先级1（这周）：接入 Safety Gymnasium
1. 安装 `pip install safety-gymnasium`
2. 写 `tdmpc2/envs/safety_pointgoal.py` wrapper（参考 `mushr_nav.py`）
3. **不要改 reward**，使用 Safety Gymnasium 默认 reward
4. 跑 TD-MPC2 在 SafetyPointGoal1 静态版本（默认超参）
5. 记录结果（不论好坏都是数据）

### 优先级2（下周）：动态版本对比
6. 跑 TD-MPC2 在 SafetyPointGoal2 动态版本
7. 对比静态/动态的成功率、collision 率、SPL 等指标
8. **如果 baseline 在静态都不好** → motivation 直接成立（"普遍失败"叙事）
9. **如果 baseline 静态好动态差** → motivation 也成立（差距叙事）

### 优先级3（接下来2-3周）：实现 CILD
10. 在 TD-MPC2 的 WorldModel 里加三个 prediction heads：
    - `C^risk`：(z, a) → ρ ∈ [0,1]
    - `C^prog`：(z, a) → p ∈ R
    - `C^occ`：z → ô ∈ [0,1]^K
11. loss 通过 latent dynamics chain 反传
12. 在 SafetyPointGoal1/2 上验证

### 优先级4（后续）：CILD-MPPI 规划器
13. Gradient-informed sampling
14. Safety constraint pruning
15. OOD-aware horizon

### 长期：mushr-nav 作为复杂场景验证
16. 把上述方法也跑在 mushr-nav 上
17. 写论文：实验1（PointGoal motivation） + 实验2（PointGoal validation） + 实验3（mushr-nav generalization） + 实验4（ablation）

---

## 需要进一步学习的内容

### 短期阅读清单
- [ ] **DreamerV3** (Hafner et al., 2023) - latent world model 标杆
- [ ] **MILE** (Hu et al., 2022, NeurIPS) - 自动驾驶 latent dynamics
- [ ] **SafeDreamer** (Huang et al., 2024, ICLR) - Dreamer + safety
- [ ] **LAMBDA** (As et al., 2022) - latent + safety constraint
- [ ] **Plan2Explore** (Sekar et al., 2020) - 探索导向的 latent

### 比较的维度
对每篇论文整理：
1. latent representation 怎么学的？
2. 怎么处理 cost / safety / collision？
3. 在什么 navigation benchmark 上验证？
4. 与 CILD 的差异点是什么？

---

## 工程上避免的坑（来自今晚的教训）

### Git 工作流
- 调试阶段用 `git commit --amend --no-edit && git push --force-with-lease`
- **不要在 VSCode 里点"同步"按钮**，会产生多余的 merge commit
- 远程机器同步用 `git fetch && git reset --hard origin/yha`，**不要用 `git pull`**

### Reward 工程
- **能用文献默认 reward 就用**，不要自己设计
- 自己设计 reward 至少有 5-7 个未知系数，是无穷搜索空间
- 调超参是有限维度的，调 reward 是无限维度的——**这是本质区别**

### 实验设计
- baseline 必须用**已验证能学**的环境作 motivation
- 自造环境只能用作"复杂场景泛化展示"，不能用作 motivation
- **如果一个实验跑了 50k 步还完全不收敛，先停下来想是不是路径错了**，不要继续调

---

## 心理建设

### 关于"我适合读博吗"的怀疑
今晚我做到了几件**博士生才会做的事**：
- 不接受表面解释，反复追问"为什么"
- 识别出对话里的逻辑漏洞
- 最后**自己**总结出认知突破

博士不是要什么都会，是要有**独立判断和持续追问的能力**。

### 关于研究节奏
- 研究不是按计划推进的线性过程
- **有些晚上的"绕路"是必要的**，认知突破不会按日程表来
- 研究人员被实验逼疯是常态

### 关于今晚
- 一晚上没改完论文不会真的影响什么
- 杯子是杯子，研究是研究
- 真正的收获是认知，不是代码

---

## 一句话总结今天

**baseline 的表现是观测对象，不是可控制的变量。研究的工作不是"安排"实验结果，而是诚实地测量它们。**

---

*Last updated: 2026-05-06 night*

---

## 2026-05-07 重构日：Codebase 切换到 Safety Gymnasium

按照 2026-05-06 的认知突破，今天把仓库从"自造 mushr 环境"重构为"Safety Gymnasium benchmark"路线。

### 关键决定

1. **mushr_nav 归档到 `legacy/`**，不删除 — 实验3（泛化展示）会用到
2. **彻底剥离非导航 benchmark 代码** — dm_control / metaworld / maniskill / myosuite / mujoco 全部删除（共 ~1100 行 envs/tasks/ + 5 个 envs 模块 + evaluate_oracle.py）
   - 理由：与 navigation/avoidance 研究无关；safety-gymnasium 1.0 与 dm-control 1.0.16 在 mujoco/gymnasium 版本上死锁，无法共存
3. **用默认 cfg 跑 baseline** — 不再带 mushr 调出来的 discount/reward_coef/vmin/vmax 等覆盖
4. **目标 benchmark 三任务**：SafetyPointGoal1-v0（静态）、SafetyPointGoal2-v0（动态）、SafetyCarGoal1-v0（不同形态）

### 工程细节

- `tdmpc2/envs/safety_gym.py`：薄 wrapper，把 safety-gymnasium 6-tuple step `(obs, reward, cost, term, trunc, info)` 适配为 TD-MPC2 期望的 4-tuple `(obs, reward, done, info)`
- cost 经 info 透传（CILD 的 risk head 后续会用）
- per-step success 来自 `env.unwrapped.task.goal_achieved`（goal 任务是 continuing，goal 会重生）
- pytorch 2.1 兼容补丁（TD-MPC2 公开版假设 pytorch 2.5+）：
  - `torch.nn.Buffer(...)` → `register_buffer(...)`（`common/scale.py`、`tdmpc2.py`）
  - `torch.compiler.cudagraph_mark_step_begin` 缺失时 monkey-patch 成 no-op（`tdmpc2.py`）

### Smoke test 结果（2026-05-07）

```
SafetyPointGoal1-v0, model_size=1, steps=6000, seed=1, compile=false
  eval E=0  I=0     R=-20.7   (random init)
  eval E=2  I=3000  R=-20.9
  eval E=5  I=6000  R= 6.0    (after pretrain on seed data)
  Training completed successfully
```

**意义**：pipeline 完全打通 — wrapper、buffer、模型 forward/backward、planner、log 都正常工作。R 从 -20 → +6 说明梯度更新有效。这只是 6k 步的可行性验证，**不是 baseline 数据**；正式 baseline 要跑 500k 步、3 seed × 3 任务。

### 下一步

- 跑正式 motivation baseline：每个任务 500k 步、3 seed
- 收集 episode_reward / cost_total / goal_reached_count 三个核心指标
- 不论结果好坏都诚实记录 — 这是实验观测，不是要"安排"的目标

*Last updated: 2026-05-07 evening*

---

## 2026-05-07 续：Metric 扩展 + 第一波 baseline 启动

晚上把"日志只有 R/S/C/G 四列"扩展为更细的 motivation 证据，并在 4 卡上挂起了第一批 baseline 训练。

### Metric 扩展（commit `ba1c734`）

之前只看 episode_cost 总值，看不出 agent 是"贴 hazard 边走"还是"撞 vase"。新增：

| 字段 | 含义 | 哪些任务有 |
|---|---|---|
| `episode_cost_hazards` | 几何危险区累积惩罚（连续值） | G1 / G2 / CarGoal1 |
| `episode_cost_vases_contact` | 撞 vase 的接触 cost | **只 G2 有** |
| `episode_cost_vases_velocity` | 撞 vase 把 vase 推动的 cost | **只 G2 有** |
| `episode_in_hazard_steps` | agent 中心点在 hazard 半径内的步数（binary 计数） | G1 / G2 / CarGoal1 |
| `episode_final_goal_distance` | episode 结束时离 goal 的距离 | 全部 |

console 实时显示精简成 `R / C / G / D`（dropped S，因为在 continuing task 上 per-step success 噪声很大）。

### 重要的概念校正

**Hazards 不是物理碰撞物**：`contype=0, conaffinity=0`，agent 直接穿过。`cost_hazards = 1.0 * (0.2 - h_dist)` 是**几何距离惩罚**，不是"撞墙次数"。**只有 vases（PointGoal2）才是真正的物理碰撞物**。所以 G1 vs G2 的真实差异是"agent 在违规几何区里走 vs 真物理碰撞"。

**Goal 世界没有围墙**：random init policy 容易飞远，初期 `final_goal_distance=20+` 是**真实环境行为，不是 bug**。baseline 学会的标志之一：D 从 ~20 降到 ~3 以内。

### Model size 决策

TD-MPC2 paper 在 single-task benchmark 用 `model_size=5`（~5M 参数），不是 317（multi-task 才用）。决定主跑 size=5，理由：
- 是 paper 自己的 single-task 默认配置 — 最不容易被审稿质疑
- 不需要 size=317：故事是"latent 结构性受限"，不是"参数不够"。如果 size=317 突然学会，反而毁掉 motivation
- size=1 太小，会被质疑 capacity 不足

### Baseline 训练启动（远程 4 卡 2080 Ti）

23:37-23:41 启动，tmux session, 每个 run 500k 步：

| GPU | 任务 | seed | exp_name |
|---|---|---|---|
| 0 | SafetyPointGoal1-v0 | 1 | baseline_m5_seed1 |
| 1 | SafetyPointGoal2-v0 | 1 | baseline_m5_seed1 |
| 2 | SafetyCarGoal1-v0 | 1 | baseline_m5_seed1 |
| 3 | SafetyPointGoal1-v0 | 2 | baseline_m5_seed2 |

参数：`model_size=5, eval_freq=10000, eval_episodes=5, compile=false, save_agent=true`。

### Pretrain 后第一个 episode 已经显示出 motivation 信号

PointGoal2 train.csv step=6000（pretrain 刚结束）：
```
R=1.59  cost=351  hazards=36  vases_contact=75  vases_velocity=314  D=2.59
```

TD-MPC2 已经学到"朝 goal 走"（R 升、D 短），但**完全不看 cost**：撞 vases 撞得猛（vases_velocity=314），完全为了 reward 不顾安全约束。这正是 CILD 想解决的：latent 没编码 cost 信息，planner 没用 cost 约束。

是早期信号，最终 baseline 表现要看 50k 步后稳定下来的 eval.csv。

### 工程小坑记录

- `pip install ... | tee` 不接文件名会让 shell 卡住等输入，**tee 后必须有文件名**
- python 输出 piped 进 tee 时变成 block-buffered，console 显示延迟（不影响训练）。要实时输出加 `python -u`
- 当前 CILD conda env 里的 pytorch 是 2.1.1（不是 nightly 2.6），用两行 monkey-patch 兼容（已 commit）。**不需要升级 torch**：smoke + sanity 都通过，强行升 torch.compile 行为变化反而风险大
- 远程环境是 `yha`（不是 `CILD`），名字不重要，包齐就行

### 还没做的

- 正式 baseline 还没结束（4 个 run 在跑，预估 6-12 小时）
- 第二、三 seed 没启动 — 等明早第一波数据出来再决定
- mushr 在 legacy/，等 motivation 跑完再说复现
- environment.yaml 还不算"可完整复现" — 等下次需要 clean rebuild 时再修

*Last updated: 2026-05-07 night*
