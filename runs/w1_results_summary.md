# Week 1 — DINO-WM Push-T 官方基线复现结果

> 环境: AutoDL 单机 RTX 4080 SUPER 32G；conda env `/root/envs/dino_wm` (Python 3.9.19)
> 代码: `gaoyuezhou/dino_wm` commit `0a9492fa`（本地改动见 `../dino_wm/README`）
> 权重: Push-T 官方 `model_latest.pth` (361M)
> 配置: `dino_wm/conf/plan_pusht_local.yaml`（本地 launcher + 输出重定向到系统盘）
> 数据集: `pusht_noise`，规划目标取自 val 集；`goal_source=dset, goal_H=5`
> 所有运行固定 `seed=99, n_evals=10`（2-ep smoke 除外）

## 1. 主结果：官方权重在 Push-T 上的规划评测

| run | episodes | 成功率 | mean_state_dist | mean_visual_dist | mean_proprio_dist | MPC 轮数 | CEM eval 次数 | 产物目录 |
|---|---|---|---|---|---|---|---|---|
| 2-ep smoke | 2 | **1.0 (2/2)** | 26.27 | — | — | 1 | 8 | `20261126215344_pusht_gH5` |
| 10-ep 正式评测 | 10 | **1.0 (10/10)** | 40.24 | 2.41 | 13.58 | 1 | 13 | `20261126223924_pusht_gH5` |

10-ep 评测的 CEM 在第 13 步全成功提前终止（`opt_steps=30` 是上限）。前 12 步成功率为 0.9，
10 个里有 1 个（索引 2）state_dist 一直在 100+，第 13 步降到 104.5 后被判成功，成功率升到 1.0 后 break。

## 2. 规划参数时延对比（B 部分最后一项）

固定 `seed=99, n_evals=10`，每次只改一个参数。

| 指标 | 基线 | 变体 A: `num_samples=150` | 变体 B: `eval_every=5` |
|---|---|---|---|
| WALL 墙钟 | ~597s (≈10 min)¹ | **6762s (113 min)** | 725s (≈12 min) |
| MPC 轮数 | 1 | **9（未收敛）** | 1 |
| 最终成功率 | **1.0 (10/10)** | **0.9 (9/10，卡住)** | **1.0 (10/10)** |
| mean_state_dist | 40.24 | 28.38 | 36.64 |
| mean_visual_dist | 2.41 | 2.39 | 2.36 |
| mean_proprio_dist | 13.58 | 10.75 | 12.87 |
| CEM eval 次数 | 13 | **310** | 6 |
| 产物（PNG / mp4） | 13 / 20 | 30 / 90 | 4 / 20 |
| 产物目录 | `20261126223924` | `20261127113321` | `20261127132557` |

¹ 基线未用 `SECONDS` 精确计时，由输出目录文件时间戳（22:39:24 → 22:49:21）推算；变体 A/B 为日志中 `WALL=` 精确值。

### 结论（仅理解时延，不作为论文新结论）

- **`num_samples` 减半是负面案例**：候选 300→150 没有线性提速，反而慢 10 倍以上且成功率从 1.0 掉到 0.9。
  日志显示 10 个 episode 中 1 个始终规划不成功，成功率卡在 0.9，MPC 的 `while` 循环
  （`max_iter=null`，全成功才停）反复重规划，跑了 9 轮、310 次 eval 仍未收敛。
  说明 `num_samples` 不只是算力旋钮，它直接决定搜索质量与收敛。
- **`eval_every` 拉大基本不提速**：725s vs 基线 ~597s，只是 eval 从 13 次降到 6 次、PNG 从 13 张降到 4 张，
  但 CEM 的 300 候选 rollout 一次没少。说明瓶颈在 GPU 里的想象 rollout，不在 eval 的物理回放+存图。
- 两个旋钮在当前配置下都没带来明显提速；此表按 Gate G1 要求如实记录，不宣称任何未经验证的成功率结论。

## 3. 关键实现细节（写 notes 时可直接引用）

- 规划是**在线闭环**：外层 MPC 滚动重规划（执行 `n_taken_actions=5` 个宏动作 = 25 帧后从真实观测重规划），
  内层 CEM 在世界模型潜空间里开环优化（每轮采样 300 条 `(horizon=5, action_dim=10)` 序列，取 top-30 更新 mu/sigma）。
- 每次 CEM eval 做两次 rollout：世界模型想象（GPU, latent）+ 真实 Push-T 仿真回放（从固定起点重放整条动作，
  给可信 success 判据并用于 early-stop）。早停条件 `np.all(successes): break`。
- observation：DINOv2(vits14) patch tokens `(B,T,196,384)`（196=14×14，图像先 resize 到 196 再编码）；
  action：原始 2 维 `(dx,dy)` × frameskip 5 = 10 维宏动作；state：5 维 `[agent_x, agent_y, T_x, T_y, T_angle]` + 2 维速度；
  goal：val 轨迹 replay `goal_H=5` 步后的观测。
- 成功判据：`‖goal[:4]-cur[:4]‖ < 20` 像素 且 角度差 `< π/9`。`mean_state_dist` 是 7 维全向量范数，
  与成功判据不是一回事（含速度差），不要拿它当阈值看。
