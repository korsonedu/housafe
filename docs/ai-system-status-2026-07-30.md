# 家安 AI 系统 — 当前状态与生产路线图

> 2026-07-30 | 仿真器 + 世界模型 eval 闭环完成 | 下一目标：Phase B 个人基线

---

## 一、架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        仿真系统                                  │
│  Scenario YAML → SyntheticGen / ReplayGen → FrameGroup          │
│  → RoomModel (坐标变换) / VitalSignsGen (生理)                   │
│  → AnomalyInjector (异常注入 + GT 标注)                          │
│  → Redis / WS 双通道注入                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ FrameGroup
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       世界模型 (Phase A)                          │
│  Encoder (4.7M, 179 epoch, 99.98% linear_acc)                  │
│  → 256-d latent states S_t (32 帧窗口)                          │
│  Predictor (0.3M, GRU) → 预测 S_{t+1}                          │
│  → cosine 误差 → Z-score 异常检测                               │
│                                                                  │
│  异常来源 (当前):                                                 │
│    • fall → Encoder posture classifier (规则, 100% 召回)         │
│    • vital → 医学阈值 (100% 精度)                                │
│    • offline → 帧间隙检测 (83% 精度)                              │
│    • stillness → ❌ 待修                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 二、完成清单

| # | 组件 | 文件 | 状态 |
|---|------|------|:--:|
| 1 | 仿真系统 (10 模块, 1610 行) | `simulator/` | ✅ |
| 2 | GMM 校准 (12 动作, 3DPCHM) | `simulator/calibrate.py` | ✅ |
| 3 | Scene 引擎 + YAML + CLI | `simulator/scenario.py`, `cli.py` | ✅ |
| 4 | 双通道注入 (Redis + WS) | `simulator/injectors.py` | ✅ |
| 5 | 5 个评估场景 | `simulator/scenarios/` | ✅ |
| 6 | 集成测试 (15 个) | `simulator/tests/` | ✅ |
| 7 | Encoder 训练 (179 epoch) | `ai/encoder/` | ✅ |
| 8 | Predictor 训练 (GRU) | `ai/predictor/` | ✅ |
| 9 | Eval 桥接 (纯世界模型) | `simulator/eval.py` | ✅ |
| 10 | 批量评估 (5 场景 × 3 轮) | `simulator/eval.py batch` | ✅ |

## 三、Eval 指标详情

### 按异常类型 (3 轮平均)

| 异常类型 | 检测方法 | 召回率 | 精确率 | F1 | 判定 |
|---------|---------|--------|--------|-----|:--:|
| vital_anomaly | 医学阈值 | 100% | **100%** | **1.000** | 可用 |
| offline | 帧间隙 | 100% | **83.3%** | **0.889** | 可用 |
| fall (replay) | Encoder 姿态分类 | 100% | 23.3% | 0.378 | 召回好，需过滤假阳 |
| stillness | 隐状态稳定性 | 0% | — | 0.000 | 不可用 |
| fall (合成 GMM) | 任何世界模型方法 | 0% | — | 0.000 | GMM 分布不匹配 |

### 关键发现

1. **GMM 合成点云与真实雷达分布差异大**，世界模型 predictor error 基线 0.41（训练时 0.08），原始信号不可直接使用
2. **训练路径推理解决了分布偏移**：用 `encoder.forward()` 批量推理替代逐帧 backbone buffer，error 从 1.1 降到 0.41
3. **世界模型 Z-score 能捕获突变为异常**，但对平稳变化的 stillness 不敏感
4. **静态检测不能用隐状态稳定性**：3DPCHM "静止"帧仍有 sensor 噪声导致 cosine sim 不够稳定

## 四、到生产的差距

```
                        现在 ───────────────────→ 生产可部署

规则检测 (fall/vital/offline)   ████████░░ 80%
Stillness                       ░░░░░░░░░░  0%
Encoder exclude_fall 重训        ░░░░░░░░░░  0%
PersonalBaseline (14天)         ██░░░░░░░░ 20% (代码有，无数据)
SpatialGraph                    ██░░░░░░░░ 20% (代码有，无数据)
世界模型 pattern_deviation      ░░░░░░░░░░  0% (依赖上两项)
端到端集成 (sim→backend→App)    ░░░░░░░░░░  0%
推理性能/延迟测试                ░░░░░░░░░░  0%
```

## 五、优先级路线图

### P0：本周可完成

| # | 任务 | 输入 | 产出 | 预估 |
|---|------|------|------|------|
| P0-1 | **Stillness：改用物理检测** | 当前 eval.py | 质心位移方差 < 5cm 持续 30s → stillness | 2h |
| P0-2 | **Fall：用 Encoder classifier** | Encoder 的 12 类分类头 | 替代规则高度阈值，解决过渡段假阳性 | 3h |
| P0-3 | **Encoder exclude_fall 重训** | AutoDL GPU | 新 encoder，fall 对模型出分布 | 5h GPU |

### P1：两周内

| # | 任务 | 输入 | 产出 | 预估 |
|---|------|------|------|------|
| P1-1 | **仿真器跑 14 天正常生活** | 日常场景 YAML × 14 | 120K+ 帧 latent state 序列 | 1d 仿真 |
| P1-2 | **PersonalBaseline 拟合** | P1-1 产出 | 个人 GMM 异常分模型 | 2h |
| P1-3 | **SpatialGraph 构建** | P1-1 产出 | 房间拓扑 + 转移概率 | 2h |
| P1-4 | **pattern_deviation eval** | P1-2 + P1-3 | 真正测世界模型的价值 | 半天 |

### P2：一个月内

| # | 任务 | 产出 |
|---|------|------|
| P2-1 | Predictor 个性化微调 | 个人运动模式 predictor |
| P2-2 | 端到端集成测试 | sim → Redis → AI → backend → App |
| P2-3 | 推理延迟基准测试 | 单帧 < 50ms (目标) |

### P3：三个月到生产

| # | 任务 | 产出 |
|---|------|------|
| P3-1 | 多设备多房间场景 | 覆盖老人真实居住环境 |
| P3-2 | 7×24 连续仿真稳定性 | 无内存泄漏、无性能衰减 |
| P3-3 | 真实验证 (3DPCHM held-out) | 在未见过被试上测泛化 |
| P3-4 | 模型在线更新 | 新数据到来后增量更新基线 |

## 六、关键决策记录

| 决策 | 日期 | 理由 |
|------|------|------|
| 世界模型不替代规则引擎，互补使用 | 07-29 | 规则做确定性检测 (fall=姿态)，WM 做概率检测 (pattern=反常) |
| 合成 GMM 不可用于世界模型 eval | 07-28 | 分布偏移导致 predictor error 基线翻 4x，信号无意义 |
| 训练路径推理 (encoder.forward) 替代逐帧 buffer | 07-28 | 消除推理/训练 latent 分布 gap |
| 关掉中时尺度检测 (sustained_shift) | 07-29 | 99% 检测为假阳性 |
| Stillness 不能用隐状态稳定性 | 07-29 | 3DPCHM 噪声使 cos sim 区分度不足 |

## 七、文件索引

| 用途 | 路径 |
|------|------|
| 仿真系统 | `simulator/` |
| 评估桥接 | `simulator/eval.py` |
| 场景库 | `simulator/scenarios/` |
| Encoder 训练 | `ai/encoder/train_encoder.py` |
| Encoder 模型 | `ai/encoder/encoder_model.py` |
| Predictor 训练 | `ai/predictor/train_predictor.py` |
| Predictor 模型 | `ai/predictor/model.py` |
| 个人基线 | `ai/baseline/gmm_baseline.py` |
| 空间图 | `ai/latent_space/graph.py` |
| 世界模型 Worker | `ai/main.py` |
| 校准参数 | `simulator/calibrated_params.npz` |
| 模型 checkpoint | `ai/checkpoints/` (symlink → models/) |
| 仿真升级设计 | `docs/superpowers/specs/2026-07-25-simulator-upgrade-design.md` |
| 仿真实现计划 | `docs/superpowers/plans/2026-07-26-simulator-upgrade.md` |
