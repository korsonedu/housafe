# 家安 AI 系统 — 当前状态与路线图

> 2026-08-01 | LatentBaseline 验证通过 (AUC 0.83) | 下一目标：事前健康管理

---

## 一、架构总览（更新）

```
┌─────────────────────────────────────────────────────────────────┐
│                        仿真系统                                  │
│  Scenario YAML → ReplayGen (3DPCHM) → FrameGroup               │
│  → RoomModel (坐标变换) / VitalSignsGen (生理)                   │
│  → AnomalyInjector (异常注入 + GT 标注)                          │
│  → Redis / WS 双通道注入                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    世界模型 (Phase B)                             │
│                                                                  │
│  Encoder (4.7M, 179 epoch, 99.98% acc)                         │
│  → backbone: point cloud(64×5) → 1024-d feature               │
│  → TCN buffer(32帧) → 256-d latent state S_t                   │
│       │                                                         │
│       ├── LatentBaseline: PCA(12) + GMM(3) × 6 contexts        │
│       │   → S_t 密度估计 → anomaly score                       │
│       │   → 规则引擎做不到：检测行为模式偏离个人常轨                │
│       │                                                         │
│       └── Predictor (0.3M GRU, 已重训):                        │
│           S_t-31:t → Ŝ_t+1 → 辅助信号 (AUC 0.68)              │
│                                                                  │
│  融合层 (AnomalyDecoder):                                        │
│    规则(确定性) + LatentBaseline(概率, AUC 0.83)                 │
│    + Predictor误差(辅助) + NLL偏离(辅助, score_nll_percentile)   │
└─────────────────────────────────────────────────────────────────┘
```

## 二、关键实验记录

### 2.1 Predictor 重训（7/31）

| 指标 | 旧 Predictor | 新 Predictor | 说明 |
|------|-------------|-------------|------|
| 训练数据 | 孤立动作片段 | 4h 连续生活场景 | 包含跨动作转移 |
| Val error | 0.08 (训练集上) | 0.01 | 分布对齐了 |
| Replay error | 0.79 | 0.01 | **80x 改善** |
| Fall/正常比 | 0.95x | 1.48x | 仍不够 |
| AUC (replay fall) | - | 0.68 | 弱信号 |

**结论：** Predictor error 解决不了瞬态异常（TCN 平滑）。保留作为辅助信号。

### 2.2 LatentBaseline 验证（8/1）

| 实验 | 数据 | PCA | GMM | Contexts | AUC (fall) |
|------|------|-----|-----|----------|------------|
| v0 | 0.5h | 32d | 5 | 1 (时间戳bug) | 0.86 |
| v1 | 6h | 32d | 5 | 1 (同上) | 0.77 |
| v2 | 1h | 12d | 3 | **6** | **0.83** |

**结论：**
- S_t 密度估计能检测 fall（AUC 0.83），比 predictor error（0.68）强 22%
- 时间分上下文 + 小 PCA + 少分量 = 泛化好
- Stillness/vital 检测不到（预期——不是空间模式异常）
- 核心理念成立：**Encoder S_t 空间包含了可用的个人常轨表示**

## 三、时域检测能力矩阵

| 时间尺度 | 检测器 | 方法 | 状态 |
|---------|--------|------|:--:|
| 秒 | FallbackEngine | 确定性规则 (fall/vital/stillness/offline) | ✅ |
| 秒 | LatentBaseline | 单帧 S_t 密度异常 | ✅ AUC 0.83 |
| 秒 | Predictor error | cosine 误差辅助信号 | ⚠️ AUC 0.68 |
| 天 | — | 当日 S_t 分布 vs 基线分布 | ❌ 待建 |
| 周 | — | 跨周分布漂移 (KL/Wasserstein) | ❌ 待建 |
| 月 | — | 长期行为趋势退化 | ❌ 待建 |

## 四、到生产的差距

```
现在 ──────────────────────→ 生产可部署

单帧异常 (规则+WM)         ██████████ 100%
Predictor 重训             ████████░░  80%
LatentBaseline 验证        ████████░░  80% (仅 1h 仿真，需 14 天)
端到端集成                 ███░░░░░░░  30% (代码有，未联调)
推理延迟                   ████████░░  80% (CPU 上 ~1ms/帧 backbone)
事前健康管理               ░░░░░░░░░░   0%
```

## 五、新增文件索引

| 用途 | 路径 |
|------|------|
| **LatentBaseline** | `ai/baseline/latent_baseline.py` |
| **拟合+Eval 流水线** | `ai/baseline/fit_and_eval.py` |
| Predictor 训练 v2 | `ai/predictor/train_predictor_v2.py` |
| 训练数据生成 | `ai/predictor/generate_training_data.py` |
| LatentBaseline checkpoint | `ai/checkpoints/latent_baseline.pkl` |
| 训练数据 | `ai/data/training/predictor_continuous.npz` |
| 旧 Predictor 备份 | `ai/checkpoints/predictor_best.pt.bak` |

## 六、关键决策更新

| 决策 | 日期 | 理由 |
|------|------|------|
| 世界模型核心 = LatentBaseline (latent density)，不是 predictor error | 08-01 | predictor 信号太弱(AUC 0.68), 密度估计可用(AUC 0.83) |
| Predictor 保留为辅助信号 | 08-01 | 重训后 error 正常化(0.01)，但缺区分度 |
| S_t 密度估计 > FeatureVector 密度估计 | 08-01 | Encoder 学的语义空间比手写特征丰富 |
| PCA 12d + GMM 3 分量 > PCA 32d + GMM 5 分量 | 08-01 | 少参数 → 更好泛化 |
| backbone+buffer+TCN 提取方式比 encoder.forward() 快 100x | 08-01 | 生产可行 (CPU 上 ~1ms/帧) |
| 世界模型不替代规则引擎，互补 | 07-29 | 规则做确定性，WM 做概率性 |
