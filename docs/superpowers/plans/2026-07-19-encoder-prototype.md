# Phase B0 · 世界模型完整训练与可用性验证

> **目标：** 在公开数据集上完成 Encoder + Predictor 完整训练，产出端到端的世界模型原型，用硬数字回答"世界模型路线能不能走通"。
>
> **这不是可行性实验。这是 Phase B/C 的提前演练。** 用公开数据替代合成数据，跑通完整训练→评估→决策闭环。

## 要验证的 4 个核心假设

这 4 个假设来自算法文档，目前全是理论推导。本计划用实验逐一验证：

| # | 假设 | 验证方式 | 如果为真 | 如果为假 |
|---|------|------|------|------|
| H1 | 稀疏毫米波点云包含足够信息编码人体状态 | Encoder 隐空间的线性分类准确率 ≥ 85% | Encoder 可行 | 需要更高分辨率传感器或更多雷达 |
| H2 | 隐空间是连续的（过渡帧在"坐"和"站"之间连续分布，不是二值跳变） | 可视化 + 过渡帧余弦距离梯度 | 世界模型根基成立 | Encoder 退化为分类器，世界模型失去意义 |
| H3 | Predictor 能从历史 S_t 预测未来 S_{t+1}，误差显著优于"预测不变"基线 | cosine(Ŝ, S_real) vs cosine(S_t, S_real) | 预测偏差可以作为异常信号 | 时序信息无用，Predictor 不需要 |
| H4 | 只在正常动作上训练的 Predictor，对未见过的跌倒序列产生显著更高的预测误差 | 正常预测误差 vs 跌倒预测误差，t-test p < 0.01，effect size > 2× | **世界模型路线成立** | 世界模型的"预测偏差=异常"理论不成立 |

**H4 是最终判决。** H1-H3 通过但 H4 不通过 = Encoder 能工作，但世界模型的独特价值不存在——退回到规则引擎照样做。

## 数据集

**主要：mmWave-3DPCHM-1.0**

| 维度 | 值 |
|------|-----|
| 雷达 | TI IWR1443-ISK + Vayyar vBlu |
| 动作 | 12 类：stand, sit, fall, walk, squat, jump, punch, wave_left, wave_right, lean_left, lean_right, open_arms |
| 被试 | 7 人 |
| 点云密度 | 10-50 点/帧（论文数据） |
| 来源 | [雷达学报](https://www.radars.ac.cn/web/data/getData?newsColumnId=1003be8d-90fe-449d-8096-e1c9a1cdb497) |

**辅助：mmPrivPose3D**

| 维度 | 值 |
|------|-----|
| 雷达 | TI IWR6843AOPEVM（**与家安目标同款**） |
| 动作 | 行走 + 手势 |
| 被试 | 15 人 |
| 数据量 | ~11 万帧 |
| 额外价值 | 有 3D 骨骼关键点 ground truth，可用于验证隐空间的物理可解释性 |
| 来源 | [Mendeley](https://data.mendeley.com/datasets/pmdr5rgn8c/1) |

**数据划分策略（关键）：**

```
训练集: 5 人的 11 种正常动作（排除跌倒）
        + mmPrivPose3D 行走数据（增加数据量 + 不同雷达 domain）

验证集: 1 人的 11 种正常动作（调参用）

测试集: 1 人的 全部 12 种动作（含跌倒）
        用于最终评估 H1-H4

跌倒序列: 从所有 7 人的跌倒动作中提取（仅用于 H4 测试，不参与训练）
```

**为什么 leave-one-subject-out：** 如果随机 split 帧，同一人的相邻帧可能分到训练和测试集 → 数据泄露。留一人做测试是最严格的泛化验证。

---

## 架构总览

```
                  公开数据集
                      │
          ┌───────────┴───────────┐
          │                       │
    正常动作点云              全部动作点云
    (11类, 5人)             (12类, 1 held-out 人)
          │                       │
          ▼                       │
┌─────────────────────┐           │
│     Encoder          │           │
│  PointNet++ → TCN    │           │
│  输出: S_t (256-d)   │           │
└─────────┬───────────┘           │
          │                       │
    正常 S_t 序列                 │
          │                       │
          ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│    Predictor         │  │   评估               │
│  Causal Transformer  │  │  H1: 线性分类        │
│  输入: [S_{t-16:t}]  │  │  H2: 隐空间连续性    │
│  输出: Ŝ_{t+1}       │  │  H3: 预测精度        │
└─────────┬───────────┘  │  H4: 异常检测 ← 最终  │
          │              └─────────────────────┘
          ▼
  正常预测误差分布
  (baseline for H4)
          │
          ▼
  跌倒序列 → Predictor → 预测误差
                            │
                    误差显著高于正常?
                    Yes → H4 成立
```

---

## Task 1: 数据准备（2 天）

### 1.1 下载

```bash
# mmWave-3DPCHM-1.0
# 注册: https://www.radars.ac.cn
# 下载页: https://www.radars.ac.cn/web/data/getData?newsColumnId=1003be8d-90fe-449d-8096-e1c9a1cdb497
mkdir -p ai/data/public/raw/mmwave_3dpchm

# mmPrivPose3D
# 下载: https://data.mendeley.com/datasets/pmdr5rgn8c/1
mkdir -p ai/data/public/raw/mmprivpose3d
```

### 1.2 解析 + 统计

写 `ai/data/public/parse_3dpchm.py` 和 `ai/data/public/parse_mmprivpose3d.py`，输出：

- `frames: list[dict]`：每个 dict = `{subject_id, action_label, points: np.array[N,5], frame_idx}`
- 统计报告：每类动作帧数、每帧点数分布（min/max/mean/std）、每被试动作分布
- 3D 散点图：每类动作随机抽 3 帧，保存为 PNG（肉眼检查数据质量）

### 1.3 统一 Dataset

写 `ai/data/public/dataset.py`：

```python
class RadarActionDataset(torch.utils.data.Dataset):
    """每个样本: (points: Tensor[max_points, 5], subject_id: int, action_label: int)"""
    
    def __init__(self, frames, max_points=64, split="train", heldout_subject=None):
        # split="train": 排除 heldout_subject + 排除 fall 动作
        # split="val":   heldout_subject 的 11 类正常动作
        # split="test":  heldout_subject 的全部 12 类动作
        ...

class SequenceDataset(torch.utils.data.Dataset):
    """每个样本: (seq: Tensor[window, max_points, 5], action_label: int)
    从连续帧中滑动窗口采样。窗口长度 = 32 帧 (~2s)。
    """
    def __init__(self, frames, window_size=32, stride=4, ...):
        # 按 subject + 时间戳排序
        # 滑动窗口，步长 4 帧（时间重叠增加数据量）
        ...
```

---

## Task 2: Encoder 训练（5-7 天）

### 2.1 模型

```
PointNet++ Backbone (开源 yanx27/Pointnet_Pointnet2_pytorch)
  ├─ 修改: 输入通道 3 → 5 (x,y,z,velocity,intensity)
  ├─ Set Abstraction ×3 → Global Max Pooling → 1024-d global feature
  └─ 参数量 ~1.5M

Temporal ConvNet
  ├─ 4 层因果空洞卷积, kernel_size=3, dilations=[1,2,4,8]
  ├─ 输入: (B, T, 1024) PointNet++ per-frame features
  ├─ 输出: (B, T, 256) temporal-aware features
  └─ 参数量 ~1M

投影头
  ├─ Linear(256, 128) → ReLU → Linear(128, 256)
  └─ 用于对比学习（SimCLR style projection head）

分类头 (仅辅助训练)
  └─ Linear(256, 12)
```

### 2.2 损失函数（多任务联合训练）

```python
# 1. 时间对比损失（主损失，自监督）
#    同一序列内相邻帧 (S_t, S_{t+1}) 是 positive pair
#    不同序列的帧是 negative pairs
L_contrastive = InfoNCE(S_t, S_pos, S_neg, temperature=0.1)

# 2. 辅助分类损失（有标签数据）
#    权重低 (0.05)，仅防止隐空间完全坍塌
L_aux = CrossEntropy(classifier(S_mean), action_label)

# 3. Uncertainty 正则
#    防止 Σ_t 爆炸或退化
L_var = 0.01 * KL(N(S_t, Σ_t) || N(0, I))

# 4. 连续性正则（核心世界模型约束）
#    相邻帧的 S_t 变化应该平滑
L_smooth = mean(|S_{t+1} - S_t|_2)

# 总损失
L_encoder = L_contrastive + 0.05*L_aux + L_var + 0.1*L_smooth
```

### 2.3 训练配置

```yaml
optimizer: AdamW
lr: 1e-4 (cosine schedule, warmup 10 epochs)
batch_size: 32 sequences (每个 32 帧)
epochs: 300
GPU: RTX 4090 24GB
estimated_time: ~8h
validation: 每 10 epochs
early_stop: val_contrastive_loss plateau 30 epochs
```

### 2.4 评估（每 10 epochs）

- 验证集对比损失
- 线性分类准确率（冻结 Encoder → LogisticRegression 在验证集上）
- t-SNE 可视化（训练集 2000 帧采样，不同动作着色）

---

## Task 3: Predictor 训练（3-4 天）

### 3.1 数据准备

```
冻结 Task 2 训练好的 Encoder
遍历训练集所有正常动作序列 → 编码为 S_t 序列
输出格式: (S_{t-window}, ..., S_t, S_{t+1})  ← 前 window 帧做输入，S_{t+1} 做 target
```

### 3.2 模型

```python
class CausalStatePredictor(nn.Module):
    """小规模 Causal Transformer — 预测下一帧隐状态"""
    
    def __init__(self, latent_dim=256, num_layers=4, num_heads=8):
        self.input_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, latent_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        # Causal mask: 每个位置只能看到自己和过去
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(32, 32) * float('-inf'), diagonal=1))
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        # ~8M params
    
    def forward(self, S_history):
        # S_history: (B, window, 256)
        x = self.input_proj(S_history) + self.pos_embed[:, :S_history.shape[1]]
        # Self-attention with causal mask
        x = self.transformer(x, x, tgt_mask=self.causal_mask[:x.shape[1], :x.shape[1]])
        # 最后一帧的输出 = 对下一帧的预测
        pred = self.output_proj(x[:, -1, :])
        return pred  # (B, 256)
```

### 3.3 损失函数

```python
# 主损失: 预测 Ŝ_{t+1} 与实际 S_{t+1} 的余弦距离
L_pred = (1 - cosine_similarity(Ŝ_{t+1}, S_{t+1})).mean()

# 多尺度预测（可选，增强异常检测能力）
# 不只是预测 t+1，还预测 t+8 (0.5s后), t+32 (2s后)
L_multi = L_pred_1frame + 0.5*L_pred_8frame + 0.3*L_pred_32frame

# 正则
L_reg = 0.01 * (pred.pow(2).mean())  # 防止预测爆炸

L_predictor = L_pred + 0.3 * L_multi + L_reg
```

### 3.4 训练配置

```yaml
optimizer: AdamW
lr: 3e-4 (cosine schedule)
batch_size: 64
epochs: 200
window_size: 32 帧
GPU: RTX 4090 (或 CPU — 数据是向量，训练很快)
estimated_time: ~2h
```

### 3.5 Baseline 对比

```python
# Persistence baseline: "下一帧 = 当前帧"
baseline_error = mean(cosine_distance(S_t, S_{t+1}))

# Oracle baseline: 在测试集上直接拟合线性回归 S_{t+1} = W * S_t
# 这是理论上任何线性 Predictor 能到的最优值
```

**H3 通过标准：** Predictor 误差 < baseline_error × 0.7（比"预测不变"强 30% 以上）

---

## Task 4: 世界模型可用性评估（2-3 天）

这是整个计划的**最终判决实验**。

### 4.1 H4 实验：预测偏差检测异常

```
┌─────────────────────────────────────────────────────┐
│                   H4 实验流程                         │
│                                                      │
│  训练集（正常动作, 11类, 5人）                        │
│    │                                                 │
│    ├→ Encoder → 正常 S_t 序列                        │
│    │             │                                    │
│    │             ├→ 训练 Predictor                    │
│    │             └→ 计算正常预测误差分布                │
│    │                  mean_normal, std_normal          │
│    │                                                  │
│  测试集（全部动作, 12类, 1人）                         │
│    │                                                 │
│    ├→ 正常动作序列 → Encoder → Predictor              │
│    │    └→ 预测误差: error_normal_test                │
│    │                                                  │
│    └→ 跌倒序列 → Encoder → Predictor                  │
│         └→ 预测误差: error_fall                       │
│                                                      │
│  判决: error_fall / error_normal > 2.0?              │
│        t-test: error_fall vs error_normal p < 0.01?  │
└─────────────────────────────────────────────────────┘
```

### 4.2 评估指标（完整清单）

| # | 指标 | 计算方式 | 门禁 | 验证假设 |
|---|------|------|:---:|:---:|
| M1 | 线性分类准确率 | 冻结 Encoder → LogisticRegression 12 类 | ≥ 85% | H1 |
| M2 | 同动作余弦相似度 | 同一被试同一动作不同帧 → cos(S_a, S_b) | ≥ 0.85 | H1, H2 |
| M3 | 跨被试余弦相似度 | 被试 A 的"站" vs 被试 B 的"站" | ≥ 0.75 | H2 (身份不受损) |
| M4 | 过渡连续性 | 站→坐过渡帧序列的 S_t 沿 PCA 第一主成分的单调性 | Spearman ρ > 0.8 | H2 |
| M5 | 预测精度 vs 基线 | cosine(Ŝ, S_real) vs baseline（预测不变） | 改善 ≥ 30% | H3 |
| M6 | **异常检测 AUC** | 用预测误差作为异常分数，ROC-AUC（正常 vs 跌倒） | **≥ 0.90** | **H4 (最终)** |
| M7 | **异常检测 precision@90** | 固定 90% recall 时的 precision | ≥ 0.70 | H4 |
| M8 | 推理延迟（Encoder+Predictor） | 端到端 CPU 推理 | < 100ms | 工程可行性 |
| M9 | 模型大小 | Encoder + Predictor 总参数量 | < 30MB (ONNX) | 部署可行性 |

**M6 和 M7 是最重要的两个数字。** 它们直接回答：世界模型的"预测偏差=异常"机制，能不能替代传统的规则引擎。

### 4.3 消融实验

```
完整系统: PointNet++ + TCN + CausalTransformer
  对比:
  A. 无 TCN（单帧 PointNet++ 直接输出 S_t）
     → 验证时序聚合对隐空间质量的增益
  
  B. 线性 Predictor（S_{t+1} = W * S_t + b）
     → 验证 Causal Transformer 是否必要，还是一阶马尔可夫就够
  
  C. 无对比损失（纯监督分类训练 Encoder）
     → 验证自监督对比损失是否比纯分类产出更好的隐空间
  
  D. 规则 Baseline
     用点云质心高度做跌倒判断: height < 0.3m + 持续 5s
     → 世界模型相对于最简单规则引擎的优势有多大？
```

### 4.4 可视化输出

```
1. PCA 降维 256-d → 2D 散点图
   - 12 种动作各用不同颜色
   - 跌倒序列用红色轨迹线标出时间顺序
   - 图上标注"正常区域"和"跌倒轨迹进入正常区域的距离"

2. 预测误差时序图
   - x轴 = 时间帧
   - y轴 = 预测误差
   - 正常动作 = 蓝色线（稳定低值，偶有抖动）
   - 跌倒 = 红色线（跌倒瞬间峰值）
   - 阈值线 = mean_normal + 3*std（虚线）

3. 混淆矩阵
   - 行 = 真实动作，列 = 线性分类预测
   - 重点看：跌倒被误判为什么（最有信息量的失败模式）

4. t-SNE 按被试着色
   - 验证 Encoder 是否学到了身份无关的表征
```

---

## Task 5: 最终报告（1 天）

### 5.1 报告结构

```
1. 执行摘要
   - H1-H4 判决（通过/不通过/+ 关键数字）
   - 一句话建议: 继续投 / 调整方向 / 放弃

2. 实验设置
   - 数据集、模型架构、训练配置、评估方法

3. 结果
   3.1 Encoder 训练（H1, H2）
       - 分类准确率、隐空间可视化、连续性分析
   3.2 Predictor 训练（H3）
       - 预测精度、vs baseline、多尺度预测质量
   3.3 世界模型验证（H4）← 核心
       - 异常检测 AUC、预测误差分布对比
       - 跌倒 vs 正常的分辨力
   3.4 消融实验
       - 各组件的贡献

4. 讨论
   - 为什么行 / 为什么不行
   - 公开数据 vs 真实部署的 gap 分析
   - 如果行：下一步需要什么
   - 如果不行：替代方案

5. 附录
   - 所有可视化大图
   - 训练日志
   - 代码仓库位置
```

### 5.2 决策矩阵

| H1 | H2 | H3 | H4 | 判决 | 行动 |
|:---:|:---:|:---:|:---:|------|------|
| ✅ | ✅ | ✅ | ✅ | **全通过** | 世界模型路线成立，启动 Phase B1 合成管线 + Phase C |
| ✅ | ✅ | ✅ | ❌ | Encoder 能用，但世界模型不成立 | 简化路线：Encoder + 规则引擎 + GMM 基线。不做 Predictor |
| ✅ | ✅ | ❌ | ❌ | 点云可编码，但时序没用 | 单帧分类器 + 规则引擎。不走自监督 |
| ✅ | ❌ | - | - | Encoder 退化为分类器 | 传统管道架构，放弃世界模型 |
| ❌ | - | - | - | 毫米波点云信息量不够 | **根本性路线错误**。需要加传感器或换模态 |

---

## 时间线

```
Week 1:
  Day 1-2:   Task 1 (数据准备)
  Day 3-7:   Task 2 (Encoder 训练 + 调参)

Week 2:
  Day 1-3:   Task 2 (Encoder 最终评估)
  Day 4-5:   Task 3 (Predictor 训练)
  Day 6-7:   Task 4 (H4 实验 + 消融)

Week 3:
  Day 1:     Task 4 (可视化)
  Day 2:     Task 5 (报告)
```

**总计约 3 周。**

训练 GPU 时间估计：

| 阶段 | GPU 时间 | 
|------|------|
| Encoder 300 epochs | ~8h |
| 调参重训 ×2 | ~16h |
| Predictor 200 epochs | ~2h |
| 消融实验 ×4 | ~12h |
| **总计** | **~38h ≈ 80 元（4090 2元/h）** |

---

## 代码结构

```
housafe/ai/
├── encoder/
│   ├── __init__.py
│   ├── pointnet_backbone.py    # 开源 PointNet++ (修改输入通道)
│   ├── tcn.py                  # Temporal ConvNet
│   ├── encoder_model.py        # 完整 Encoder (PointNet++ + TCN + 投影头)
│   ├── losses.py               # 对比损失 + 分类损失 + 平滑损失
│   └── train_encoder.py        # 训练脚本
│
├── predictor/
│   ├── __init__.py
│   ├── model.py                # CausalStatePredictor
│   ├── losses.py               # 预测损失（余弦距离 + 多尺度）
│   ├── train_predictor.py      # 训练脚本
│   └── anomaly_detector.py     # 预测偏差 → 异常分数
│
├── eval/
│   ├── __init__.py
│   ├── latent_space.py         # H1, H2: 分类 + 连续性 + 可视化
│   ├── prediction.py           # H3: 预测精度评估
│   ├── anomaly.py              # H4: 异常检测 AUC + 预测误差对比
│   └── ablation.py             # 消融实验
│
├── data/
│   ├── __init__.py
│   └── public/
│       ├── __init__.py
│       ├── parse_3dpchm.py     # mmWave-3DPCHM-1.0 解析
│       ├── parse_mmprivpose3d.py
│       ├── dataset.py          # RadarActionDataset + SequenceDataset
│       └── stats.py            # 数据统计 + 可视化
│
├── configs/
│   ├── encoder.yaml
│   └── predictor.yaml
│
└── checkpoints/                # 训练产物
    ├── encoder_best.pt
    └── predictor_best.pt
```

## 风险

| 风险 | 概率 | 影响 | 对策 |
|------|:---:|------|------|
| mmWave-3DPCHM 数据集下载不了 | 中 | 只能用 mmPrivPose3D（没有跌倒）→ H4 无法验证 | 联系作者邮件获取；或自己录 10 段跌倒视频用 TI SDK 生成点云 |
| 训练不收敛（对比损失坍缩） | 中 | 拿不到有效 Encoder | 退回到纯监督分类训练 → 评估 H1/H2 → 至少拿到"点云能否编码"的答案 |
| 数据量太小（7人），过拟合 | 中 | 测试集性能虚低，误判"不可行" | 用 leave-one-subject-out 交叉验证（7折）取平均，不受单次 split 影响 |
| 公开数据是实验室环境，点云比真实干净 | 高 | Encoder 在真实部署时性能下降 | **这是已知问题，不是本计划的失败。** 本计划只验证"架构能否工作"，domain gap 由 Phase B1 合成管线 + fine-tune 解决 |
| Predictor 预测误差区分不了跌倒和剧烈运动（跳/跑） | 中 | H4 假阴性 | 检查混淆矩阵——如果跌倒和"跳"的误差都高，至少证明了能区分"正常"和"意外"。用持续性（连续 N 帧高误差）做二次过滤 |
