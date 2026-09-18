# Phase B0 世界模型训练报告

## 结论

**世界模型路线成立。** 仅用 7 人正常行为数据训练，模型在从未见过跌倒的情况下，靠"预测偏差"检出异常，AUC 0.94。

## H1-H4 验证结果

| 假设 | 指标 | 结果 | 产品含义 |
|------|------|------|---------|
| H1 信号够用 | 12 类动作线性分类准确率 | **> 95%** | 毫米波点云包含足够信息编码人体状态 |
| H2 状态平滑 | 相邻帧隐状态余弦相似度 | **0.9582**（96% 帧 > 0.85） | 状态变化是渐进的，能测量退化过程 |
| H3 行为可预测 | 下一帧预测准确率 | **91.5%**（比基线提升 47%） | 正常生活是可预测的，偏离即异常 |
| H4 未知也能发现 | 异常检测 AUC（8 帧窗口） | **0.91** | 见过 11 种正常，能发现第 12 种危险 |

窗口聚合详细结果：

| 窗口 | 时长 | AUC | 正常误差 | 跌倒误差 | Ratio |
|------|------|-----|----------|----------|-------|
| 逐帧 | 50ms | 0.74 | 0.081 | 0.429 | 5.3x |
| Win-4 | 0.2s | 0.87 | 0.156 | 0.838 | 5.4x |
| Win-8 | 0.4s | **0.91** | 0.200 | 1.094 | 5.5x |
| Win-16 | 0.8s | **0.94** | 0.254 | 1.337 | 5.3x |

## 数据集

**mmWave-3DPCHM-1.0**（江苏科技大学 / 雷达学报，2025）

| 维度 | 值 |
|------|-----|
| 传感器 | TI IWR1443-ISK（77GHz 毫米波） |
| 被试 | 7 人 |
| 动作 | 12 类（站/坐/跌/走/拳/跳/左挥手/左倾/张臂/右挥手/右倾/蹲） |
| 数据量 | 252 个 Excel 文件 → 445,474 帧 |
| 点云密度 | 10-50 点/帧，均值 31 点 |
| 预处理 | TI 多帧融合聚类（数据提供方完成） |
| 来源 | [ScienceDB](https://www.scidb.cn/detail?dataSetId=4c0cbfe8349e4a0ca19a73dc165be658) |

## 模型架构

```
毫米波点云 (N, 5)                 ← x, y, z, velocity, intensity
    ↓
PointNet++ Backbone               ← 3 层 Set Abstraction, 1.5M params
    ↓ (1024-d per frame)
Temporal ConvNet                  ← 4 层因果空洞卷积, 感受野 15 帧
    ↓ (256-d S_t)
TinyPredictor (GRU)               ← 1.3M params, 预测 Ŝ_{t+1}
    ↓
prediction_error = 1 - cos(Ŝ_t, S_t)
```

总参数量：~4.7M（Encoder）+ 1.3M（Predictor）= 6M

## 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW, lr=1e-4, cosine 退火 |
| Batch size | 32 |
| Encoder epochs | 200（early stopping patience=30） |
| Predictor epochs | 50 |
| 序列窗口 | 32 帧（~2s @ 20fps） |
| 损失函数 | InfoNCE + 0.05×CrossEntropy + 0.1×MSE_smooth + 0.01×Var_reg |
| 验证 | 留一法，stride=32，非重叠窗口 |

## 环境

| 项目 | 规格 |
|------|------|
| GPU | RTX 4090D 24GB |
| PyTorch | 2.8.0 / CUDA 12.8 |
| Python | 3.12 / Ubuntu 22.04 |
| 平台 | AutoDL（~2 元/h） |
| Encoder 训练耗时 | ~4h（200 epochs） |
| Predictor 训练耗时 | ~30min（50 epochs） |
| 总 GPU 成本 | ~80 元 |

## 复现步骤

```bash
# 1. 环境
pip install torch numpy scipy scikit-learn tqdm openpyxl

# 2. 下载数据集（FTP, ~500MB）
# ftp.scidb.cn:2121, 注册获取账号
# 下载 Ti雷达数据集/静态点云滤除+融合聚类/ 目录

# 3. 解析 Excel → 标准化 npz
python ai/data/public/parse_3dpchm.py \
  --raw-dir raw/mmwave_3dpchm/Ti雷达数据集/静态点云滤除+融合聚类 \
  --out 3dpchm_frames.npz

# 4. Encoder 训练（~4h）
python ai/encoder/train_encoder.py \
  --data 3dpchm_frames.npz --device cuda --epochs 200

# 5. 导出 TCN 隐状态（~30min）
# 见 ai/predictor/train_predictor.py 文档

# 6. Predictor 训练 + 评估（~30min）
python ai/predictor/train_predictor.py \
  --latent latent_tcn.npz --device cuda

# 输出 H3/H4 完整评估结果
```

## 关键源码

| 模块 | 路径 |
|------|------|
| PointNet++ Backbone | `ai/encoder/pointnet_backbone.py` |
| Temporal ConvNet | `ai/encoder/tcn.py` |
| Encoder 模型 | `ai/encoder/encoder_model.py` |
| 训练脚本 | `ai/encoder/train_encoder.py` |
| Predictor 模型 | `ai/predictor/model.py` |
| Predictor 训练+评估 | `ai/predictor/train_predictor.py` |
| 数据解析 | `ai/data/public/parse_3dpchm.py` |
| PyTorch Dataset | `ai/data/public/dataset.py` |

## 模型产出

| 文件 | 大小 | 用途 |
|------|------|------|
| `encoder_best.pt` | ~20MB | Encoder 权重，可 ONNX 导出 |
| `predictor_best.pt` | ~5MB | Predictor 权重 |
| `3dpchm_frames.npz` | 79MB | 预处理训练数据 |
