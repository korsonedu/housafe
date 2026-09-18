# 家安 AI 算法 —— 效果测试与手动学习指南

> 算法文档 `家安_核心算法设计文档_v2.md` Phase A 工程计划 `docs/superpowers/plans/2026-07-18-phase-a-algorithms.md`

---

## 一、当前 AI 架构概览

```
当前 (placeholder v0)                  Phase A 目标 (纯 Python 无 GPU)
─────────────────────                 ──────────────────────────────────
点云帧 → if-else 高度规则             点云帧 → FeatureVector(27维)
      → random 体征生成                      ↓
      → POST 回调                  ┌───────┼───────┐
                                   ↓       ↓       ↓
                              Fallback  Personal  Spatial
                              Engine    Baseline  Graph
                              (规则引擎) (GMM基线) (空间拓扑)
                                   ↓       ↓       ↓
                                   └───→ AnomalyDecoder ←──┘
                                            ↓
                                      NotificationDecider
                                            ↓
                                      分级通知(call/sms/push/none)
```

**为什么叫"手动学习"？** 当前阶段没有 GPU 训练、没有反向传播。学习 = 规则引擎的统计积累（空间图聚类 / 个人基线 GMM 拟合），都是在线增量学习，不需要离线训练。

---

## 二、各组件如何评估效果

### 2.1 Fallback 规则引擎

**核心指标：误报率 (FPR) 和漏报率 (FNR)**

| 指标 | 怎么测 | 当前基准 |
|------|--------|----------|
| 跌倒准确率 | 注入 100 帧 walk→fall 序列，统计触发次数 | ≥ 95% 快速跌倒触发，缓慢躺下不触发 |
| 静止误报 | 卧室躺卧 8h → 不应触发 stillness | 0 次告警 |
| 静止漏报 | 卫生间躺卧 10min → 必须触发 | 1 次 warning 或 critical |
| 体征误报 | 正常心率范围（60-100）→ 不触发 | 0 次告警 |

**手动测试方法：**

```bash
# 用不同场景的 FeatureVector 序列喂给 FallbackEngine.evaluate()
cd /Users/eular/Desktop/housafe
python -m pytest ai/tests/test_fallback.py -v

# 看具体断言：
# - test_height_drop_triggers_fall: 高度骤降 0.8m → 2s 内 → 跌倒
# - test_slow_lie_down_no_fall: 缓慢躺下 10 帧 → 不触发（非跌倒）
# - test_stillness_in_bathroom_triggers: 卫生间躺 60 帧 → 触发
# - test_stillness_in_bedroom_ignored: 卧室躺 60 帧 → 不触发
```

**扩展测试场景（建议补充）：**

```python
# 边界场景：从床到卫生间再回到床
# 混合场景：站立→缓慢坐下→躺卧（正常就寝）
# 真实数据模拟：用不同时间段的特征分布
```

---

### 2.2 个人基线 GMM

**这是真正意义上的"学习"组件——从 14 天历史数据中建模正常行为。**

#### 评估指标

| 层级 | 指标 | 怎么算 |
|------|------|--------|
| **单帧** | 异常分 (anomaly score) | Mahalanobis 距离在该上下文的 percentile → 0~1 |
| **时间窗口** | ROC-AUC | 在标注了"正常/异常"的测试集上计算 |
| **上线条件** | 14 天数据后 is_ready=True | `PersonalBaseline.is_ready` |
| **稳定性** | 连续 3 天同上下文异常分方差 < 0.1 | 新基线不震荡 |

#### 怎么量化"学好没有"

```
学习前 (Day 0-6):
  - is_ready = False（数据不足）
  - 所有异常分返回 0（不参与决策）
  - 全靠 Fallback 规则引擎兜底

学习完成 (Day 14+):
  - is_ready = True
  - 每个上下文 (10 个) 的 GMM 有 ≥ 200 帧样本
  - 正常行为 → 异常分 < 0.3
  - 异常行为 → 异常分 > 0.7
  - 正常/异常分离度 > 0.4（正常分第 95 百分位 < 异常分第 5 百分位）
```

**手动测试方法：**

```bash
# 1. 跑现有测试
python -m pytest ai/tests/test_baseline.py -v

# 2. 自己构造"14 天数据"看学习效果
python << 'PYEOF'
import numpy as np
from ai.shared.types import FeatureVector, time_encode
from ai.baseline.gmm_baseline import PersonalBaseline, context_key

def make_fvs(n, posture="sit", hr=72, rr=16, hour=12, weekday=0):
    rng = np.random.RandomState(42)
    fvs = []
    for i in range(n):
        h_sin, h_cos = time_encode(hour + i * 2 / 60)
        fvs.append(FeatureVector(
            ts=1000 + i*2000, device_id="r1", room="bedroom",
            posture=posture, posture_confidence=0.9, presence=True,
            moving=False, centroid=(1.0+rng.normal(0,0.02), 1.0+rng.normal(0,0.02), 0.5),
            height=0.8, n_points=15, occupancy_estimate=1,
            resp_rate=rr+rng.normal(0,1), heart_rate=hr+rng.normal(0,3),
            vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
            weekday=weekday, velocity_variance=0.01,
        ))
    return fvs

# 模拟 14 天多时段数据
baseline = PersonalBaseline()
all_fvs = []
for day in range(14):
    for h in [8, 12, 18, 22]:
        fvs = make_fvs(50, hour=h, weekday=day%7)
        all_fvs.extend(fvs)
baseline.fit(all_fvs)

print(f"学习完成: {baseline.is_ready}")
print(f"上下文数: {len(baseline._context_gmms)}")

# 测试：正常数据异常分
normal = make_fvs(1, hour=12)[0]
print(f"正常帧 异常分: {baseline.score(normal):.3f}")  # 应 < 0.3

# 测试：异常数据异常分
anomaly = make_fvs(1, hour=12, hr=140, posture="lie")[0]
print(f"异常帧 异常分: {baseline.score(anomaly):.3f}")  # 应 > 0.7

# 分离度
norm_scores = [baseline.score(make_fvs(1, hour=12)[0]) for _ in range(20)]
anom_scores = [baseline.score(make_fvs(1, hour=12, hr=140, posture="lie")[0]) for _ in range(20)]
p95_norm = np.percentile(norm_scores, 95)
p5_anom = np.percentile(anom_scores, 5)
print(f"正常P95={p95_norm:.3f} 异常P5={p5_anom:.3f} 分离度={p5_anom-p95_norm:.3f}")
# 分离度 > 0.4 才算合格
PYEOF
```

---

### 2.3 空间图 (SpatialGraph)

**从位置轨迹中自动学习房间功能分区。这是无监督聚类——没有标注数据。**

#### 评估指标

| 指标 | 含义 | 怎么测 |
|------|------|--------|
| 节点数收敛 | 14 天后节点数稳定 | 连续 3 天节点数不变 |
| 节点语义可解释 | 每个节点对应一个房间区域 | 手动标注对照 |
| 定位准确率 | `locate(pos)` 返回正确的区域 | 用 walk→sit→lie 等模式验证 |
| 边密度合理 | 频繁 A→B→A 的节点对之间应有边 | 看 transfer 模式是否被捕捉 |
| 持久化正确 | save/load 后节点数和分数一致 | 单元测试已有 |

**手动测试方法：**

```bash
# 跑现有测试
python -m pytest ai/tests/test_graph.py -v

# 用模拟器数据验证空间图学习
# 思路：让模拟器在多个 mode 之间切换，观察质心变化 → 空间图能否自动聚类出"床区域"和"门区域"
```

```python
# 手动喂轨迹数据
python << 'PYEOF'
import numpy as np
from ai.latent_space.graph import SpatialGraph

g = SpatialGraph(min_stay_frames=30)

# 模拟 7 天轨迹：床(大量时间) + 卫生间(中量) + 厨房(少量) + 路过(极少)
positions = []
timestamps = []
ts = 0

for day in range(7):
    for hour in range(24):
        if 22 <= hour or hour <= 6:
            # 夜间：床上 (低 z，极低方差)
            pts = np.tile(np.array([0.5, 1.0, 0.2]), (len(positions) and 60 or 3600, 1))  # 这行有问题，直接写逻辑
            pass

# 简化版：直接生成连续轨迹
# 床区域 (night, 0.2m 高度, 占地 60%)
bed = np.tile(np.array([0.5, 1.0, 0.2]), (5000, 1)) + np.random.normal(0, 0.05, (5000, 3))
# 卫生间 (morning, 0.8-1.2m, 占地 20%)
bath = np.tile(np.array([3.0, 0.5, 0.5]), (2000, 1)) + np.random.normal(0, 0.1, (2000, 3))
# 客厅 (day, 1.0-1.8m, 占地 20%)
living = np.tile(np.array([2.0, 3.0, 1.5]), (2000, 1)) + np.random.normal(0, 0.15, (2000, 3))

all_pos = np.vstack([bed, bath, living]).astype(np.float32)
all_ts = np.arange(len(all_pos), dtype=np.int64) * 2000  # 每 2s 一帧

g.update(all_pos, all_ts)

print(f"学习到 {len(g.nodes)} 个空间区域")
for nid, node in g.nodes.items():
    print(f"  节点 {nid}: centroid=({node.centroid[0]:.1f}, {node.centroid[1]:.1f}, {node.centroid[2]:.1f}), "
          f"stay_ratio={node.stay_ratio:.2%}, risk={node.risk_score:.2f}, "
          f"高度={node.avg_height:.2f}m")

# 验证: locate 一个床边的位置应返回 node 0
bed_near = g.locate((0.6, 1.1, 0.25))
bath_near = g.locate((2.9, 0.6, 0.55))
print(f"床附近 → node {bed_near}")
print(f"卫生间附近 → node {bath_near}")
# 这两个应不同
PYEOF
```

---

### 2.4 通知决策 (NotificationDecider)

**评估指标：通知的"适度性"——不骚扰也不遗漏。**

| 场景 | 预期 | 指标 |
|------|------|------|
| 跌倒 critical | → call | 召回率 100% |
| 跌倒 warning | → sms | |
| 卫生间静止 10min | → push | |
| 卫生间静止 5min + 重复 | → 抑制（none） | 抑制窗口内重复率 = 0 |
| Pattern deviation info | → push 或 none | |

```bash
python -m pytest ai/tests/test_notification.py -v
```

---

### 2.5 AI 整体效果评估方案（完整流程）

```
                    标注数据集（你需要构建的）
                    ────────────────────────
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
   正常时段样本     异常注入样本      边界样本
   (3天连续正常)   (人工制造异常)    (阈值附近)
         │               │               │
         └───────────────┼───────────────┘
                         ↓
              WorldModelWorker 处理
              (规则+基线+空间+通知)
                         ↓
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
   AnomalyResult     Notification     端到端延迟
   异常分对照标注    决策对照预期      (点云→推送)
```

**构建测试数据集的方法：**

1. **正常数据**：跑模拟器 `--mode walk` 持续 1h，记录所有 FeatureVector → 全标记为 "normal"
2. **异常注入**：
   - 跑 `--mode fall` 3 帧 → 标记 "fall"
   - 修改体征为心率 150 → 标记 "vital_anomaly"
   - 卫生间 mode=lie 持续 15min → 标记 "stillness"
3. **跑全量评估**：
   ```python
   for fv, label in test_set:
       result = world_model.evaluate(fv, history)
       if label == "normal" and result is not None:
           false_positives += 1
       elif label != "normal" and result is None:
           false_negatives += 1
   ```

---

## 三、如何手动进行学习

"学习"在 Phase A 中指的是两个过程：
1. **空间图聚类** — 从位置轨迹中 HDBSCAN 聚类
2. **个人基线拟合** — 从特征向量中 GMM 密度估计

两个都是在线增量学习，不需要离线训练脚本。

### 3.1 手动触发 SpatialGraph 学习

```bash
# 跑 AI worker（已集成图学习，自动从每帧 centroid 提取轨迹）
python ai/placeholder/main.py

# 在另一个终端灌入大量数据
python simulator/feed.py rad_e2e e2e_secret --mode walk

# 空间图在内存中自动更新。想看当前图的节点数：
python << 'PYEOF'
# 这需要用共享内存或 pickle 持久化后的文件
# 当前 Phase A 实现还在开发中，WorldModelWorker 会在内存中维护 graph 实例
import pickle
try:
    with open("/tmp/housafe_graph.pkl", "rb") as f:
        g = pickle.load(f)
    print(f"节点数: {len(g.nodes)}")
    for nid, node in g.nodes.items():
        print(f"  [{nid}] centroid={node.centroid} stay={node.stay_ratio:.2%}")
except FileNotFoundError:
    print("图尚未持久化，检查 AI worker 日志输出")
PYEOF
```

### 3.2 手动触发 PersonalBaseline 学习

**基线需要至少 14 天数据。** 开发时可以用模拟器加速：

```bash
# 1. 启动 AI worker（需要 WorldModelWorker 集成 baseline）
# 当前 placeholder 不支持基线，Phase A Task 6 实现后：

# 2. 模拟 14 天的多时段数据
# 用脚本循环模拟不同 day/hour 的特征向量，喂给 baseline.fit()

# 3. 检查学习状态
python << 'PYEOF'
from ai.baseline.gmm_baseline import PersonalBaseline
try:
    baseline = PersonalBaseline.load("/tmp/housafe_baseline.pkl")
    print(f"基线就绪: {baseline.is_ready}")
    print(f"上下文数: {len(baseline._context_gmms)}")
    for ctx in baseline._context_gmms:
        print(f"  {ctx}: {baseline._context_gmms[ctx].n_components} components")
except FileNotFoundError:
    print("基线尚未持久化（数据不足 14 天或 Phase A 未完成）")
PYEOF
```

### 3.3 学习流程完整脚本（Phase A 完成后可用）

```bash
#!/bin/bash
# learn_and_eval.sh — 手动学习 + 效果评估

# 前置：启动 daphne + Redis + AI worker

# Step 1: 灌入"正常数据"让基线学习
echo "=== 阶段1: 收集基线数据（模拟 7 天） ==="
for day in {0..6}; do
    echo "Day $day"
    # 早中晚各跑 5 分钟 walk 模式
    python simulator/feed.py rad_e2e e2e_secret --mode walk &
    SIM_PID=$!
    sleep 300  # 5 分钟
    kill $SIM_PID 2>/dev/null
done

# Step 2: 保存学习成果
echo "=== 阶段2: 持久化学习成果 ==="
python << 'PYEOF'
# 假设 WorldModelWorker 有 save() 方法
import requests
r = requests.post("http://localhost:8000/api/internal/save-models",
                   json={"token": "dev-internal-token"})
print(r.json())
PYEOF

# Step 3: 注入异常数据 — 验证检测效果
echo "=== 阶段3: 异常注入测试 ==="
# 跌倒场景
python simulator/feed.py rad_e2e e2e_secret --mode fall &
SIM_PID=$!
sleep 10  # 发 5 帧跌倒
kill $SIM_PID 2>/dev/null

# 查 Alert
python << 'PYEOF'
import os, sys
sys.path.insert(0, 'backend')
os.environ['DJANGO_SETTINGS_MODULE'] = 'housafe.settings'
import django; django.setup()
from events.models import Alert
recent = Alert.objects.order_by('-ts')[:5]
for a in recent:
    print(f"  {a.alert_type} | {a.severity} | room={a.room} | score={a.payload}")
PYEOF

echo "=== 学习+评估完成 ==="
```

---

## 四、ML 特有测试 Checklist

与普通代码测试不同，ML 系统需要额外关注：

```
□ 数据质量
  □ 模拟器点云 z 轴分布是否符合真实设备？
  □ FeatureVector 27 维各维度范围是否在合理区间？
  □ 缺少生命体征数据时（None），基线如何处理？

□ 线上/线下一致性
  □ 模拟器数据 vs 真实雷达数据，基线是否能泛化？
  □ 不同设备（rad_e2e, rad_01）的特征分布是否一致？

□ 概念漂移
  □ 季节变化（冬天多在室内 vs 夏天）→ 基线是否跟踪？
  □ 老人身体变化（恢复后心率模式不同）→ 基线是否自适应？

□ 决策可解释性
  □ 每条 Alert 的 details 字段是否有足够信息回溯？
  □ anomaly_score 从哪个组件来（source: fallback/baseline/both）是否明确？

□ 安全网完整性
  □ Fallback 规则全部关闭时，基线能否独立工作？
  □ 基线未就绪时，Fallback 是否正常工作？
  □ 两者都开时，融合逻辑是否正确（取 max）？

□ 持久化/恢复
  □ 进程重启后 baseline/graph 能否从 pickle 恢复？
  □ 恢复到之前的异常分是否一致（±0.01）？
```

---

## 五、演进路径：什么时候需要"真正的训练"

当前 Phase A 是纯规则 + 无监督统计。但设计文档明确规划了演进：

| 阶段 | 方法 | 需要的数据 |
|------|------|-----------|
| **现在 (Phase A)** | 规则引擎 + GMM + HDBSCAN | 在线增量，无需标注 |
| **Phase B** | PointNet++ Encoder → 替换规则抽取 | 需标注点云帧（姿态标签）|
| **Phase C** | Temporal Transformer → 替换 TCN | 需连续帧序列 |
| **Phase D** | 通知 Decoder MLP → 从用户反馈学习 | 需用户对通知的点击/忽略行为 |

到了 Phase B/C，就需要：
- 标注数据集（合成数据 + 真实设备采集）
- 训练脚本（PyTorch DataLoader + Trainer）
- 评估集 + 测试集拆分
- 离线训练 → 模型导出 → 在线推理

这条路径已在算法设计文档中规划，但当前阶段焦点是 **跑通全链路**，ML 复杂度在后续阶段加入。

---

## 六、快速参考命令

```bash
# ── 跑所有 AI 算法测试 ──
cd /Users/eular/Desktop/housafe
python -m pytest ai/tests/ -v

# ── 分组件测试 ──
python -m pytest ai/tests/test_types.py -v       # 数据类型
python -m pytest ai/tests/test_fallback.py -v     # 规则引擎（3 类规则，~7 cases）
python -m pytest ai/tests/test_baseline.py -v     # GMM 基线（~7 cases）
python -m pytest ai/tests/test_graph.py -v        # 空间图（~7 cases）
python -m pytest ai/tests/test_anomaly.py -v      # 异常解码器融合
python -m pytest ai/tests/test_notification.py -v # 通知决策

# ── 手动喂养数据 → 看学习效果 ──
# 见正文 3.1 / 3.2 节的 Python 片段

# ── 查看"模型"持久化状态 ──
ls -la /tmp/housafe_*.pkl 2>/dev/null || echo "尚未持久化"
```
