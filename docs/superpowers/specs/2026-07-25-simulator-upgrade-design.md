# 仿真系统升级 — 设计文档

## 背景

当前 `simulator/feed.py`（106 行）是硬编码的简单脚本：随机包围盒点云 + 随机体征 + 固定 2s 间隔循环。没有场景、没有过渡、没有标注、不可配置。

**目标：** 将仿真系统从"灌数脚本"升级为 AI 验证基础设施——可控场景生成 → 自动标注 → 双通道注入 → AI 评估闭环。

## 核心设计原则

1. **统计真实，非物理真实：** 点云不需要从雷达方程推导。只要统计分布（点密度、空间散布、速度相关性）与真实数据不可区分，AI encoder 就会产生相似的隐空间表示
2. **精确标注 > 完美仿真：** 仿真点云稍有偏差可以容忍；ground truth 时间/类型标记必须精确，否则 AI 评估不可信
3. **用真实数据校准合成模型：** 3DPCHM 数据集有 445K 帧真实雷达点云。用它们拟合每个姿态的统计分布，而非猜参数

## 架构总览

```
                        ┌──────────────────────┐
                        │   Scenario YAML       │
                        │   时间线 + 参数定义     │
                        └──────┬───────────────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │ Synthetic  │   │  Replay    │   │  Anomaly   │
       │ Generator  │   │ Generator  │   │  Injector  │
       │ (GMM 采样) │   │ (3DPCHM)  │   │ (精确注入)  │
       └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
             └───────────────┼───────────────┘
                             ▼
                   ┌──────────────────┐
                   │  Room Model       │  ← 坐标系变换 + 噪声
                   │  Vital Signs Gen  │  ← 生理模型
                   └────────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │  Contract         │  ← FrameGroup → JSON
                   │  Formatter        │
                   └────────┬─────────┘
                            ▼
            ┌───────────────┴───────────────┐
            ▼                               ▼
     WS Injector                     Redis Injector
     /ws/ingest 全链路               XADD stream 直写
     (端到端验证)                    (AI 快速实验)
            │                               │
            ▼                               ▼
     ┌─────────────┐              ┌─────────────────┐
     │  Backend     │              │  Redis Stream    │
     │  ingest→AI   │              │  + GT meta key   │
     └─────────────┘              └─────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                   ┌──────────────────┐
                   │  Ground Truth     │  ← 写入 output/
                   │  JSONL            │
                   └──────────────────┘
```

## 模块设计

### 1. 点云生成器

#### 1a. 校准流程（离线，一次性）

```
python -m simulator.calibrate --dataset 3dpchm_frames.npz --out simulator/calibrated_params.npz
```

对 3DPCHM 的 12 个动作类别，逐类拟合：

| 参数 | 方法 | 说明 |
|------|------|------|
| 空间分布 | GMM (K=3-5) | 在 (x,y,z) 空间拟合，捕获人体多反射点簇 |
| 点数分布 | Poisson 拟合 λ | log(λ_posture) |
| 速度分布 | Gaussian per posture | μ_v, σ_v |
| 强度分布 | Beta 分布 | α, β per posture |
| 帧间位移 | AR(1) 过程 | 质心位移自相关 |

**为什么用 GMM：** 人体对雷达呈现多个反射簇（头/躯干/四肢）。GMM 的 K 个分量天然对应这些簇，比单一高斯或均匀包围盒准确得多。

校准后每个姿态有一个参数文件，运行时加载到内存（~50KB），无需每次重算。

#### 1b. 合成生成器 (SyntheticGenerator)

运行时采样流程：

```python
def generate_frame(posture: str, prev_centroid: np.ndarray, dt: float) -> np.ndarray:
    gmm = params[posture].gmm           # 加载的 GMM
    n = sample_point_count(posture)     # Poisson 采样
    raw = gmm.sample(n)                 # (n, 3) GMM 采样点

    # 时间平滑：限制质心位移
    centroid = raw.mean(axis=0)
    max_disp = max_displacement[posture] * dt
    if np.linalg.norm(centroid - prev_centroid) > max_disp:
        centroid = prev_centroid + max_disp * direction

    raw += (centroid - raw.mean(axis=0))  # 移动到平滑位置

    # 速度：point-wise 位移 / dt
    velocity = compute_velocity(raw, prev_points) / dt

    # 强度：Beta 采样
    intensity = sample_intensity(posture, n)

    return np.column_stack([raw, velocity, intensity])  # (n, 5)
```

#### 1c. 回放生成器 (ReplayGenerator)

从 3DPCHM 直接取真实帧序列：

```python
def generate_sequence(action: str, duration_s: float, subject_ids: list[int] | None) -> Iterator[np.ndarray]:
    # 找到匹配的帧序列
    candidates = dataset[(dataset.action == action) & (dataset.subject.isin(subjects))]
    # 随机选一个起点，按时间顺序 yield 帧
    # 如果 duration 超过序列长度，循环播放（加随机扰动避免完全重复）
```

日常场景（stand/sit/walk）用真实数据，异常场景（fall）可用合成生成精确控制。

### 2. 房间模型

```
RoomConfig:
  size: [4.0, 3.0, 2.8]    # 长宽高（米）
  radar_pos: [2.0, 2.5, 2.5]  # 雷达安装位置（天花板墙角）
  noise:
    sigma: 0.02              # 加性高斯噪声标准差（米）
    dropout_rate: 0.01       # 随机丢点率
```

对每帧点云施加：
1. 坐标平移（世界坐标系 → 雷达坐标系）
2. 墙壁裁剪（房间外的点丢弃）
3. 高斯噪声（σ=2cm，对应 IWR6843 距离分辨率）
4. 随机丢点（模拟检测不稳定）

### 3. 体征生成器 (VitalSignsGen)

基于活动状态+心率变异性模型的生理信号生成：

```python
# 基础参数
RESTING_HR = 72     # bpm
RESTING_RR = 16     # bpm
HRV_SDNN = 50       # ms（正常成人 50±20）

def generate_vitals(activity: str, t: float) -> tuple[float, float, float]:
    # 活动相关心率升高
    activity_mult = {"lying": 0.9, "sitting": 1.0, "standing": 1.05, "walking": 1.3, "fall": 1.6}
    hr_base = RESTING_HR * activity_mult[activity]

    # 叠加呼吸性窦性心律不齐 + 随机波动
    hr = hr_base + 3 * sin(2π * RESTING_RR/60 * t) + normal(0, 3)

    # 呼吸率：活动相关
    rr_mult = {"lying": 1.0, "sitting": 1.0, "standing": 1.1, "walking": 1.5, "fall": 2.0}
    rr = RESTING_RR * rr_mult[activity] + normal(0, 1)

    # 质量：合成数据天然高，数据集回放可能低
    quality = 0.95 + normal(0, 0.02)

    return hr, rr, quality
```

**对照关系：** 活动强度 → 心率↑ + 呼吸率↑ + 心率变异性↓，符合真实生理规律。

### 4. 异常注入器 (AnomalyInjector)

在时间线精确位置注入异常，同时产出标注：

```python
ANOMALY_TEMPLATES = {
    "fall": {
        "transition_duration": 1.0,   # 秒
        "end_posture": "lie",
        "velocity_spike": True,        # 点速度突变
        "height_drop_rate": 1.0,       # m/s
    },
    "stillness": {
        "min_duration": 1800,          # 30 分钟
        "max_displacement": 0.05,      # 质心位移阈值（米）
    },
    "vital_anomaly": {
        "hr_range": (40, 130),         # 超出正常的心率范围
        "rr_range": (5, 35),
        "ramp_duration": 60,           # 逐步恶化（秒）
    },
    "offline": {
        "gap_duration": 120,           # 断连秒数
    },
}
```

注入逻辑：
- **跌倒：** 取时间线上注入点前后帧，插入过渡序列（质心 z 快速线性下降，结束姿态=lie）
- **静止：** 不改变点云，仅修改 ground truth 标注，标记 `anomaly_type=stillness`
- **体征异常：** 修改 VitalSignsGen 输出，逐步 ramp 到异常范围
- **离线：** 在时间线上插入空白（不发帧）

### 5. FrameGroup 内部协议

所有生成器输出统一的 `FrameGroup`：

```python
@dataclass
class FrameGroup:
    frame_id: str           # 全局唯一帧 ID
    ts: int                 # UTC ms
    device_id: str
    room: str
    # 点云
    points: np.ndarray      # (N, 5) float32 [x,y,z,velocity,intensity]
    # 体征（可选，VitalSignsGen 产出）
    vitals: VitalRecord | None
    # 标注（AnomalyInjector 产出）
    gt: GroundTruth | None

@dataclass
class GroundTruth:
    frame_id: str
    ts: int
    posture: str                      # 实际姿态
    posture_confidence: float         # 1.0（合成） 或 0.7-0.95（回放）
    heart_rate_true: float | None
    resp_rate_true: float | None
    anomaly_type: str | None          # "fall" | "stillness" | "vital_anomaly" | "offline" | None
    anomaly_severity: str | None      # "info" | "warning" | "critical"
    anomaly_start: bool
    anomaly_end: bool
    scenario_name: str
    generator: str                    # "synthetic" | "replay" | "hybrid"
    params_snapshot: dict             # 生成参数快照（复现用）
```

### 6. 双通道注入

```python
class WSInjector:
    """WebSocket 注入器 — 走 /ws/ingest 全链路"""
    async def inject(self, frames: Iterator[FrameGroup], url: str, device_id: str, secret: str):
        async with websockets.connect(url) as ws:
            await self._auth(ws, device_id, secret)
            for fg in frames:
                await self._send_point_cloud(ws, fg)
                if fg.vitals:
                    await self._send_vital(ws, fg)
                # 等待 ack，记录延迟
                ack = await ws.recv()

class RedisInjector:
    """Redis Stream 直写 — 跳过 ingest，直接到 AI 消费端"""
    def inject(self, frames: Iterator[FrameGroup], redis_url: str, stream_key: str):
        r = redis.Redis.from_url(redis_url)
        for fg in frames:
            data = frame_group_to_redis_dict(fg)
            msg_id = r.xadd(stream_key, data, maxlen=10000)
            # 标注写到并行 key
            if fg.gt:
                r.set(f"gt:{fg.frame_id}", fg.gt.to_json())
```

WS 通道用于验证全链路延迟和可靠性，Redis 通道用于 AI 快速迭代（跳过 ingest 鉴权和校验开销）。

### 7. CLI 入口

```bash
# 列出可用场景
python -m simulator.cli list

# 运行场景（双通道注入）
python -m simulator.cli run scenarios/morning_normal.yaml \
  --device rad_test \
  --secret test_secret \
  --ws ws://localhost:8000/ws/ingest \
  --redis redis://localhost:6379/0 \
  --out output/

# 校准参数（首次使用前运行一次）
python -m simulator.cli calibrate \
  --dataset ai/data/public/processed/3dpchm_frames.npz \
  --out simulator/calibrated_params.npz

# 批量实验 — 参数扫描
python -m simulator.cli batch scenarios/ --param noise.sigma 0.01 0.03 0.05 --runs 5
```

## 场景文件格式

```yaml
# scenarios/elderly_day_normal.yaml
name: "老人日常 - 正常全天"
version: 1
device_id: rad_elderly_01
rooms:
  bedroom:
    size: [4.0, 3.5, 2.8]
    radar_pos: [2.0, 3.0, 2.5]
  living:
    size: [5.0, 4.0, 2.8]
    radar_pos: [2.5, 3.5, 2.5]

vitals:
  resting_hr: 68
  resting_rr: 15

generator_defaults:
  method: synthetic    # synthetic | replay | hybrid
  fps: 10              # IWR6843 典型帧率

annotations:
  output: output/      # ground truth 落盘目录
  format: jsonl

timeline:
  # 7:00 起床
  - at: 0s
    room: bedroom
    activity: lying
    duration: 120s
    generator: synthetic   # 数据集无 lying 动作，用合成

  - at: 120s
    room: bedroom
    activity: sitting
    duration: 30s
    transition: {from: lying, to: sitting, duration: 3s}

  # 7:02 走向客厅
  - at: 150s
    room: bedroom
    activity: walking
    duration: 10s
    path: [[2,2,0], [2,0.5,0]]
    transition: {from: sitting, to: standing, duration: 2s}
    generator: replay    # 用真实 walk 数据

  # 7:03-8:00 客厅沙发坐着
  - at: 162s
    room: living
    activity: sitting
    duration: 3400s
    generator: replay

  # 8:00 去厕所
  - at: 3562s
    room: living
    activity: walking
    duration: 15s
    path: [[2,0.5,0], [3,0.5,0]]

  # ... （后续省略，实际场景会覆盖 24h）

# 异常注入（可选，覆盖在时间线上）
injections:
  - at: 39600s          # 11:00
    type: fall
    room: living
    params:
      height_drop_rate: 0.8
    ground_truth:
      severity: critical
      desc: "客厅行走时滑倒"

  - at: 50400s          # 14:00
    type: vital_anomaly
    room: bedroom
    duration: 300s
    params:
      hr_range: [45, 55]
      rr_range: [8, 12]
    ground_truth:
      severity: warning
      desc: "午睡时心率过低"
```

## 仿真有效性自检

在用于 AI 评估前，仿真数据必须通过自检：

```python
def validate_simulation(frames: list[FrameGroup]) -> dict:
    """对生成数据跑统计检查"""
    checks = {}

    # 1. 点数范围：每帧 5-100 点（真实雷达范围）
    n_points = [len(f.points) for f in frames if f.points is not None]
    checks["point_count_range"] = (min(n_points) >= 5 and max(n_points) <= 100)

    # 2. 坐标范围：x,y 在房间内，z 在 0-2m（人体高度）
    all_z = np.concatenate([f.points[:, 2] for f in frames if f.points is not None])
    checks["z_range"] = (all_z.min() >= -0.1 and all_z.max() <= 2.2)

    # 3. 时序连续性：相邻帧时间间隔在 80-120ms（10fps 附近）
    ts_diffs = np.diff([f.ts for f in frames])
    checks["temporal_continuity"] = (ts_diffs.min() > 50 and ts_diffs.max() < 150)

    # 4. 姿态一致性：标注姿态与点云 z 分布匹配
    for f in frames:
        if f.gt and f.points is not None:
            z_mean = f.points[:, 2].mean()
            expected = EXPECTED_Z[f.gt.posture]
            if not (expected[0] <= z_mean <= expected[1]):
                checks.setdefault("posture_z_consistency", []).append(f.frame_id)

    return checks
```

这些检查作为 CI 测试运行，确保仿真更新没有破坏数据质量。

## 与 AI 实验的衔接

仿真产出的 `frames.jsonl` + `ground_truth.jsonl` 对可以直接用于 AI 评估：

```python
# 评估脚本示例
from simulator.ground_truth import load_ground_truth
from ai.shared.redis_client import RedisStreamConsumer

# 1. 启动仿真注入 Redis
injector = RedisInjector(...)
injector.inject(scenario.frames, redis_url, stream_key)

# 2. AI pipeline 消费（已有代码）
consumer = RedisStreamConsumer(stream_key, consumer_group)
for frame in consumer:
    result = world_model.process(frame)  # encoder → predictor → decoders

# 3. 对比 ground truth
gt = load_ground_truth("output/2026-07-25_fall_test/ground_truth.jsonl")
for anomaly in gt.anomalies():
    detected = find_matching_alert(anomaly.ts, alerts)
    if detected:
        latency = detected.ts - anomaly.ts
        # 记录 metrics: 召回+1, 延迟=latency
    else:
        # 记录: 漏报+1
```

## 实现计划（高优先级优先）

| 阶段 | 模块 | 产出 | 预估投入 |
|------|------|------|---------|
| 1 | `calibrate.py` + `GMMParams` | 从 3DPCHM 拟合 12 动作的统计参数 | 200-300 行 |
| 2 | `SyntheticGenerator` | 基于校准参数的运行时点云生成 | 150-200 行 |
| 3 | `ReplayGenerator` | 3DPCHM 帧序列回放 + 变换 | 100-150 行 |
| 4 | `VitalSignsGen` + `RoomModel` | 体征生成 + 房间坐标系 | 80-120 行 |
| 5 | `AnomalyInjector` + `GroundTruth` | 异常注入 + 标注产出 | 120-150 行 |
| 6 | `ScenarioEngine` + YAML parser | 时间线编排引擎 | 150-200 行 |
| 7 | `RedisInjector` + `WSInjector` | 双通道注入 | 100-150 行 |
| 8 | `cli.py` + 场景库 | CLI + 5-8 个预定义场景 | 200-300 行 |
| 9 | 自检 + 集成测试 | 统计检查 + 端到端验证 | 100-150 行 |

**总计：~1,200-1,700 行新代码**，分 9 步交付，每步可独立测试。

## 风险与简化

| 风险 | 缓解 |
|------|------|
| GMM 拟合效果不好（某些姿态点云过于分散） | 增加 K 值或回退到核密度估计（KDE），优先验证 stand/sit/walk/fall 四个关键类 |
| 3DPCHM 缺少 lying 动作 | 用合成生成 lying（已有 squat 的近地姿态可以参考 z 分布） |
| 时序平滑不够，相邻帧看起来不连续 | AR(2) 模型替换 AR(1)，加人体运动学约束（加速度上限） |
| 仿真数据与真实数据统计差异大 | 跑 MMD（最大均值差异）作为定量指标，不合格则调参 |

**刻意不做的事：**
- 不模拟 IWR6843 射频链路（天线方向图、FMCW 波形）——成本极高，对 AI 验证无边际收益
- 不做多人仿真——独居假设是 MVP 硬约束
- 不做跨房间切换的连续模拟——每房间独立场景即可覆盖
- 不生成 phase signal 原始数据——体征用 VitalSignsFrame 注入，不走 extractor pipeline
