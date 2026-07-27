# 仿真系统

可控场景生成 → 自动标注 → 双通道注入 → AI 评估闭环。

## 快速开始

```bash
# 列出可用场景
python3 -m simulator.cli list

# 运行场景（仅输出 ground truth，不注入）
python3 -m simulator.cli run scenarios/elderly_day_normal.yaml --out output/

# 运行场景 + Redis 直写（AI 快速实验）
python3 -m simulator.cli run scenarios/elderly_day_normal.yaml \
  --redis redis://localhost:6379/0 --out output/

# 运行场景 + WebSocket 全链路
python3 -m simulator.cli run scenarios/elderly_day_normal.yaml \
  --ws ws://localhost:8000/ws/ingest --device rad_test --secret test_secret

# 时间加速（10x）
python3 -m simulator.cli run scenarios/elderly_day_normal.yaml --speed 10 --out output/
```

## 校准（首次使用前）

```bash
# 从 3DPCHM 数据集拟合 GMM 参数
python3 -m simulator.cli calibrate \
  --dataset ai/data/public/processed/3dpchm_frames.npz \
  --out simulator/calibrated_params.npz
```

校准后运行场景时指定参数：

```bash
python3 -m simulator.cli run scenarios/elderly_day_normal.yaml \
  --params simulator/calibrated_params.npz --out output/
```

## 模块结构

| 模块 | 文件 | 职责 |
|------|------|------|
| 类型定义 | `types.py` | FrameGroup, GroundTruth, VitalRecord, RoomConfig |
| 校准 | `calibrate.py` | 从 3DPCHM 拟合 12 动作类 GMM 参数 |
| 点云生成 | `generators.py` | SyntheticGenerator (GMM 采样) + ReplayGenerator (3DPCHM 回放) |
| 房间+体征 | `room_vitals.py` | RoomModel (坐标变换+噪声) + VitalSignsGen (生理模型) |
| 异常注入 | `anomaly.py` | fall/stillness/vital_anomaly/offline 四种异常 + 标注 |
| 场景引擎 | `scenario.py` | YAML 解析 + 时间线编排 |
| 双通道注入 | `injectors.py` | RedisInjector + WSInjector |
| CLI | `cli.py` | list/run/calibrate 命令 |
| 自检 | `validate.py` | 仿真数据统计检查 |

## 场景文件格式

见 `scenarios/elderly_day_normal.yaml` 示例。

## 测试

```bash
python3 -m pytest simulator/tests/test_simulator.py -v
```

## 与旧版 feed.py 的关系

`feed.py` 保留用于快速手动测试（单模式循环灌数）。`cli.py run` 替代其场景化需求。
