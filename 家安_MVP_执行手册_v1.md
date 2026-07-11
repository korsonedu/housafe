# 家安 housafe · MVP 执行手册

> 版本 v2.0 | 2026-07-10 | 硬件方案更新：IWR6843ISK 单板，无需 ICBOOST/DCA1000
> 目标：IWR6843ISK 开发板 → 采集点云 → 自研 ALG-0 模型 → A/B 精度对比 → 推理管线 → Web Dashboard
> 总工期：12 周（solo + AI 辅助，约 300-400 小时）
> 核心验证指标：自研点云感知模型能否在跌倒检测精度上超过 TI 固件

---

## 前置：你需要什么技能 / AI 帮你做什么

| 你要写的 | AI 帮你写的 |
|----------|------------|
| Python 点云采集脚本 | ✅ 大部分代码生成 + 调试建议 |
| 信号处理 (Range-Doppler FFT, CFAR) | AI 提示参数调整方向 |
| PyTorch 模型训练 | ✅ 数据加载/模型定义/训练循环 |
| 3D 可视化/标注工具 | ✅ matplotlib/plotly 代码 |
| Flask/FastAPI 推理服务 | ✅ Web Dashboard |
| TI mmWave SDK (C 代码) | ⚠️ TI 文档限定，AI 只能辅助理解 |
| 传感器架设/物理调试 | ❌ 你得自己爬梯子 |

---

# Phase 1：硬件采购 + 环境搭建（Week 1）

## 1.1 硬件采购清单

### 必买

| 品名 | 型号 | 用途 | 参考价 | 购买渠道 |
|------|------|------|--------|---------|
| 毫米波雷达开发板 | **IWR6843ISK** | 60GHz 雷达，3TX/4RX，AOP 封装。USB 直插电脑，板载 XDS110 调试器，串口直接输出点云 | ~$149 (~¥1,000) | [TI 官方](https://www.ti.com/tool/IWR6843ISK) 或淘宝搜 "IWR6843ISK" |
| micro-USB 线 | 高质量，长度 ≥ 2m | ISK → PC，供电+数据传输一根线搞定 | ¥15 | 买雷达时通常附赠 |
| 5V 2.5A 电源适配器 | 5.5mm-2.1mm DC 头 | 给 ISK 供电（USB 供电有时不够稳，建议独立供电） | ¥30 | 淘宝 |
| 三脚架/灯架 + 云台 | 高度可调 1-2m | 固定雷达，模拟天花板/墙壁安装 | ¥50-100 | 淘宝/京东 |
| 3M VHB 双面胶 | — | 临时固定 | ¥15 | 五金店 |
| 卷尺 | 5m | 标定采集区域 | ¥10 | — |

> **总预算：约 ¥1,200-1,500。** ISK 单板就够，不需要 ICBOOST 载板和 DCA1000 采集卡。ISK 自带 USB 转 UART，插电脑上电就能出点云。

### 可选

| 品名 | 用途 |
|------|------|
| 海凌科 LD6002（~¥50） | 快速验证雷达→云端→App 数据管道，不用于模型训练（不输出点云） |
| 瑜伽垫/缓冲垫 | 模拟跌倒保护自己 |
| GoPro/手机三脚架 | 录视频同步，辅助标注 |

---

## 1.2 软件环境搭建

### 安装清单

```
你的 PC 需要：
- Windows 10/11（mmWave SDK 和 CCS 只在 Windows 上稳定）
  - 如果你用 Mac，装 Parallels/VMware + Windows 11 ARM，或者找一台 Windows 笔记本
- Python 3.10+（Mac 上也装一份，分析点云用）
- 至少 50GB 空闲硬盘（录点云数据大）
- 推荐有 NVIDIA GPU（RTX 2060 以上），训练模型用。没有也能用 CPU 跑，就是慢
```

### Step 1: TI 工具链安装（Windows）

1. **下载 TI mmWave SDK 5.x**
   - 网址：https://www.ti.com/tool/MMWAVE-SDK
   - 选 IWR6843 对应版本（5.1.0.4 或更新）
   - 默认路径安装 `C:\ti\mmwave_sdk_05_01_00_04`

2. **下载 Code Composer Studio (CCS)**
   - 网址：https://www.ti.com/tool/CCSTUDIO
   - 版本 12.x（安装时勾选 mmWave 插件）

3. **下载 mmWave Industrial Toolbox**
   - 网址：https://dev.ti.com/tirex/explore/node?node=AIQqc9js4adANq.xuQdLGg__VLyFKFf__LATEST
   - 找到 `Out of Box Demo` 和 `Fall Detection` 两个 lab
   - Out of Box Demo 输出点云，是我们采集数据的主力固件
   - Fall Detection demo 用于 A/B 对比测试

4. **mmWave Demo Visualizer（网页版，无需安装）**
   - 网址：https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/4.3.0/
   - 直接浏览器打开，连接雷达串口即可看到实时 3D 点云

### Step 2: Python 环境搭建（Mac/Windows）

```bash
# 创建虚拟环境
python3 -m venv ~/housafe-env
source ~/housafe-env/bin/activate  # Windows: housafe-env\Scripts\activate

# 核心依赖
pip install numpy scipy matplotlib plotly jupyter notebook
pip install torch torchvision  # 选 CUDA 版本或 CPU 版本
pip install scikit-learn pandas h5py
pip install pyserial  # 串口通信，收雷达数据
pip install flask fastapi uvicorn  # 推理 server
pip install open3d  # 点云可视化

# 推荐装
pip install black isort pytest
pip install tqdm tensorboard  # 训练监控
```

### Step 3: 验证硬件链路

1. **连接硬件：**
   ```
   IWR6843ISK ←micro-USB→ PC（供电+数据传输一根线）
   （可选）ISK 的 DC 口 ←5V 电源→ 插座（USB 供电不稳时用）
   ```

2. **烧录 Out of Box Demo：**
   - 打开 CCS → Import Project → 选择 mmWave SDK 里的 Out of Box Demo
   - Build → Debug → 烧录到 ISK
   - 打开 mmWave Demo Visualizer 网页版
   - 选择串口（设备管理器里看 ISK 对应的 COM 口，有两个：选 "Application/User UART" 那个）、波特率 115200 → 连接
   - **验收标准：** 网页上出现 3D 点云散点图，能看到你的身体轮廓在动 ✅

3. **烧录 Fall Detection Demo（用于 A/B 对比）：**
   - 在 Industrial Toolbox 里找到 `Fall Detection` lab
   - 按 README 编译/烧录固件
   - 用串口终端查看输出
   - **验收标准：** 模拟跌倒时串口输出 `FALL_DETECTED` ✅

---

## 1.3 采集区域搭建

### 物理布局

```
         墙 壁
    ┌──────────────┐
    │              │
    │   3m × 3m   │  ← 采集区域，铺瑜伽垫
    │   空旷区域   │
    │              │
    │   ▲ 雷达    │  ← 装在三脚架上，高度 2.2m（模拟天花板）
    │   俯视 45°  │     或 1.2m（模拟墙壁安装）
    │              │
    └──────────────┘
```

### 安装注意事项

- 雷达前面不要有金属物品（会产生强烈多径反射）
- 采集区域内清空杂物，但保留一堵墙和一个椅子（真实环境必须有参照物）
- 采集时室内不要有其他移动的人/宠物
- 记录安装高度、角度、房间大致布局，后续数据分析会用到

---

# Phase 2：点云采集（Week 1-2，与 Phase 1 并行）

> **核心原则：拿到板子当天就开始录数据。**

## 2.1 从 ISK 串口读取点云

### 数据流原理（简化的）

```
雷达发送 FMCW chirp → 接收回波 → ISK 板载 DSP 做 Range-Doppler FFT + CFAR + AoA
                                         ↓
                                   生成点云 (x,y,z,v,i)
                                         ↓
                                   UART 串口（通过 USB 虚拟 COM 口）
                                         ↓
                                   Python 脚本读取 → 解析 → 保存 .jsonl
```

ISK 烧录 Out of Box Demo 固件后，点云已经由板载 DSP 算好了，通过串口以 TLV（Type-Length-Value）格式输出。我们不需要自己做 FFT/CFAR/AoA——省掉了整个信号处理管线。

### 第一步：用 mmWave Demo Visualizer 先验证

1. ISK 连电脑 USB → 烧录 Out of Box Demo
2. 打开 https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/
3. 连上串口 → 看到实时 3D 点云散点图
4. 在雷达前走动、坐下、躺下，确认点云正常跟踪

### 第二步：写 Python 脚本录制点云

TI 的 Out of Box Demo 通过串口输出 TLV 格式数据，包括点云帧、目标列表、存在信息等。Python 脚本解析这个协议，提取点云帧，保存为 .jsonl。

```python
# capture_pointcloud.py - 从 ISK 串口读取并保存点云

"""
ISK 烧录 Out of Box Demo后，串口输出 TLV 格式数据。
本脚本解析 TLV，提取点云帧，保存为 .jsonl。

TLV 协议参考：mmWave SDK docs 里的 "Mmwave Demo Data Structure"
关键 TLV types:
  - type=1: 检测到的目标点列表（每个点含 x,y,z,velocity）
  - type=2: 距离剖面
  - type=7: 目标跟踪信息
"""

import serial
import struct
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional


# ============ TLV 协议常量 ============

MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGETS = 7

# 帧头魔数（TI 固件固定值）
SYNC_MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'


@dataclass
class Point:
    x: float
    y: float
    z: float
    velocity: float
    intensity: float


@dataclass
class PointCloudFrame:
    ts: int
    frame_id: int
    num_points: int
    points: List[Point]


class ISKPointCloudReader:
    """从 IWR6843ISK 串口读取并解析点云帧"""

    def __init__(self, port: str = "COM3", cli_port: str = "COM4",
                 baudrate: int = 115200):
        # 数据端口（接收 TLV 输出）
        self.data_ser = serial.Serial(port, baudrate, timeout=1)
        # 命令端口（发送配置命令）
        self.cli_ser = serial.Serial(cli_port, baudrate, timeout=1)
        self.buffer = b''

    def send_config(self, config_path: str):
        """发送 .cfg 配置文件"""
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('%') and not line.startswith('//'):
                    self.cli_ser.write((line + '\n').encode())
                    time.sleep(0.01)
        print("Config sent.")

    def start(self):
        self.cli_ser.write(b'sensorStart\n')
        time.sleep(1)

    def stop(self):
        self.cli_ser.write(b'sensorStop\n')

    def read_frame(self) -> Optional[PointCloudFrame]:
        """读取一帧。返回 None 表示超时或无有效帧。"""
        # 读入缓冲区直到有足够的 header 字节
        header_size = 40  # 固定 40 字节帧头
        while len(self.buffer) < header_size:
            chunk = self.data_ser.read(max(1, header_size - len(self.buffer)))
            if not chunk:
                return None
            self.buffer += chunk

        # 搜索 sync magic
        idx = self.buffer.find(SYNC_MAGIC)
        if idx == -1:
            # 没找到，保留最后几个字节以防跨边界
            self.buffer = self.buffer[-7:]
            return None
        if idx > 0:
            self.buffer = self.buffer[idx:]

        if len(self.buffer) < header_size:
            return None

        # 解析帧头
        header = self.buffer[:header_size]
        _, _, _, _, frame_num, _, _, num_tlvs, _, _, _, _ = struct.unpack(
            'Q6I3H2I', header
        )

        # 读取 TLV 数据
        tlv_offset = header_size
        detections = []

        for _ in range(num_tlvs):
            if len(self.buffer) < tlv_offset + 8:
                # 数据不完整
                return None
            tlv_header = self.buffer[tlv_offset:tlv_offset + 8]
            tlv_type, tlv_len = struct.unpack('II', tlv_header)
            tlv_offset += 8

            if tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
                # 目标点列表
                point_data = self.buffer[tlv_offset:tlv_offset + tlv_len]
                point_size = 16  # 每点 16 字节 (4 floats)
                num_points = tlv_len // point_size
                for p_idx in range(num_points):
                    offset = p_idx * point_size
                    x, y, z, v = struct.unpack('4f', point_data[offset:offset+16])
                    detections.append(Point(
                        x=x, y=y, z=z,
                        velocity=v,
                        intensity=1.0  # Out of Box demo 不输出 intensity
                    ))
            tlv_offset += tlv_len

        # 清空已处理的缓冲区
        self.buffer = self.buffer[tlv_offset:]

        if not detections:
            return None

        return PointCloudFrame(
            ts=int(time.time() * 1000),
            frame_id=frame_num,
            num_points=len(detections),
            points=detections
        )


def record_session(
    label: str,
    duration_sec: float,
    config_file: str = "profile_20fps.cfg",
    data_port: str = "COM3",
    cli_port: str = "COM4",
    output_dir: str = "./data/raw"
):
    """录制一次采集 session"""
    reader = ISKPointCloudReader(port=data_port, cli_port=cli_port)
    reader.send_config(config_file)
    reader.start()

    output_path = Path(output_dir) / datetime.now().strftime("%Y%m%d")
    output_path.mkdir(parents=True, exist_ok=True)

    frames = []
    start_time = time.time()
    frame_count = 0
    target_frames = int(duration_sec * 20)  # 20fps

    print(f"Recording: {label} ({duration_sec}s)...")
    while time.time() - start_time < duration_sec and frame_count < target_frames:
        frame = reader.read_frame()
        if frame is not None:
            frames.append(frame)
            frame_count += 1

    reader.stop()

    # 保存
    save_path = output_path / f"{label}.jsonl"
    with open(save_path, 'w') as f:
        for frame in frames:
            f.write(json.dumps({
                "ts": frame.ts,
                "frame_id": frame.frame_id,
                "num_points": frame.num_points,
                "points": [{
                    "x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3),
                    "v": round(p.velocity, 2), "i": round(p.intensity, 2)
                } for p in frame.points]
            }) + '\n')

    print(f"Saved {len(frames)} frames → {save_path}")
    return str(save_path)


if __name__ == "__main__":
    # 查串口号：Windows 设备管理器 → 端口 → 找 "XDS110 Class Application/User UART"
    #   data_port = Application/User UART（数据口）
    #   cli_port  = XDS110 Class Auxiliary Data Port（命令口）
    record_session("test_walk_01", duration_sec=5,
                   data_port="COM3", cli_port="COM4")
```

### TI 参考 cfg 文件

保存为 `profile_20fps.cfg`：

```
% 家安 housafe - IWR6843ISK 点云采集配置
% 基于 TI Out of Box Demo，20fps
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60.25 30 10 50 0 0 50 1 256 10000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 1
chirpCfg 2 2 0 0 0 0 0 1
frameCfg 0 2 128 0 50 1 0
lowPower 0 0
guiMonitor 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 12 0
cfarCfg -1 1 0 4 2 3 1 12 0
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
```

---

## 2.2 采集脚本（动作录制 SOP）

### 跌倒类动作（重点）

每个动作录 **10 次**，从不同角度/方向。

| 编号 | 动作 | 方向 | 起始姿态 | 文件名格式 | 时长 |
|------|------|------|---------|-----------|------|
| F01 | 向前摔倒 | 正面朝向雷达 | 站立 | `fall_forward_01`~`10` | 5s |
| F02 | 向右侧摔倒 | 侧面朝向雷达 | 站立 | `fall_side_right_01`~`10` | 5s |
| F03 | 向左侧摔倒 | 侧面朝向雷达 | 站立 | `fall_side_left_01`~`10` | 5s |
| F04 | 向后摔倒 | 背对雷达 | 站立 | `fall_backward_01`~`10` | 5s |
| F05 | 从椅子上滑落 | 坐姿侧滑 | 坐椅子 | `fall_chair_slide_01`~`10` | 5s |
| F06 | 从床上滚落 | 躺姿→地面 | 躺瑜伽垫（高30cm台面） | `fall_bed_roll_01`~`10` | 5s |
| F07 | 缓慢瘫倒 | 站立→缓慢下滑 | 站立 | `fall_slow_collapse_01`~`10` | 8s |

> ⚠️ **安全提醒：** 练摔倒前铺好缓冲垫。膝盖微屈，用前臂缓冲。不需要真的狠狠摔——AI 学的是运动轨迹突变，不是冲击力。

### 假跌倒 / 易误报动作（关键区分）

这些是 TI 固件最容易误报的场景，也是自研模型的价值所在。

| 编号 | 动作 | 文件名格式 | 时长 |
|------|------|-----------|------|
| N01 | 快速坐下（"扑通"一声坐椅子上） | `nonfall_quick_sit_01`~`10` | 5s |
| N02 | 蹲下捡东西（站→蹲→站） | `nonfall_squat_pickup_01`~`10` | 5s |
| N03 | 弯腰系鞋带（站→弯腰→站） | `nonfall_bend_tie_01`~`10` | 5s |
| N04 | 快速躺下（模仿上床） | `nonfall_quick_lie_down_01`~`10` | 5s |
| N05 | 突然挥手/伸展（大范围手臂运动） | `nonfall_wave_01`~`05` | 3s |
| N06 | 拖地/扫地（弯腰+移动） | `nonfall_mopping_01`~`05` | 10s |
| N07 | 从椅子上快速站起 | `nonfall_quick_standup_01`~`05` | 3s |
| N08 | 拉窗帘/伸手够高处 | `nonfall_reach_high_01`~`05` | 5s |

### 日常动作基线

| 编号 | 动作 | 文件名格式 | 时长 |
|------|------|-----------|------|
| D01 | 正常行走（直线 3m 来回） | `daily_walk_01`~`10` | 10s |
| D02 | 站→坐椅子→站（正常速度） | `daily_sit_stand_01`~`10` | 8s |
| D03 | 躺下→起床（正常速度） | `daily_lie_getup_01`~`10` | 10s |
| D04 | 站在原地看着手机/发呆 | `daily_stand_idle_01`~`05` | 10s |
| D05 | 坐椅子上不动（看书/看手机） | `daily_sit_still_01`~`05` | 30s |
| D06 | 躺下不动（模拟睡眠） | `daily_lie_still_01`~`05` | 60s |
| D07 | 从卧室走进来→停留→走出 | `daily_enter_leave_01`~`05` | 10s |

### 安静态生命体征采集

| 编号 | 状态 | 文件名格式 | 时长 |
|------|------|-----------|------|
| V01 | 坐姿不动，正常呼吸 | `vital_sit_breathing_01`~`05` | 60s |
| V02 | 躺姿不动，正常呼吸 | `vital_lie_breathing_01`~`05` | 120s |
| V03 | 躺姿，深呼吸 | `vital_deep_breath_01`~`03` | 60s |
| V04 | 坐姿，屏住呼吸 10s（作为对照） | `vital_hold_breath_01`~`03` | 30s |
| V05 | 躺姿，同时戴手环/Apple Watch 记录心率和呼吸对照 | `vital_with_watch_01`~`03` | 120s |

### 采集协议（每次都要做）

```markdown
1. 架好 ISK，确认 Demo Visualizer 上点云稳定
2. 站在采集区域中间
3. 运行 Python 采集脚本：python capture_pointcloud.py --label fall_forward_01 --duration 5
4. 保持不动 1 秒
5. 执行动作
6. 执行完毕后保持不动 1 秒
7. 脚本自动停止
8. 【重要】立即记录到 logbook：文件名、动作描述、任何异常（如"这次摔得比较轻"、"雷达好像掉帧"）
```

总共约 **200+ 个样本文件**，预计需要 **4-8 小时** 不间断采集。分 2-4 天完成，每天做完一组后检查点云质量。

---

## 2.3 点云质量检查 Checklist

每做完一天采集，运行检查脚本：

```python
# check_pointcloud_quality.py - 架构草图

"""
对每个 .jsonl 文件做基本质量检查：
1. 帧数 ≥ 预期（5s × 20fps = 100 帧左右）
2. 每帧点数 > 0（雷达在正常工作）
3. 点云质心移动轨迹符合动作预期（跌倒: z 快速下降；行走: xy 移动）
4. 没有连续大量丢帧
"""
```

检查项：

| 检查项 | 正常 | 异常 |
|--------|------|------|
| 每帧点数 | 跌倒在 5-50 点，行走在 10-80 点 | 全 0 = 雷达没检测到人 |
| 质心高度变化 | 站立约 1-1.7m，跌倒后约 0-0.3m | 跌倒动作质心没显著下降 = 录错了 |
| 帧间隔 | 约 50ms | 间隔 > 200ms = 掉帧 |
| 噪声点比例 | < 30% 的 points velocity 过大（> 3m/s） | > 50% = 环境多径严重，要换个位置 |

---

# Phase 3：标注（Week 2-3，与 Phase 2 有重叠）

## 3.1 标注工具

你需要一个可视化标注工具。不要手写复杂 UI——用 matplotlib 交互功能就够了。

```python
# label_pointcloud.py - 架构草图

"""
逐帧播放点云 3D 散点图，键盘打标签。
标签体系: 0=none, 1=stand, 2=sit, 3=lie, 4=walk, 5=fall, 6=squat, 7=bend

交互：
- 键盘 0-7: 给当前帧打标签
- 键盘 →: 下一帧
- 键盘 ←: 上一帧
- 键盘 Space: 播放/暂停（自动播放 20fps）
- 键盘 s: 保存标签文件
- 鼠标滚轮: 缩放
- 鼠标拖拽: 旋转 3D 视角

保存格式: frames_labeled.jsonl 每行 {frame_id, label, label_str, ts}
"""
```

标注窗口设计：

```
┌──────────────────────────────────────────────┐
│  File: fall_forward_01.jsonl                 │
│  Frame: 042/098  |  Label: 5 (fall)          │
│                                              │
│         ┌─── 3D 点云视图 ───┐                │
│         │   ·  ·            │                │
│         │     · ··          │                │
│         │       ·· ·        │                │
│         │    ·     ·        │                │
│         │   ·       ·       │                │
│         └───────────────────┘                │
│                                              │
│  [0]none [1]stand [2]sit [3]lie [4]walk     │
│  [5]fall [6]squat [7]bend                   │
│  ← → 导航  Space 播放  S 保存               │
└──────────────────────────────────────────────┘
```

### 标注策略

不用每帧都标。因为采集时文件名已经含了动作标签。对于跌倒文件（`fall_*.jsonl`）：

- **启动帧**：人还在站着 → 标 `stand`
- **过渡帧**：身体在下落 → 标 `fall`
- **落地帧**：人在地面 → 标 `lie`

大部分帧可以**自动从文件名推断初始标签**，你只需要修正边界帧。

```python
# auto_label_from_filename.py - 架构草图

"""
根据文件名自动生成粗标签，再手工修正

规则：
- fall_*     → 前 20% 帧 stand, 中间 30% fall, 后 50% lie
- walk       → 全部 walk
- sit        → 前 20% stand, 后面 sit
- 等等
"""
```

### 标注数据量估算

| 类别 | 文件数 | 每文件帧数 | 总帧数（估算） |
|------|--------|-----------|---------------|
| stand (站立) | ~30 | ~20 | ~600 |
| sit (坐着) | ~25 | ~60 | ~1,500 |
| lie (躺着) | ~30 | ~60 | ~1,800 |
| walk (行走) | ~15 | ~100 | ~1,500 |
| fall (正在跌) | ~70 | ~20 | ~1,400 |
| squat (蹲) | ~10 | ~60 | ~600 |
| bend (弯腰) | ~10 | ~60 | ~600 |
| **总计** | | | **~8,000 帧** |

8,000 帧 × 10-50 点/帧 = 对点云来说是极小数据集。这也是为什么我们先用帧级分类（PointNet++ on single frame），时序模型等数据多了再说。

---

# Phase 4：ALG-0 模型（Week 3-6）

## 4.1 数据预处理 Pipeline

```python
# preprocess.py - 架构草图

"""
输入: raw/ 目录下的 .jsonl 点云文件 + labels/ 目录下的标签文件
输出: train/val/test splits，统一格式的 .h5 或 .pt 文件

处理步骤:
1. 读取每帧点云 + 标签
2. 点云归一化: center at origin, scale to unit sphere (或保留物理尺度)
3. 点云均匀采样到固定点数 (如 64 点):

   - 如果点数 < 64: 重复随机采样补齐
   - 如果点数 > 64: farthest point sampling (FPS) 降采样到 64

4. 保存为: {points: [64, 5], label: int, orig_n: int, ts: int}

特征维度 5: (x, y, z, velocity, intensity)
"""
```

### 数据增强

小数据集必须做增强：

| 增强方法 | 参数 | 说明 |
|----------|------|------|
| 随机旋转（绕 Z 轴） | ±30° | 模拟不同安装角度 |
| 随机平移 | ±0.2m | 模拟人在不同位置 |
| 随机缩放 | 0.9-1.1 | 模拟不同身高的人 |
| 加点噪声 | σ=0.02m (x,y,z), σ=0.1m/s (v) | 模拟雷达测量噪声 |
| 随机丢点 | 10-30% 的点 | 模拟遮挡/低 SNR |
| 镜像翻转 | 左右 | 模拟不同朝向 |

```python
# augment.py - 架构草图

def augment_pointcloud(points: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
    """
    points: [N, 5] (x, y, z, v, i)
    返回增强后的点云 + 标签（标签不变，跌倒/蹲/弯腰翻折后还是同类）
    """
    # 1. 随机绕 Z 轴旋转
    theta = np.random.uniform(-np.pi/6, np.pi/6)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    points[:, :3] = points[:, :3] @ rot_mat.T

    # 2. 平移
    points[:, :3] += np.random.uniform(-0.2, 0.2, size=3)

    # 3. 缩放
    scale = np.random.uniform(0.9, 1.1)
    points[:, :3] *= scale
    points[:, 3] *= scale  # velocity also scales

    # 4. 噪声
    points[:, :3] += np.random.normal(0, 0.02, size=(points.shape[0], 3))
    points[:, 3] += np.random.normal(0, 0.1, size=points.shape[0])

    # 5. 随机丢点
    mask = np.random.random(points.shape[0]) > np.random.uniform(0.1, 0.3)
    points = points[mask]

    return points, label
```

---

## 4.2 姿态分类模型（0c）

### 模型定义

```python
# model_pointnet.py - 架构草图

"""
轻量 PointNet++ 风格姿态分类器

输入: [B, N, 5]  B=batch, N=64 (采样后点数), 5=(x,y,z,v,i)
输出: [B, 7]  7类: none/stand/sit/lie/walk/fall/squat+bend

注意：fall 在帧级分类中是一个"瞬时姿态"，为方便训练，把标注为 fall 的帧直接归为 fall 类。
后续 ALG-2 会在时序上做跌落的最终确认。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """轻量空间变换网络，学习点云旋转不变性"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, k, N]
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 初始化为单位矩阵
        iden = torch.eye(self.k, dtype=torch.float32).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetClassifier(nn.Module):
    """轻量 PointNet for mmWave 点云姿态分类"""

    def __init__(self, num_classes=7, input_dim=5):
        super().__init__()
        self.input_transform = TNet(k=input_dim)
        self.feat_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        x: [B, N, input_dim]
        返回: [B, num_classes] logits
        """
        # 转置为 [B, input_dim, N]
        x = x.transpose(1, 2)

        # 输入变换
        trans_input = self.input_transform(x)
        x = torch.bmm(trans_input, x)  # [B, input_dim, N]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 特征变换
        trans_feat = self.feat_transform(x)  # [B, 64, 64]
        x = torch.bmm(trans_feat, x)  # [B, 64, N]

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 1024, N]

        # 全局 max pooling
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)

        # MLP
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
```

### 训练配置

```python
# train_posture.py - 架构草图

"""
训练姿态分类器

配置（候选·待实验确定）：
- Optimizer: AdamW, lr=0.001, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=100 epochs
- Batch size: 32
- Epochs: 100 (early stop patience=15)
- Loss: CrossEntropyLoss (class weights 平衡样本不均)
- Val split: 20% stratified by class
- Test split: 20% stratified by class

评估指标:
- Per-class precision/recall/f1
- Confusion matrix
- Macro F1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from datetime import datetime


def train_one_epoch(model, loader, optimizer, criterion, device):
    """单 epoch 训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        # points: [B, N, 5], labels: [B]

        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(all_labels, all_preds, target_names=[
        'none', 'stand', 'sit', 'lie', 'walk', 'fall', 'squat_bend'
    ])

    return total_loss / len(loader), acc, report


def main():
    # Config
    NUM_CLASSES = 7
    INPUT_DIM = 5
    NUM_POINTS = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    # train_ds = PointCloudDataset('data/processed/train', num_points=NUM_POINTS, augment=True)
    # val_ds = PointCloudDataset('data/processed/val', num_points=NUM_POINTS, augment=False)
    # test_ds = PointCloudDataset('data/processed/test', num_points=NUM_POINTS, augment=False)

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = PointNetClassifier(num_classes=NUM_CLASSES, input_dim=INPUT_DIM).to(device)

    # Loss — 类别加权，因为各类样本量不均衡
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0]).to(device)
    # fall 和 squat_bend 权重更高，因为样本少且重要
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Logger
    writer = SummaryWriter(f'runs/posture_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    best_val_acc = 0
    patience_counter = 0
    PATIENCE = 15

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_report = validate(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_posture_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()

    # Final test
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load('checkpoints/best_posture_model.pt'))
    _, test_acc, test_report = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.3f}")
    print(test_report)
```

### 训练小数据集的注意事项

1. **BatchNormalization 要调小 momentum**（0.9→0.5），小 batch 下 BN 不稳定
2. **K-fold cross validation (k=5)** 比单次 train/val/test 更靠谱，8,000 帧太少
3. **别过度关注 test accuracy**，关注 confuse matrix 里 `fall` 和 `sit`/`squat` 的区分度
4. **如果 PointNet++ 效果不好**，退回到更简单的方案：手写特征 + XGBoost/LightGBM

### Fallback 方案：手工特征 + LightGBM

如果深度学习在 8,000 帧上效果不好，用这个方案：

```python
# features_handcrafted.py - 架构草图

"""
从点云帧提取手工特征，做经典 ML 分类

特征列表（每帧输出一个特征向量）：
1. 点云点数
2. 质心高度 (z_mean)
3. 质心高度方差 (z_std)
4. 点云在 x/y/z 方向的散布 (x_std, y_std, z_std)
5. 速度均值、最大值、最小值、标准差
6. 强度均值、最大值、标准差
7. 点云高度直方图（10 bin = 10 维特征）
8. 点云 PCA 前 3 个主成分的解释方差比
9. 点云凸包体积
10. 点云中最高点和最低点的 z 差

总共约 25-30 维特征 → LightGBM → 7 分类
这通常比小数据上的深度学习更稳定，但天花板低。
"""
```

---

## 4.3 跌倒初判（0d）

```python
# fall_detector.py - 架构草图

"""
跌倒初判：从姿态分类序列中检测跌倒候选事件

两种方案：

方案 A（规则，MVP 首选）：
- 规则 1: 连续 3 帧内出现了 stand/walk → fall → lie 的姿态转移
- 规则 2: 质心 z 在 1 秒内下降 > 0.5m
- 规则 3: 速度向量方差突然增大（跌倒时四肢乱动）
- 三个规则中的至少两个满足 → 标记为跌倒候选

方案 B（时序分类器，数据多了再换）：
- 滑窗取 90 帧（4.5 秒）姿态序列 → 1D CNN / LSTM → 二分类
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class FallCandidate:
    ts: int
    confidence: float
    evidence: list   # 触发了哪些规则
    pre_posture: str
    post_posture: str


class RuleBasedFallDetector:
    """基于规则的跌倒初判器"""

    def __init__(self,
                 height_drop_threshold: float = 0.5,    # 质心下降阈值 (m)
                 time_window_sec: float = 1.0,          # 姿态转移的时间窗口 (秒)
                 frame_rate: int = 20):
        self.height_threshold = height_drop_threshold
        self.window_frames = int(time_window_sec * frame_rate)
        self.posture_history = deque(maxlen=self.window_frames + 10)

    def update(self, frame_id: int, ts: int,
               posture: str, posture_conf: float,
               centroid_z: float, velocity_var: float) -> Optional[FallCandidate]:
        """
        每帧调用。如果有跌倒候选事件，返回 FallCandidate；否则返回 None。

        posture: 姿态分类结果 (stand/sit/lie/walk/fall)
        centroid_z: 质心高度
        velocity_var: 点云速度向量的方差
        """
        self.posture_history.append({
            'frame_id': frame_id, 'ts': ts,
            'posture': posture, 'conf': posture_conf,
            'z': centroid_z, 'vel_var': velocity_var
        })

        # 需要足够的帧
        if len(self.posture_history) < 5:
            return None

        # 规则 1: 姿态序列中出现 stand/walk → fall → lie
        postures = [p['posture'] for p in list(self.posture_history)[-self.window_frames:]]
        rule1 = self._check_posture_sequence(postures)
        score = 0

        if rule1:
            score += 1

        # 规则 2: 质心高度在窗口内下降 > threshold
        z_values = [p['z'] for p in list(self.posture_history)[-self.window_frames:]]
        max_z = max(z_values[:self.window_frames//2])  # 前半段最高点
        min_z = min(z_values[-self.window_frames//2:]) # 后半段最低点
        if max_z - min_z > self.height_threshold:
            score += 1

        # 规则 3: 速度方差突然增大
        vel_vars = [p['vel_var'] for p in list(self.posture_history)[-self.window_frames:]]
        if len(vel_vars) > 3:
            baseline_var = np.median(vel_vars[:-3])
            recent_var = np.mean(vel_vars[-3:])
            if recent_var > baseline_var * 3:  # 3x 突增
                score += 1

        # 至少满足 2/3 规则 → 跌倒候选
        if score >= 2:
            return FallCandidate(
                ts=ts,
                confidence=score / 3.0,
                evidence=[
                    f"Posture sequence: {rule1}",
                    f"Height drop: {max_z - min_z:.2f}m",
                    f"Velocity spike: {recent_var:.2f} vs baseline {baseline_var:.2f}"
                ],
                pre_posture=postures[0] if postures else '?',
                post_posture=postures[-1] if postures else '?'
            )

        return None

    def _check_posture_sequence(self, postures: list) -> bool:
        """检查是否出现了 stand/walk → fall → lie 的模式"""
        posture_str = '→'.join(postures)
        # 简单字符串匹配
        patterns = [
            'stand→fall→lie',
            'walk→fall→lie',
            'stand→fall',
            'walk→fall'
        ]
        for pat in patterns:
            if pat in posture_str:
                return True
        return False
```

---

## 4.4 安静态生命体征提取（0e）

```python
# vital_signs.py - 架构草图

"""
从安静态（sit/lie）点云中提取呼吸频率和心率

原理：
雷达发射 FMCW 信号，遇到人体胸腔后反射，
胸腔的微小周期性移动（呼吸 ~12mm 振幅、心跳 ~0.5mm 振幅）产生相位调制。
在距离 bin 的相位变化中，通过 FFT 提取呼吸和心跳频率。

步骤：
1. 判定安静状态：连续 N 秒姿态为 sit/lie + 速度方差 < 阈值
2. 选择目标距离 bin：找到点云质心对应的距离 bin
3. 相位提取：对该 bin 的连续 chirp 相位做 unwrap
4. 带通滤波：
   - 呼吸: 0.1-0.5 Hz (6-30 次/分)
   - 心跳: 0.8-3.0 Hz (48-180 bpm)
5. FFT 找出峰值频率
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass


@dataclass
class VitalSigns:
    resp_rate: float       # 呼吸频率 (次/分)
    heart_rate: float      # 心率 (bpm)
    resp_confidence: float  # 0-1
    heart_confidence: float
    is_quiet: bool         # 是否处于安静态


def is_quiet_state(posture: str, velocity_var: float,
                   quiet_duration_sec: float = 5.0,
                   max_velocity_var: float = 0.1) -> bool:
    """判断是否处于安静态"""
    return posture in ('sit', 'lie') and velocity_var < max_velocity_var


def extract_vital_signs(phase_signal: np.ndarray,
                        sample_rate: float = 20.0) -> VitalSigns:
    """
    从相位信号中提取呼吸和心率

    phase_signal: 连续 chirp 的相位值序列，已 unwrap
    sample_rate: 采样率 (Hz)，等于 chirp 循环频率
    """
    n = len(phase_signal)

    # 去直流
    phase_signal = phase_signal - np.mean(phase_signal)

    # 呼吸带通滤波 (0.1 - 0.5 Hz)
    b_resp, a_resp = signal.butter(4, [0.1, 0.5], btype='band', fs=sample_rate)
    resp_signal = signal.filtfilt(b_resp, a_resp, phase_signal)

    # 心跳带通滤波 (0.8 - 3.0 Hz)
    b_hr, a_hr = signal.butter(4, [0.8, 3.0], btype='band', fs=sample_rate)
    hr_signal = signal.filtfilt(b_hr, a_hr, phase_signal)

    # FFT 找峰值
    resp_fft = np.abs(np.fft.rfft(resp_signal))
    hr_fft = np.abs(np.fft.rfft(hr_signal))
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)

    # 呼吸
    resp_band = (freqs >= 0.1) & (freqs <= 0.5)
    resp_peak_idx = np.argmax(resp_fft[resp_band])
    resp_rate = freqs[resp_band][resp_peak_idx] * 60  # 转为 次/分

    # 心率
    hr_band = (freqs >= 0.8) & (freqs <= 3.0)
    hr_peak_idx = np.argmax(hr_fft[hr_band])
    heart_rate = freqs[hr_band][hr_peak_idx] * 60

    # 置信度：峰值 vs 背景噪声的比值
    resp_snr = resp_fft[resp_band][resp_peak_idx] / np.median(resp_fft[resp_band])
    hr_snr = hr_fft[hr_band][hr_peak_idx] / np.median(hr_fft[hr_band])

    return VitalSigns(
        resp_rate=resp_rate,
        heart_rate=heart_rate,
        resp_confidence=min(resp_snr / 3.0, 1.0),   # SNR > 3 → 高置信度
        heart_confidence=min(hr_snr / 2.0, 1.0),     # SNR > 2 → 高置信度
        is_quiet=True
    )
```

---

# Phase 5：TI 固件 A/B 对比（Week 6-7）

## 5.1 对比测试 Design

```python
# ab_test.py - 架构草图

"""
在相同的测试集上，比较：
1. 你的 ALG-0 模型（姿态分类 → 跌倒初判 → 状态机确认）
2. TI 官方 Fall Detection 固件

测试集：20 个跌倒 + 20 个假跌倒 + 20 个日常动作
"""

@dataclass
class ABResult:
    method: str                     # 'ours' | 'ti_firmware'
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    total_samples: int = 0
    avg_latency_ms: float = 0      # 从事件发生到检测输出的延迟

    @property
    def recall(self):
        return self.true_positive / (self.true_positive + self.false_negative) if (self.true_positive + self.false_negative) > 0 else 0

    @property
    def precision(self):
        return self.true_positive / (self.true_positive + self.false_positive) if (self.true_positive + self.false_positive) > 0 else 0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    @property
    def false_alarm_rate(self):
        """误报率：假跌倒中被误报为真跌倒的比例"""
        return self.false_positive / (self.false_positive + self.true_negative) if (self.false_positive + self.true_negative) > 0 else 0


def run_ab_test(test_dir: str) -> tuple[ABResult, ABResult]:
    """
    跑 A/B 对比测试
    对 test_dir 下每个文件运行两个方法，记录结果
    """
    our_result = ABResult(method='ours (ALG-0 + ALG-2)')
    ti_result = ABResult(method='TI Fall Detection Firmware')

    for file in Path(test_dir).glob('*.jsonl'):
        true_label = extract_true_label(file.name)  # 从文件名推断 ground truth

        # 跑我们的模型
        our_pred, our_latency = run_our_model(file)
        update_ab_result(our_result, our_pred, true_label)

        # 跑 TI 固件（需要同时采集 TI 固件输出，或者回放点云给固件）
        ti_pred, ti_latency = run_ti_firmware(file)
        update_ab_result(ti_result, ti_pred, true_label)

    return our_result, ti_result


def print_ab_report(ours: ABResult, ti: ABResult):
    """打印 A/B 对比报告"""
    print("=" * 60)
    print("  FALL DETECTION A/B COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Ours':>15} {'TI Firmware':>15}")
    print("-" * 60)
    print(f"{'Recall':<25} {ours.recall:>14.1%} {ti.recall:>14.1%}")
    print(f"{'Precision':<25} {ours.precision:>14.1%} {ti.precision:>14.1%}")
    print(f"{'F1 Score':<25} {ours.f1:>14.1%} {ti.f1:>14.1%}")
    print(f"{'False Alarm Rate':<25} {ours.false_alarm_rate:>14.1%} {ti.false_alarm_rate:>14.1%}")
    print(f"{'Avg Latency (ms)':<25} {ours.avg_latency_ms:>14.0f} {ti.avg_latency_ms:>14.0f}")
    print("=" * 60)

    # 结论
    if ours.f1 > ti.f1 and ours.false_alarm_rate < ti.false_alarm_rate:
        print("✅ OURS WINS: Better F1 AND lower false alarm rate.")
    elif ours.f1 > ti.f1:
        print("⚡ Ours has better F1, but false alarm rate comparison is mixed.")
    elif ours.false_alarm_rate < ti.false_alarm_rate:
        print("⚡ Ours has lower false alarm rate, but F1 comparison is mixed.")
    else:
        print("⚠️  TI firmware still outperforms. Keep iterating.")
```

## 5.2 测试报告模板

```
家安 housafe · 跌倒检测 A/B 测试报告
日期: 2026-XX-XX
数据集: n=20 real falls, n=20 false falls, n=20 daily activities

┌──────────────────────┬───────────────┬───────────────┐
│ 指标                  │ ALG-0+ALG-2   │ TI Firmware   │
├──────────────────────┼───────────────┼───────────────┤
│ 召回率 (真跌倒不漏)    │      95%      │      90%      │
│ 精确率 (告警是真)      │      82%      │      65%      │
│ F1 Score             │      0.88     │      0.75     │
│ 误报率 (假跌倒误报)    │      15%      │      35%      │
│ 平均检测延迟           │     4.2s     │     ~1.0s     │
└──────────────────────┴───────────────┴───────────────┘

结论: [自研胜/持平/落后]

关键洞察:
- 我们的模型对 "快速坐下" 的误报明显低于 TI 固件（原因是...）
- 对 "躺姿起摔" 场景，双方检测都弱，ALG-0 漏了 3/10 个样本
```

---

# Phase 6：实时推理管线（Week 7-9）

## 6.1 推理 Server

```python
# server.py - 架构草图

"""
FastAPI 推理服务
- POST /infer: 接收单帧点云，返回姿态分类 + 跌倒初判 + 生命体征
- 内部状态机维护 ALG-2 延迟确认
- WebSocket /stream: 实时双向通信（点云帧进 → 结构化事件出）

部署：uvicorn server:app --host 0.0.0.0 --port 8765
"""

from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import time
from collections import defaultdict
import asyncio


app = FastAPI(title="家安 housafe · Inference Server")

# 全局状态
model: torch.nn.Module = None
device: torch.device = None
fall_detectors: dict = defaultdict(lambda: RuleBasedFallDetector())  # 每个 radar_id 一个检测器


# ============ REST API ============

class PointInput(BaseModel):
    x: float
    y: float
    z: float
    velocity: float
    intensity: float


class FrameInput(BaseModel):
    ts: int
    radar_id: str
    frame_id: int
    points: List[PointInput]


class PostureResult(BaseModel):
    posture: str           # stand/sit/lie/walk/fall/none
    confidence: float
    probs: dict            # 各类概率


class FallAlert(BaseModel):
    ts: int
    is_fall: bool
    confidence: float
    confirmed: bool        # ALG-2 确认后为 True
    evidence: List[str]


class InferenceResult(BaseModel):
    ts: int
    frame_id: int
    posture: PostureResult
    fall_candidate: Optional[FallAlert]
    num_points: int
    centroid_z: float
    centroid_z_change: float
    inference_time_ms: float


@app.post("/infer")
async def infer_frame(frame: FrameInput) -> InferenceResult:
    """
    单帧推理
    """
    t0 = time.time()

    # 1. 点云转 tensor
    points_np = np.array([[p.x, p.y, p.z, p.velocity, p.intensity]
                          for p in frame.points])
    centroid_z = np.mean(points_np[:, 2]) if len(points_np) > 0 else 0

    # 2. 点云预处理（标准化 + 采样到 64 点）
    points_tensor = preprocess_points(points_np, num_points=64).unsqueeze(0).to(device)

    # 3. 姿态分类推理
    with torch.no_grad():
        logits = model(points_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    posture_idx = int(np.argmax(probs))
    posture_names = ['none', 'stand', 'sit', 'lie', 'walk', 'fall', 'squat_bend']
    posture = posture_names[posture_idx]
    confidence = float(probs[posture_idx])

    # 4. 跌倒初判
    detector = fall_detectors[frame.radar_id]
    velocity_var = np.var(points_np[:, 3]) if len(points_np) > 0 else 0
    fall_cand = detector.update(
        frame.frame_id, frame.ts, posture, confidence, centroid_z, velocity_var
    )

    # 5. ALG-2 延迟确认（简化版：初判后等 5 秒，如果后续持续 lie → 确认）
    if fall_cand:
        # 实际实现中这里要有一个异步状态机，等 5 秒后检查后续姿态
        confirmed = False  # placeholder
        evidence = fall_cand.evidence
    else:
        confirmed = False
        evidence = []

    t1 = time.time()

    return InferenceResult(
        ts=frame.ts,
        frame_id=frame.frame_id,
        posture=PostureResult(
            posture=posture,
            confidence=confidence,
            probs={posture_names[i]: float(p) for i, p in enumerate(probs)}
        ),
        fall_candidate=FallAlert(
            ts=frame.ts,
            is_fall=fall_cand is not None,
            confidence=fall_cand.confidence if fall_cand else 0,
            confirmed=confirmed,
            evidence=evidence
        ) if fall_cand else None,
        num_points=frame.points and len(frame.points) or 0,
        centroid_z=centroid_z,
        centroid_z_change=centroid_z - detector._prev_z if hasattr(detector, '_prev_z') else 0,
        inference_time_ms=(t1 - t0) * 1000
    )


# ============ WebSocket ============

@app.websocket("/stream")
async def stream_inference(websocket: WebSocket):
    """
    实时双向推理流
    客户端发 JSON 帧 → 服务端回 JSON 结果
    """
    await websocket.accept()
    print(f"Client connected")

    detector = RuleBasedFallDetector()
    prev_z = None

    while True:
        try:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            # 解析帧
            points_np = np.array([[p['x'], p['y'], p['z'], p['v'], p['i']]
                                  for p in data.get('points', [])])
            centroid_z = np.mean(points_np[:, 2]) if len(points_np) > 0 else 0

            # 推理
            points_tensor = preprocess_points(points_np, num_points=64).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(points_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            posture_idx = int(np.argmax(probs))
            posture = posture_names[posture_idx]

            # 跌倒初判
            velocity_var = np.var(points_np[:, 3]) if len(points_np) > 0 else 0
            fall_cand = detector.update(
                data.get('frame_id', 0), data.get('ts', int(time.time()*1000)),
                posture, float(probs[posture_idx]), centroid_z, velocity_var
            )

            result = {
                'ts': data.get('ts'),
                'frame_id': data.get('frame_id'),
                'posture': posture,
                'fall_alert': fall_cand is not None,
                'fall_confidence': fall_cand.confidence if fall_cand else 0,
            }

            await websocket.send_text(json.dumps(result))

        except Exception as e:
            print(f"Error: {e}")
            break


# ============ 启动 ============

def load_model(checkpoint_path: str = 'checkpoints/best_posture_model.pt'):
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetClassifier(num_classes=7, input_dim=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

---

## 6.2 ALG-2 跌倒延迟确认状态机

```python
# fall_confirmer.py - 架构草图

"""
ALG-2: 跌倒延迟确认

原理：
收到 ALG-0 跌倒初判后，不立即告警。
等待 3-5 秒（可配置），观察老人后续行为：
- 如果恢复了站立/行走 → 误报，取消告警，仅记录
- 如果持续躺在地上或缓慢爬起 → 真跌倒，发告警

状态机：
IDLE → CANDIDATE (收到初判) → 等待 N 秒 → VERIFY → CONFIRMED / DISMISSED
"""

from enum import Enum
import time
from dataclasses import dataclass, field
from typing import List, Optional


class FallState(Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"       # 收到初判，等待窗口
    VERIFYING = "verifying"       # 窗口结束，检查最终状态
    CONFIRMED = "confirmed"       # 确认真跌倒
    DISMISSED = "dismissed"       # 确认为误报
    COOLDOWN = "cooldown"         # 告警后冷却，避免重复告警


@dataclass
class FallConfirmer:
    """ALG-2 跌倒延迟确认状态机"""

    confirm_window_sec: float = 5.0         # 初判后等多久再判决
    cooldown_sec: float = 60.0              # 确认后多长时间不重复告警
    recovery_postures: tuple = ('stand', 'walk')  # 恢复姿态
    fall_postures: tuple = ('lie',)               # 跌倒姿态

    state: FallState = FallState.IDLE
    candidate_ts: int = 0
    candidate_evidence: list = field(default_factory=list)
    observation_buffer: list = field(default_factory=list)  # 初判后观察到的姿态
    last_alert_ts: int = 0

    def update(self, ts: int, posture: str, fall_candidate: Optional[dict] = None) -> Optional[dict]:
        """
        每帧调用。

        返回 None: 无告警
        返回 dict: {ts, confidence, evidence, is_fall: bool}
        """

        # 检查冷却
        if self.state == FallState.COOLDOWN:
            if (ts - self.last_alert_ts) / 1000 > self.cooldown_sec:
                self.state = FallState.IDLE
            else:
                return None  # 冷却中，不处理

        # 收到跌倒初判
        if fall_candidate and self.state == FallState.IDLE:
            self.state = FallState.CANDIDATE
            self.candidate_ts = ts
            self.candidate_evidence = fall_candidate.get('evidence', [])
            self.observation_buffer = [posture]
            return None  # 先不告警，观察

        # 观察中
        if self.state == FallState.CANDIDATE:
            self.observation_buffer.append(posture)

            elapsed_sec = (ts - self.candidate_ts) / 1000
            if elapsed_sec >= self.confirm_window_sec:
                # 窗口到，做最终判决
                self.state = FallState.VERIFYING
                return self._make_decision(ts)
            else:
                # 如果期间恢复站立/行走 → 可以直接 DISMISS
                if posture in self.recovery_postures:
                    self.state = FallState.DISMISSED
                    result = self._make_decision(ts)
                    self.state = FallState.IDLE
                    return result

            return None

        return None

    def _make_decision(self, ts: int) -> dict:
        """根据观察窗口内的姿态序列做最终判决"""
        # 统计观察期内的主导姿态
        postures = self.observation_buffer
        lie_ratio = sum(1 for p in postures if p in self.fall_postures) / len(postures) if postures else 0
        sit_ratio = sum(1 for p in postures if p == 'sit') / len(postures) if postures else 0

        # 决策逻辑
        if lie_ratio > 0.5:
            # 观察期内超过一半时间是躺着的 → 真跌倒
            is_fall = True
            confidence = min(lie_ratio, 1.0)
            self.state = FallState.CONFIRMED
            self.last_alert_ts = ts
        elif sit_ratio > 0.5:
            # 更多是坐着 → 误报（可能只是快速坐下）
            is_fall = False
            confidence = 1 - sit_ratio
            self.state = FallState.DISMISSED
        else:
            # 不确定，偏保守——如果是跌倒漏了代价大
            is_fall = lie_ratio > 0.3
            confidence = 0.5
            self.state = FallState.CONFIRMED if is_fall else FallState.DISMISSED
            if is_fall:
                self.last_alert_ts = ts

        # 确认后进冷却
        if is_fall:
            self.state = FallState.COOLDOWN

        return {
            'ts': ts,
            'is_fall': is_fall,
            'confidence': confidence,
            'evidence': self.candidate_evidence + [
                f"Observation window: lie_ratio={lie_ratio:.2f}, sit_ratio={sit_ratio:.2f}",
                f"Posture sequence: {postures[-10:]}"
            ]
        }

    def reset(self):
        self.state = FallState.IDLE
        self.observation_buffer = []
```

---

# Phase 7：Web Dashboard + 告警通知（Week 9-10）

## 7.1 Web Dashboard

一个简单的前端页面（纯 HTML + JS，不需要框架），显示实时推理结果。

```html
<!-- dashboard.html - 架构草图 -->

<!DOCTYPE html>
<html>
<head>
    <title>家安 housafe · Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
        .container { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr auto; height: 100vh; gap: 12px; padding: 12px; }
        .panel { background: #1e293b; border-radius: 12px; padding: 16px; overflow: hidden; }
        .status-bar { grid-column: 1 / -1; display: flex; gap: 16px; align-items: center; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .status-indicator.online { background: #22c55e; animation: pulse 2s infinite; }
        .status-indicator.offline { background: #ef4444; }
        .alert-banner { background: #dc2626; color: white; padding: 16px; border-radius: 8px; font-size: 1.2em; animation: shake 0.5s; display: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-8px); } 75% { transform: translateX(8px); } }
    </style>
</head>
<body>
    <div class="container">
        <!-- 状态栏 -->
        <div class="panel status-bar">
            <span><span class="status-indicator online" id="status-light"></span> 雷达在线</span>
            <span>姿态: <strong id="posture-text">--</strong></span>
            <span>帧率: <span id="fps-text">--</span></span>
            <span>推理延迟: <span id="latency-text">--</span></span>
        </div>

        <!-- 跌倒告警横幅 -->
        <div class="alert-banner" id="alert-banner" style="grid-column: 1 / -1;">
            ⚠️ 检测到跌倒！置信度: <span id="alert-confidence">--</span>
        </div>

        <!-- 3D 点云视图 -->
        <div class="panel" id="pointcloud-plot" style="grid-row: span 2;"></div>

        <!-- 姿态概率分布 -->
        <div class="panel" id="posture-bar"></div>

        <!-- 告警历史 -->
        <div class="panel" id="alert-history" style="overflow-y: auto;">
            <h3>告警历史</h3>
            <div id="alert-list"></div>
        </div>
    </div>

    <script>
        // WebSocket 连接推理 server
        const ws = new WebSocket('ws://localhost:8765/stream');

        ws.onopen = () => {
            document.getElementById('status-light').className = 'status-indicator online';
        };

        ws.onclose = () => {
            document.getElementById('status-light').className = 'status-indicator offline';
        };

        let alerts = [];

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // 更新姿态
            document.getElementById('posture-text').textContent = data.posture;

            // 更新 3D 点云图（这需要从 server 拿到点云坐标——调整协议）
            // 实际实现：WebSocket 双向传递点云坐标

            // 跌倒告警
            if (data.fall_alert) {
                const banner = document.getElementById('alert-banner');
                banner.style.display = 'block';
                document.getElementById('alert-confidence').textContent =
                    (data.fall_confidence * 100).toFixed(0) + '%';

                alerts.unshift({
                    ts: new Date(data.ts).toLocaleTimeString(),
                    confidence: data.fall_confidence,
                    posture: data.posture
                });
                if (alerts.length > 50) alerts.pop();

                // 更新告警历史列表
                document.getElementById('alert-list').innerHTML = alerts.map(a =>
                    `<div style="padding:8px;border-bottom:1px solid #334155">
                        <span style="color:#94a3b8">${a.ts}</span>
                        <span style="color:#f87171;font-weight:bold">跌倒告警</span>
                        <span>置信度 ${(a.confidence * 100).toFixed(0)}%</span>
                    </div>`
                ).join('');

                // 3 秒后自动隐藏横幅
                setTimeout(() => {
                    document.getElementById('alert-banner').style.display = 'none';
                }, 3000);
            }
        };

        // Plotly 初始化空图
        const layout = {
            scene: {
                xaxis: { title: 'X', range: [-2, 2] },
                yaxis: { title: 'Y', range: [-2, 2] },
                zaxis: { title: 'Z', range: [0, 2.5] },
                aspectmode: 'data'
            },
            margin: { l: 0, r: 0, t: 0, b: 0 },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#1e293b'
        };
        Plotly.newPlot('pointcloud-plot', [], layout);
    </script>
</body>
</html>
```

## 7.2 桌面通知 + 短信

```python
# notifier.py - 架构草图

"""
告警通知模块
- 桌面通知: macOS 用 osascript, Windows 用 win10toast
- 短信: Twilio API（免费额度够 MVP 测试）
"""

import subprocess
import os
# from twilio.rest import Client  # pip install twilio

# Twilio 配置（注册免费账号获取）
# TWILIO_SID = "your_account_sid"
# TWILIO_TOKEN = "your_auth_token"
# TWILIO_PHONE = "+1234567890"
# EMERGENCY_PHONE = "+8613800138000"  # 紧急联系人手机号


def send_desktop_notification(title: str, message: str):
    """发送桌面弹窗通知"""
    system = os.uname().sysname if hasattr(os, 'uname') else 'Windows'

    if system == 'Darwin':  # macOS
        subprocess.run([
            'osascript', '-e',
            f'display notification "{message}" with title "{title}" sound name "Glass"'
        ])
    elif system == 'Windows':
        # 用 win10toast
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=10)
    else:
        print(f"[NOTIFICATION] {title}: {message}")


def send_sms_alert(message: str):
    """发送短信告警（Twilio）"""
    # client = Client(TWILIO_SID, TWILIO_TOKEN)
    # client.messages.create(
    #     body=f"[家安 housafe] {message}",
    #     from_=TWILIO_PHONE,
    #     to=EMERGENCY_PHONE
    # )
    print(f"[SMS] {message}")


def handle_fall_alert(alert: dict):
    """处理跌倒告警"""
    title = "⚠️ 检测到跌倒"
    message = f"时间: {alert['ts']} | 置信度: {alert['confidence']:.0%}"

    # 1. 桌面通知
    send_desktop_notification(title, message)

    # 2. 如果是高置信度（>80%），发短信
    if alert['confidence'] > 0.8:
        send_sms_alert(message)

    # 3. 记录日志
    with open('alerts.log', 'a') as f:
        f.write(f"{alert['ts']} | fall | conf={alert['confidence']:.3f} | {alert['evidence']}\n")
```

---

# Phase 8：数据飞轮 + 内测（Week 10-12）

## 8.1 内测流程

```
目标：找 2-5 个人帮你测试

Day 1: 在朋友家/自己家架雷达
Day 1-3: 让朋友自由活动（不刻意做动作），录 2-3 天日常数据
Day 4: 让朋友执行跌倒脚本（模拟向前摔 + 快速坐下 + 蹲下），录对照数据
Day 5: 回访——哪些时候系统误报了？哪些时候漏了？
```

## 8.2 数据回环脚本

```python
# data_flywheel.py - 架构草图

"""
每次内测后：
1. 把新标注数据加入训练集
2. 重新训练模型
3. 对比新旧模型的精度
4. 如果新模型更好 → 替换线上模型
5. 记录版本号和精度变化
"""

MODEL_REGISTRY = "models/registry.json"

def add_to_training_set(new_data_dir: str):
    """将新标注数据加入训练集"""
    # 1. 读取新数据
    # 2. 与已有数据合并
    # 3. 重新 split train/val/test
    # 4. 启动重训
    pass

def compare_models(old_ckpt: str, new_ckpt: str, test_loader) -> dict:
    """对比新旧模型"""
    old_model = load_model(old_ckpt)
    new_model = load_model(new_ckpt)

    old_metrics = evaluate(old_model, test_loader)
    new_metrics = evaluate(new_model, test_loader)

    return {
        'old': old_metrics,
        'new': new_metrics,
        'better': new_metrics['f1'] > old_metrics['f1']
    }
```

---

# 附录

## A. 项目目录结构

```
housafe/
├── data/
│   ├── raw/                  # 原始点云 .jsonl 文件（按日期分目录）
│   │   ├── 20260715/
│   │   ├── 20260716/
│   │   └── ...
│   ├── labels/               # 标注文件
│   ├── processed/            # 预处理后的训练数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── ab_test/              # A/B 对比测试集
│
├── models/
│   ├── checkpoints/          # 模型权重
│   │   ├── best_posture_model.pt
│   │   └── ...
│   ├── registry.json         # 模型版本记录
│   └── exported/             # ONNX/TensorRT 导出
│
├── src/
│   ├── capture/              # 点云采集
│   │   ├── isk_reader.py         # ISK 串口 TLV 解析
│   │   ├── record_session.py     # 录制主循环
│   │   └── capture_pointcloud.py # 入口脚本
│   │
│   ├── preprocess/           # 数据预处理
│   │   ├── label_tool.py
│   │   ├── auto_label.py
│   │   ├── preprocess.py
│   │   └── augment.py
│   │
│   ├── models/               # 模型定义与训练
│   │   ├── pointnet.py
│   │   ├── train_posture.py
│   │   ├── fall_detector.py
│   │   ├── fall_confirmer.py
│   │   └── vital_signs.py
│   │
│   ├── server/               # 推理服务
│   │   ├── server.py
│   │   ├── notifier.py
│   │   └── dashboard.html
│   │
│   └── eval/                 # 评估
│       ├── ab_test.py
│       └── data_flywheel.py
│
├── config/
│   ├── profile_20fps.cfg     # 雷达 chirp 配置
│   └── server_config.yaml    # 推理服务配置
│
├── notebooks/                # Jupyter notebooks (探索性分析)
│   ├── 01_explore_pointcloud.ipynb
│   ├── 02_train_posture_baseline.ipynb
│   └── 03_ab_test_analysis.ipynb
│
├── docs/
│   ├── 家安_PRD_总纲_v1.md
│   ├── ...
│   ├── 家安_MVP_执行手册_v1.md  ← 就是本文档
│   └── logbook.md            # 实验日志（每天记录）
│
├── requirements.txt
└── README.md
```

## B. 实验日志模板

每天花 5 分钟记录，积累成最重要的工程资产：

```markdown
# 2026-07-15 (Day 1)

## 做了什么
- IWR6843ISK 到货，烧录 Out of Box Demo，Demo Visualizer 显示 3D 点云正常
- 编译了 Fall Detection demo 固件

## 发现/问题
- ISK 插上后设备管理器没识别——换了 USB 口好使，确认是 micro-USB 线问题
- 点云帧率不稳定，有时候掉到 15fps——尝试换 5V 独立电源而非 USB 供电

## 明天要做
- 写 Python 采集脚本，跑通 ISK 串口 → .jsonl 通路
- 录第一批跌倒数据（向前摔 × 10）

## 数据状态
- 今天录制: 0 文件
- 累计: 0 / 200+
- TI 固件对比基准: 未开始
```

## C. 关键验证节点

| 时间 | 检查点 | 通过标准 |
|------|--------|---------|
| Week 1 末 | 硬件链路 | mmWave Demo Visualizer 显示 3D 点云 ✅ |
| Week 2 末 | 数据采集 | ≥ 100 个原始点云文件，质量检查通过 ✅ |
| Week 3 末 | 标注完成 | ≥ 8,000 帧有 correct label ✅ |
| Week 4 末 | 姿态分类 v1 | 5-fold CV macro F1 ≥ 0.85 ✅ |
| Week 6 末 | 跌倒初判 v1 | 召回 ≥ 95% ✅ |
| Week 7 末 | A/B 对比 | 自研 vs TI 固件对比报告 ✅ |
| Week 9 末 | 推理管线 | FastAPI server 运行，单帧 < 200ms ✅ |
| Week 10 末 | Dashboard | 实时点云 + 姿态 + 告警显示 ✅ |
| Week 12 末 | 内测 | 2-5 人在真实环境测试 ≥ 3 天 ✅ |
