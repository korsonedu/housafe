"""仿真系统共享类型 — 所有模块通过 dataclass 通信"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VitalRecord:
    """单帧生命体征"""
    heart_rate: float       # bpm
    resp_rate: float        # bpm
    quality: float          # 0-1


@dataclass
class GroundTruth:
    """精确标注 — AI 评估的参考基准"""
    frame_id: str
    ts: int                           # UTC ms
    posture: str                      # stand/sit/lie/walk/fall
    posture_confidence: float         # 1.0=synthetic, 0.7-0.95=replay
    heart_rate_true: float | None
    resp_rate_true: float | None
    anomaly_type: str | None          # fall/stillness/vital_anomaly/offline/None
    anomaly_severity: str | None      # info/warning/critical
    anomaly_start: bool = False
    anomaly_end: bool = False
    scenario_name: str = ""
    generator: str = ""               # synthetic/replay/hybrid
    params_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if k == "anomaly_start":
                d[k] = int(v)
            elif k == "anomaly_end":
                d[k] = int(v)
            else:
                d[k] = v
        return d


@dataclass
class FrameGroup:
    """统一帧协议 — 所有生成器输出"""
    frame_id: str
    ts: int                  # UTC ms
    device_id: str
    room: str
    points: np.ndarray | None = None    # (N,5) float32 RoomModel 变换后
    raw_points: np.ndarray | None = None # (N,5) float32 body-centered（世界模型用）
    vitals: VitalRecord | None = None
    gt: GroundTruth | None = None
    heartbeat: bool = False             # True=心跳帧（无点云/体征，仅保活）


@dataclass
class RoomConfig:
    """房间物理参数"""
    name: str
    size: tuple[float, float, float]         # (x, y, z) 米
    radar_pos: tuple[float, float, float]    # 雷达安装位置
    noise_sigma: float = 0.02                # 高斯噪声标准差（米）
    dropout_rate: float = 0.01               # 随机丢点率
