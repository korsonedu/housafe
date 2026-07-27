"""房间模型 + 生命体征生成"""
import math
import numpy as np
from simulator.types import RoomConfig, VitalRecord

RESTING_HR = 72
RESTING_RR = 16

ACTIVITY_MULT = {
    "lie": 0.9, "sit": 1.0, "stand": 1.05,
    "walk": 1.3, "fall": 1.6, "squat": 1.1,
}
RR_MULT = {
    "lie": 1.0, "sit": 1.0, "stand": 1.1,
    "walk": 1.5, "fall": 2.0, "squat": 1.1,
}


class RoomModel:
    """房间坐标变换 + 噪声"""

    def __init__(self, config: RoomConfig):
        self._config = config
        self._rng = np.random.RandomState(42)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """对点云施加坐标变换和噪声，返回过滤后的点云"""
        if points is None or len(points) == 0:
            return points

        pts = points.copy()

        # 1. 坐标平移：世界坐标系 → 雷达相对坐标
        rx, ry, rz = self._config.radar_pos
        pts[:, 0] -= rx
        pts[:, 1] -= ry
        pts[:, 2] -= rz

        # 2. 墙壁裁剪（雷达相对坐标）
        sx, sy, sz = self._config.size
        mask = ((pts[:, 0] >= -rx) & (pts[:, 0] <= sx - rx) &
                (pts[:, 1] >= -ry) & (pts[:, 1] <= sy - ry) &
                (pts[:, 2] >= -rz) & (pts[:, 2] <= sz - rz))
        pts = pts[mask]

        if len(pts) == 0:
            return pts

        # 3. 高斯噪声（σ 对应 IWR6843 距离分辨率）
        noise = self._rng.normal(0, self._config.noise_sigma, size=(len(pts), 3))
        pts[:, :3] += noise

        # 4. 随机丢点
        if self._config.dropout_rate > 0:
            keep = self._rng.random(len(pts)) > self._config.dropout_rate
            pts = pts[keep]

        # 坐标平移回绝对坐标
        pts[:, 0] += rx
        pts[:, 1] += ry
        pts[:, 2] += rz

        return pts.astype(np.float32)


class VitalSignsGen:
    """生理信号模型 — 活动相关 HR/RR + 呼吸性窦性心律不齐"""

    def __init__(self, resting_hr: float = RESTING_HR,
                 resting_rr: float = RESTING_RR, seed: int = 42):
        self.resting_hr = resting_hr
        self.resting_rr = resting_rr
        self._rng = np.random.RandomState(seed)

    def generate(self, activity: str, t: float) -> VitalRecord:
        """t: 场景时间（秒）"""
        hr_mult = ACTIVITY_MULT.get(activity, 1.0)
        rr_mult = RR_MULT.get(activity, 1.0)

        # 呼吸性窦性心律不齐（RSA）：吸气时心率↑，呼气时↓
        rr_cycle = self.resting_rr * rr_mult
        rr_freq = rr_cycle / 60.0  # Hz
        rsa_amplitude = 3.0        # bpm

        hr = (self.resting_hr * hr_mult
              + rsa_amplitude * math.sin(2 * math.pi * rr_freq * t)
              + self._rng.normal(0, 3))

        rr = rr_cycle + self._rng.normal(0, 1)

        quality = 0.95 + self._rng.normal(0, 0.02)
        quality = max(0.1, min(1.0, quality))

        return VitalRecord(
            heart_rate=round(max(30, min(200, hr)), 1),
            resp_rate=round(max(5, min(40, rr)), 1),
            quality=round(quality, 2),
        )
