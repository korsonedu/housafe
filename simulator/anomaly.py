"""异常注入器 — 在时间线精确位置注入异常并产出标注"""
import numpy as np
from simulator.types import FrameGroup, GroundTruth, VitalRecord

ANOMALY_TEMPLATES = {
    "fall": {
        "transition_duration": 1.0,
        "end_posture": "lie",
        "velocity_spike": True,
        "height_drop_rate": 1.0,
    },
    "stillness": {
        "min_duration_s": 1800,
        "max_displacement": 0.05,
    },
    "vital_anomaly": {
        "hr_range": (40, 130),
        "rr_range": (5, 35),
        "ramp_duration_s": 60,
    },
    "offline": {
        "gap_duration_s": 120,
    },
}


class AnomalyInjector:
    """在帧序列中注入异常并标注 ground truth"""

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def inject(self, frames: list[FrameGroup],
               injections: list[dict]) -> list[FrameGroup]:
        """按 injection 配置修改帧序列。倒序处理避免索引偏移。"""
        injections = sorted(injections, key=lambda x: x["at_s"], reverse=True)
        result = list(frames)

        for inj in injections:
            at_s = inj["at_s"]
            inj_type = inj["type"]
            inj_params = inj.get("params", {})
            gt_meta = inj.get("ground_truth", {})

            if inj_type == "fall":
                result = self._inject_fall(result, at_s, inj_params, gt_meta)
            elif inj_type == "stillness":
                result = self._inject_stillness(result, at_s, inj_params, gt_meta)
            elif inj_type == "vital_anomaly":
                result = self._inject_vital_anomaly(result, at_s, inj_params, gt_meta)
            elif inj_type == "offline":
                result = self._inject_offline(result, at_s, inj_params, gt_meta)

        return result

    def _find_insert_index(self, frames: list[FrameGroup], at_s: float) -> int:
        """找到 at_s 对应的时间索引"""
        if not frames:
            return 0
        at_ts = int(at_s * 1000)
        for i, f in enumerate(frames):
            if f.ts >= at_ts:
                return i
        return len(frames)

    # ── 跌倒 ──

    def _inject_fall(self, frames: list[FrameGroup], at_s: float,
                     params: dict, gt_meta: dict) -> list[FrameGroup]:
        idx = self._find_insert_index(frames, at_s)
        if idx >= len(frames):
            return frames

        duration = params.get("transition_duration", 1.0)
        drop_rate = params.get("height_drop_rate", 1.0)
        n_frames = max(1, int(duration * 10))  # 10 fps
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            progress = (i - idx) / max(n_frames - 1, 1)
            if f.points is not None and len(f.points) > 0:
                f.points[:, 2] -= drop_rate * duration * progress / n_frames
                f.points[:, 2] = np.maximum(f.points[:, 2], 0.0)
            f.gt = self._make_gt(f, "fall", gt_meta, i == idx, i == end_idx - 1,
                                 posture="fall" if progress < 0.8 else "lie")
        return frames

    # ── 静止 ──

    def _inject_stillness(self, frames: list[FrameGroup], at_s: float,
                          params: dict, gt_meta: dict) -> list[FrameGroup]:
        idx = self._find_insert_index(frames, at_s)
        min_dur = params.get("min_duration_s", 1800)
        n_frames = max(1, int(min_dur * 10))
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            self._apply_gt(f, "stillness", gt_meta, i == idx, i == end_idx - 1)
        return frames

    # ── 体征异常 ──

    def _inject_vital_anomaly(self, frames: list[FrameGroup], at_s: float,
                              params: dict, gt_meta: dict) -> list[FrameGroup]:
        idx = self._find_insert_index(frames, at_s)
        ramp_dur = params.get("ramp_duration_s", 60)
        hr_range = params.get("hr_range", (40, 130))
        rr_range = params.get("rr_range", (5, 35))
        n_frames = max(1, int(ramp_dur * 10))
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            progress = (i - idx) / max(n_frames - 1, 1)
            target_hr = hr_range[0] + (hr_range[1] - hr_range[0]) * progress
            target_rr = rr_range[0] + (rr_range[1] - rr_range[0]) * progress
            f.vitals = VitalRecord(
                heart_rate=round(target_hr + self._rng.normal(0, 2), 1),
                resp_rate=round(target_rr + self._rng.normal(0, 0.5), 1),
                quality=round(0.95 + self._rng.normal(0, 0.02), 2),
            )
            self._apply_gt(f, "vital_anomaly", gt_meta, i == idx, i == end_idx - 1)
        return frames

    # ── 断连 ──

    def _inject_offline(self, frames: list[FrameGroup], at_s: float,
                        params: dict, gt_meta: dict) -> list[FrameGroup]:
        idx = self._find_insert_index(frames, at_s)
        gap_dur = params.get("gap_duration_s", 120)
        n_frames = max(1, int(gap_dur * 10))
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            f.points = None
            f.vitals = None
            f.heartbeat = False
            self._apply_gt(f, "offline", gt_meta, i == idx, i == end_idx - 1,
                           posture="", confidence=0.0)
        return frames

    # ── helpers ──

    def _make_gt(self, f: FrameGroup, anomaly_type: str, gt_meta: dict,
                 is_start: bool, is_end: bool, posture: str = "lie",
                 confidence: float = 1.0) -> GroundTruth:
        return GroundTruth(
            frame_id=f.frame_id, ts=f.ts, posture=posture,
            posture_confidence=confidence,
            heart_rate_true=f.vitals.heart_rate if f.vitals else None,
            resp_rate_true=f.vitals.resp_rate if f.vitals else None,
            anomaly_type=anomaly_type,
            anomaly_severity=gt_meta.get("severity", "warning"),
            anomaly_start=is_start, anomaly_end=is_end,
            scenario_name=gt_meta.get("desc", ""), generator="synthetic",
        )

    def _apply_gt(self, f: FrameGroup, anomaly_type: str, gt_meta: dict,
                  is_start: bool, is_end: bool, posture: str = "lie",
                  confidence: float = 1.0):
        if f.gt is None:
            f.gt = self._make_gt(f, anomaly_type, gt_meta, is_start, is_end,
                                 posture=posture, confidence=confidence)
        else:
            f.gt.anomaly_type = anomaly_type
            f.gt.anomaly_severity = gt_meta.get("severity", "warning")
            f.gt.anomaly_start = is_start
            f.gt.anomaly_end = is_end
