"""场景引擎 — YAML 解析 + 时间线编排"""
import re
import yaml
from pathlib import Path
import numpy as np
from simulator.types import FrameGroup, GroundTruth, RoomConfig
from simulator.generators import SyntheticGenerator, ReplayGenerator
from simulator.room_vitals import RoomModel, VitalSignsGen
from simulator.anomaly import AnomalyInjector

DEFAULT_FPS = 10

_TIME_RE = re.compile(r"^(\d+\.?\d*)\s*(s|m|h)?$")
_TIME_MULT = {"s": 1, "m": 60, "h": 3600}


def _parse_time(val) -> float:
    """解析时间字符串 → 秒。支持 '10s', '2m', '0.5h', 纯数字。"""
    if isinstance(val, (int, float)):
        return float(val)
    m = _TIME_RE.match(str(val).strip())
    if not m:
        raise ValueError(f"无法解析时间: {val!r}")
    num = float(m.group(1))
    unit = m.group(2) or "s"
    return num * _TIME_MULT[unit]


class ScenarioEngine:
    """解析 YAML 场景，编排时间线，产出 FrameGroup 序列"""

    def __init__(self, yaml_path: str, speed: float = 1.0,
                 params_path: str | None = None,
                 dataset_path: str | None = None):
        with open(yaml_path) as f:
            self._cfg = yaml.safe_load(f)

        self._speed = speed
        self._fps = self._cfg.get("generator_defaults", {}).get("fps", DEFAULT_FPS)
        self._dt = 1.0 / self._fps

        # 房间模型
        self._rooms: dict[str, RoomModel] = {}
        self._room_configs: dict[str, RoomConfig] = {}
        for room_name, room_data in self._cfg.get("rooms", {}).items():
            cfg = RoomConfig(
                name=room_name,
                size=tuple(room_data["size"]),
                radar_pos=tuple(room_data["radar_pos"]),
            )
            self._rooms[room_name] = RoomModel(cfg)
            self._room_configs[room_name] = cfg

        # 生成器
        self._synth = SyntheticGenerator(
            params_path=params_path) if params_path else SyntheticGenerator()
        self._replay = ReplayGenerator(dataset_path) if dataset_path else None

        # 体征
        vcfg = self._cfg.get("vitals", {})
        self._vitals = VitalSignsGen(
            resting_hr=vcfg.get("resting_hr", 72),
            resting_rr=vcfg.get("resting_rr", 16),
        )

        # 异常注入器
        self._anomaly = AnomalyInjector()

        self._device_id = self._cfg.get("device_id", "sim_device")
        self._scenario_name = self._cfg.get("name", Path(yaml_path).stem)
        self._frame_seq = 0

    def run(self) -> list[FrameGroup]:
        """运行整个时间线，返回全部帧列表（含异常注入）"""
        timeline = self._cfg.get("timeline", [])
        injections = self._cfg.get("injections", [])

        frames: list[FrameGroup] = []
        for segment in timeline:
            segment_frames = self._run_segment(segment)
            frames.extend(segment_frames)

        # 按时间排序
        frames.sort(key=lambda f: f.ts)

        # 注入异常：将 YAML 的 at (时间字符串) → at_s (float 秒)
        if injections:
            for inj in injections:
                if "at" in inj and "at_s" not in inj:
                    inj["at_s"] = _parse_time(inj.pop("at"))
            frames = self._anomaly.inject(frames, injections)

        # 插入心跳帧（每 10 帧一次）
        frames = self._insert_heartbeats(frames)

        return frames

    def _run_segment(self, seg: dict) -> list[FrameGroup]:
        """执行一个时间线段"""
        at_s = _parse_time(seg.get("at", 0))
        duration_s = _parse_time(seg.get("duration", 10))
        activity = seg.get("activity", "sit")
        room_name = seg.get("room", "bedroom")
        generator_type = seg.get("generator", "synthetic")

        # 过渡段
        transition = seg.get("transition")
        frames: list[FrameGroup] = []
        trans_duration = 0.0

        if transition:
            trans_dur = _parse_time(transition.get("duration", 2))
            trans_frames = self._run_transition(transition, at_s, room_name, generator_type)
            frames.extend(trans_frames)
            trans_duration = trans_dur
            at_s += trans_dur
            duration_s -= trans_dur

        if duration_s <= 0:
            return frames

        # 主动作段
        n_frames = max(1, int(duration_s * self._fps))
        actual_dt = duration_s / n_frames

        # 回放：预取帧序列
        replay_frames: list[np.ndarray] = []
        effective_generator = generator_type
        if generator_type in ("replay", "hybrid") and self._replay:
            replay_frames = self._replay.generate_sequence(activity, duration_s)
            if not replay_frames:
                effective_generator = "synthetic"
        elif generator_type == "replay" and not self._replay:
            effective_generator = "synthetic"

        if effective_generator == "replay":
            effective_generator = "replay"
        elif effective_generator == "synthetic":
            pass  # effective_generator stays as is

        prev_centroid = None
        for i in range(n_frames):
            t = at_s + i * actual_dt

            # 生成点云
            if effective_generator == "replay" and i < len(replay_frames):
                points = replay_frames[i].copy()
            elif activity in ("lie", "lying"):
                # 3DPCHM 无 lying，用 squat 近似 z 分布
                points = self._synth.generate("squat", prev_centroid, actual_dt)
            else:
                points = self._synth.generate(activity, prev_centroid, actual_dt)

            if points is not None and len(points) > 0:
                prev_centroid = points[:, :3].mean(axis=0)

            # 保存 body-centered 原始点云（世界模型用）
            raw = points.copy() if points is not None and len(points) > 0 else None

            # 将 body-centered 点云平移到房间雷达正下方（地面位置）
            if points is not None and len(points) > 0 and room_name in self._room_configs:
                cfg = self._room_configs[room_name]
                points[:, 0] += cfg.radar_pos[0]
                points[:, 1] += cfg.radar_pos[1]

            # 房间变换
            if room_name in self._rooms:
                points = self._rooms[room_name].apply(points)

            # 体征
            vitals = self._vitals.generate(activity, t)

            # 帧 ID
            self._frame_seq += 1
            frame_id = f"{self._device_id}-{self._frame_seq:06d}"

            # Ground truth
            gt = GroundTruth(
                frame_id=frame_id, ts=int(t * 1000),
                posture=activity,
                posture_confidence=1.0 if effective_generator == "synthetic" else 0.85,
                heart_rate_true=vitals.heart_rate,
                resp_rate_true=vitals.resp_rate,
                anomaly_type=None, anomaly_severity=None,
                scenario_name=self._scenario_name,
                generator=effective_generator,
            )

            frames.append(FrameGroup(
                frame_id=frame_id, ts=int(t * 1000),
                device_id=self._device_id, room=room_name,
                points=points, raw_points=raw, vitals=vitals, gt=gt,
            ))

        return frames

    def _run_transition(self, trans: dict, at_s: float, room: str,
                        generator: str) -> list[FrameGroup]:
        """生成姿态过渡帧序列"""
        from_posture = trans.get("from", "stand")
        to_posture = trans.get("to", "sit")
        duration = _parse_time(trans.get("duration", 2.0))

        n_frames = max(2, int(duration * self._fps))
        actual_dt = duration / n_frames
        frames: list[FrameGroup] = []

        prev_centroid = None
        for i in range(n_frames):
            progress = i / max(n_frames - 1, 1)
            t = at_s + i * actual_dt

            posture = from_posture if progress < 0.5 else to_posture
            points = self._synth.generate(posture, prev_centroid, actual_dt)
            if points is not None and len(points) > 0:
                prev_centroid = points[:, :3].mean(axis=0)

            # 保存 body-centered 原始点云
            raw = points.copy() if points is not None and len(points) > 0 else None

            # 平移到房间雷达正下方
            if points is not None and len(points) > 0 and room in self._room_configs:
                cfg = self._room_configs[room]
                points[:, 0] += cfg.radar_pos[0]
                points[:, 1] += cfg.radar_pos[1]

            if room in self._rooms:
                points = self._rooms[room].apply(points)

            vitals = self._vitals.generate(posture, t)

            self._frame_seq += 1
            frames.append(FrameGroup(
                frame_id=f"{self._device_id}-{self._frame_seq:06d}",
                ts=int(t * 1000), device_id=self._device_id, room=room,
                points=points, raw_points=raw, vitals=vitals,
                gt=GroundTruth(
                    frame_id=f"{self._device_id}-{self._frame_seq:06d}",
                    ts=int(t * 1000), posture=posture,
                    posture_confidence=0.85,
                    heart_rate_true=vitals.heart_rate,
                    resp_rate_true=vitals.resp_rate,
                    anomaly_type=None, anomaly_severity=None,
                    scenario_name=self._scenario_name, generator=generator,
                ),
            ))
        return frames

    def _insert_heartbeats(self, frames: list[FrameGroup]) -> list[FrameGroup]:
        """每 10 帧在序列中标记心跳帧（在现有帧上标记，不新增帧）"""
        for i, f in enumerate(frames):
            if (i + 1) % 10 == 0:
                f.heartbeat = True
        return frames
