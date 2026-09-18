#!/usr/bin/env python3
"""
家安世界模型 · Phase A Worker

替代 ai/placeholder/main.py 的规则模拟器。
消费 Redis Stream 点云/生命体征帧 → 提取特征 → Fallback 规则引擎
→ 基线比较 → 异常检测 → 通知决策 → POST DecoderOutput 回 backend。

用法: python -m ai.main [redis_url] [backend_url] [token]
"""
import json
import os
import sys
import time
import math
from collections import defaultdict
from collections import deque
from datetime import datetime

import numpy as np
import requests

from ai.shared.types import (
    FeatureVector, AnomalyResult, NotificationDecision,
    PointCloudFrame, VitalFrame, time_encode,
    POSTURES,
)
from ai.shared.redis_client import FrameConsumer
from ai.fallback.rules import FallbackEngine
from ai.latent_space.graph import SpatialGraph
from ai.baseline.gmm_baseline import PersonalBaseline
from ai.baseline.latent_baseline import LatentBaseline
from ai.decoders.anomaly import AnomalyDecoder
from ai.decoders.notification import NotificationDecider
from ai.decoders.report import ReportDecoder
from ai.decoders.tracking import TrackingDecoder

# Phase B: 训练好的世界模型组件
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "encoder_best.pt")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "predictor_best.pt")
LATENT_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "latent_baseline.pkl")

try:
    import torch
    import torch.nn.functional as torch_F
    from ai.encoder.encoder_model import EncoderModel
    from ai.predictor.model import TinyPredictor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── 配置 ──────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", sys.argv[1] if len(sys.argv) > 1 else "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("BACKEND_URL", sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000")
INTERNAL_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", sys.argv[3] if len(sys.argv) > 3 else "dev-internal-token")
DECODER_ENDPOINT = f"{BACKEND_URL}/api/internal/decoder-output"

STATE_DIR = os.environ.get("AI_STATE_DIR", "/tmp/housafe_ai_state")
STATE_PERSIST_INTERVAL_S = 300  # 5 分钟


# ── 特征提取（临时替代 Encoder）─────────────────────

def extract_features_from_pointcloud(frame: PointCloudFrame) -> FeatureVector:
    """从点云帧提取特征向量（规则方法，等 Encoder 替换）"""
    pts = frame.points
    n = len(pts)

    if n == 0:
        return FeatureVector(
            ts=frame.ts, device_id=frame.device_id, room=frame.room,
            posture="stand", posture_confidence=0.0, presence=False,
            moving=False, centroid=(0, 0, 0), height=0.0, n_points=0,
            occupancy_estimate=0, resp_rate=None, heart_rate=None,
            vital_quality=None, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.0,
        )

    # 质心
    centroid = tuple(pts.mean(axis=0)[:3].astype(float))

    # 高度 = z 跨度
    zs = pts[:, 2]
    height = float(zs.max() - zs.min())

    # 姿态（规则：高度阈值）
    posture = "stand"
    if height < 0.3:
        posture = "lie"
    elif height < 0.8:
        posture = "sit"
    elif height >= 1.8:
        posture = "walk"

    # 运动
    velocities = pts[:, 3]
    moving = bool(np.mean(np.abs(velocities)) > 0.1)
    velocity_variance = float(np.var(velocities))

    # 人数估计（简单：点数阈值）
    occupancy_estimate = 1 if n >= 5 else 0

    # 置信度：基于点数和信号强度
    intensities = pts[:, 4]
    confidence = min(float(np.mean(intensities)), 1.0) if n > 0 else 0.0

    # 时间编码
    dt = datetime.utcfromtimestamp(frame.ts / 1000.0)
    hour = dt.hour + dt.minute / 60.0
    h_sin, h_cos = time_encode(hour)
    weekday = dt.weekday()

    return FeatureVector(
        ts=frame.ts, device_id=frame.device_id, room=frame.room,
        posture=posture, posture_confidence=round(confidence, 3),
        presence=True, moving=moving,
        centroid=centroid, height=round(height, 3),
        n_points=n, occupancy_estimate=occupancy_estimate,
        resp_rate=None, heart_rate=None, vital_quality=None,
        hour_sin=h_sin, hour_cos=h_cos, weekday=weekday,
        velocity_variance=round(velocity_variance, 3),
    )


def extract_features_from_vital(frame: VitalFrame) -> FeatureVector:
    """从生命体征帧提取特征向量（更新已有 FV 的生命体征部分）"""
    dt = datetime.utcfromtimestamp(frame.ts / 1000.0)
    hour = dt.hour + dt.minute / 60.0
    h_sin, h_cos = time_encode(hour)

    return FeatureVector(
        ts=frame.ts, device_id=frame.device_id, room=frame.room,
        posture="stand", posture_confidence=0.0, presence=True,
        moving=False, centroid=(0, 0, 0), height=0.0, n_points=0,
        occupancy_estimate=1,
        resp_rate=frame.resp_rate, heart_rate=frame.heart_rate,
        vital_quality=frame.quality,
        hour_sin=h_sin, hour_cos=h_cos, weekday=dt.weekday(),
        velocity_variance=0.0,
    )


# ── Worker ────────────────────────────────────────────

class WorldModelWorker:
    """世界模型主进程。"""

    def __init__(self, redis_url: str = REDIS_URL, backend_url: str = BACKEND_URL,
                 token: str = INTERNAL_TOKEN):
        self.redis_url = redis_url
        self.backend_url = backend_url
        self.token = token
        self.endpoint = f"{backend_url}/api/internal/decoder-output"

        # 组件
        self.fallback = FallbackEngine()
        self.graph = SpatialGraph()
        self.baseline = PersonalBaseline()
        self.latent_baseline: LatentBaseline | None = None
        self.anomaly = AnomalyDecoder(self.fallback, self.baseline, self.graph)
        self.notifier = NotificationDecider()
        self.reporter = ReportDecoder()
        self.tracker = TrackingDecoder()

        # 世界模型（Phase B）
        self._models_available = False
        self.encoder: EncoderModel | None = None
        self.predictor: TinyPredictor | None = None
        self._backbone_buffer: deque = deque(maxlen=32)  # (1024,) tensors
        self._state_buffer: deque = deque(maxlen=32)      # (256,) tensors
        self._load_models()

        # 状态
        self.consumer: FrameConsumer | None = None
        self.feature_buffer: dict[str, deque[FeatureVector]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self.recent_decisions: deque[NotificationDecision] = deque(maxlen=100)
        self._last_state_persist = time.time()

        # 状态恢复
        os.makedirs(STATE_DIR, exist_ok=True)
        self._load_state()

    # ── 模型加载 ──────────────────────────────────────

    def _load_models(self):
        """加载 Encoder + Predictor，任一步失败则降级为纯规则模式"""
        if not _TORCH_AVAILABLE:
            print("[WorldModel] torch not installed, running in rules-only mode")
            return
        if not os.path.exists(ENCODER_PATH) or not os.path.exists(PREDICTOR_PATH):
            print("[WorldModel] Model files not found, running in rules-only mode")
            return

        try:
            # Encoder
            self.encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
            enc_ckpt = torch.load(ENCODER_PATH, map_location="cpu")
            if isinstance(enc_ckpt, dict) and "model_state_dict" in enc_ckpt:
                self.encoder.load_state_dict(enc_ckpt["model_state_dict"])
            else:
                self.encoder.load_state_dict(enc_ckpt)
            self.encoder.eval()

            # Predictor
            self.predictor = TinyPredictor(in_dim=256, hid=128)
            pred_ckpt = torch.load(PREDICTOR_PATH, map_location="cpu")
            if isinstance(pred_ckpt, dict) and "model_state_dict" in pred_ckpt:
                self.predictor.load_state_dict(pred_ckpt["model_state_dict"])
            else:
                self.predictor.load_state_dict(pred_ckpt)
            self.predictor.eval()

            self._models_available = True
            enc_params = sum(p.numel() for p in self.encoder.parameters())
            pred_params = sum(p.numel() for p in self.predictor.parameters())
            print(f"[WorldModel] Models loaded: encoder ({enc_params/1e6:.1f}M) "
                  f"+ predictor ({pred_params/1e6:.1f}M)")

            # LatentBaseline
            if os.path.exists(LATENT_BASELINE_PATH):
                self.latent_baseline = LatentBaseline.load(LATENT_BASELINE_PATH)
                print(f"[WorldModel] LatentBaseline loaded: "
                      f"{len(self.latent_baseline._gmms)} contexts, "
                      f"ready={self.latent_baseline.is_ready}")
            else:
                print("[WorldModel] LatentBaseline not found, latent scoring disabled")
        except Exception as e:
            print(f"[WorldModel] Model load failed: {e}, running in rules-only mode")
            self._models_available = False
            self.encoder = None
            self.predictor = None

    def _encode_frame(self, points) -> torch.Tensor | None:
        """单帧点云 → 256-d S_t。返回 None 表示缓冲区未就绪。
        points: np.ndarray (N, 5) [x,y,z,velocity,intensity]"""
        if not self._models_available or self.encoder is None:
            return None
        n = len(points)
        if n < 4:  # PointNet++ FPS 需要至少几个点
            return None

        pts = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)  # (1, N, 5)
        xyz = pts[:, :, :3]
        feat = pts[:, :, 3:]

        with torch.no_grad():
            bb = self.encoder.backbone(xyz, feat)  # (1, 1024)

        self._backbone_buffer.append(bb.squeeze(0))  # (1024,)

        # TCN 需要序列上下文，缓冲区至少 4 帧
        if len(self._backbone_buffer) < 4:
            return None

        seq = torch.stack(list(self._backbone_buffer)).unsqueeze(0)  # (1, T, 1024)
        with torch.no_grad():
            S_seq = self.encoder.tcn(seq)  # (1, T, 256)
        return S_seq[0, -1]  # (256,) — 当前帧隐状态

    # ── 主循环 ──────────────────────────────────────

    def run(self):
        """阻塞式主循环"""
        self.consumer = FrameConsumer(self.redis_url)
        print(f"[WorldModel] Phase B worker started "
              f"(models={'on' if self._models_available else 'off'})")
        print(f"[WorldModel] Redis: {self.redis_url}")
        print(f"[WorldModel] Backend: {self.endpoint}")

        while True:
            result = self.consumer.consume_one()
            if result is None:
                self._periodic_tasks()
                continue

            stream_name, frame, msg_id = result
            self._process_frame(stream_name, frame)
            self._periodic_tasks()

    def _process_frame(self, stream_name: str, frame: PointCloudFrame | VitalFrame):
        """处理单帧"""
        # 提取特征（保留手写规则，FallbackEngine 依赖 posture/height 等）
        if isinstance(frame, PointCloudFrame):
            fv = extract_features_from_pointcloud(frame)
        else:
            fv = extract_features_from_vital(frame)

        # 更新 buffer
        buf = self.feature_buffer[fv.device_id]
        buf.append(fv)
        history = list(buf)

        # 世界模型推理：latent density score（主信号） + predictor error（辅助）
        latent_score = 0.0
        predictor_error = 0.0
        if isinstance(frame, PointCloudFrame) and self._models_available:
            S_t = self._encode_frame(frame.points)
            if S_t is not None:
                # 隐状态密度估计 — 世界模型核心信号
                if self.latent_baseline is not None and self.latent_baseline.is_ready:
                    latent_score = self.latent_baseline.score(S_t.numpy(), fv.ts)

                # Predictor error — 辅助信号
                if len(self._state_buffer) >= 2:
                    hist = torch.stack(list(self._state_buffer)).unsqueeze(0)  # (1, T, 256)
                    with torch.no_grad():
                        pred = self.predictor(hist)  # (1, 256)
                    error = 1.0 - torch_F.cosine_similarity(
                        pred, S_t.unsqueeze(0), dim=-1
                    )
                    predictor_error = float(error.item())
                self._state_buffer.append(S_t)

        # 运行异常检测
        anomaly = self.anomaly.evaluate(fv, history,
                                        predictor_error=predictor_error,
                                        latent_score=latent_score)
        if anomaly is None:
            return

        # 通知决策
        decision = self.notifier.decide(anomaly, list(self.recent_decisions))
        self.recent_decisions.append(decision)

        # 发送到 backend
        self._post_result(fv, anomaly, decision)

        # 日志
        ts_str = datetime.utcfromtimestamp(fv.ts / 1000).strftime("%H:%M:%S")
        parts = [f"[WorldModel] {ts_str} {fv.room} {fv.posture}",
                 f"→ {anomaly.anomaly_type}/{anomaly.severity}",
                 f"(score={anomaly.anomaly_score:.2f}"]
        if latent_score > 0:
            parts[-1] += f", latent={latent_score:.2f}"
        if predictor_error > 0:
            parts[-1] += f", pred_err={predictor_error:.3f}"
        parts[-1] += ")"
        parts.append(f"→ {decision.level}")
        print(" ".join(parts))

    def _post_result(self, fv: FeatureVector, anomaly: AnomalyResult,
                     decision: NotificationDecision):
        """POST decoder outputs 到 backend AIBridgeView"""
        # 构造符合 contracts 的 decoder outputs
        outputs = []

        # 姿态
        outputs.append({
            "kind": "posture",
            "family_id": "unknown",
            "payload": {
                "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                "posture": fv.posture, "confidence": fv.posture_confidence,
                "moving": fv.moving, "presence": fv.presence,
            },
        })

        # 生命体征（如果有）
        if fv.resp_rate is not None or fv.heart_rate is not None:
            outputs.append({
                "kind": "vital",
                "family_id": "unknown",
                "payload": {
                    "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                    "resp_rate": fv.resp_rate, "heart_rate": fv.heart_rate,
                    "quality": fv.vital_quality or 0.0,
                },
            })

        # 异常告警
        if decision.level != "none":
            outputs.append({
                "kind": "alert",
                "family_id": "unknown",
                "payload": {
                    "ts": anomaly.ts, "device_id": anomaly.device_id,
                    "room": anomaly.room,
                    "alert_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "payload": {
                        "anomaly_score": anomaly.anomaly_score,
                        "source": anomaly.source,
                        "notification_level": decision.level,
                        "reason": decision.reason,
                        **anomaly.details,
                    },
                },
            })

        # 异常分数
        outputs.append({
            "kind": "anomaly",
            "family_id": "unknown",
            "payload": {
                "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                "anomaly_score": anomaly.anomaly_score,
            },
        })

        try:
            rv = requests.post(
                self.endpoint,
                json={"token": self.token, "outputs": outputs},
                timeout=5,
            )
            if rv.status_code != 200:
                print(f"[WorldModel] POST error: HTTP {rv.status_code} {rv.text[:100]}")
        except requests.RequestException as e:
            print(f"[WorldModel] POST failed: {e}")

    # ── 定期任务 ─────────────────────────────────────

    def _periodic_tasks(self):
        now = time.time()
        if now - self._last_state_persist >= STATE_PERSIST_INTERVAL_S:
            self._persist_state()
            self._last_state_persist = now

    def _persist_state(self):
        try:
            self.graph.save(os.path.join(STATE_DIR, "graph.pkl"))
            self.baseline.save(os.path.join(STATE_DIR, "baseline.pkl"))
        except Exception as e:
            print(f"[WorldModel] persist error: {e}")

    def _load_state(self):
        graph_path = os.path.join(STATE_DIR, "graph.pkl")
        baseline_path = os.path.join(STATE_DIR, "baseline.pkl")
        if os.path.exists(graph_path):
            try:
                self.graph = SpatialGraph.load(graph_path)
                print(f"[WorldModel] Loaded graph: {len(self.graph.nodes)} nodes")
            except Exception:
                pass
        if os.path.exists(baseline_path):
            try:
                self.baseline = PersonalBaseline.load(baseline_path)
                self.anomaly.baseline = self.baseline
                print(f"[WorldModel] Loaded baseline: ready={self.baseline.is_ready}")
            except Exception:
                pass


def main():
    worker = WorldModelWorker()
    worker.run()


if __name__ == "__main__":
    main()
