"""AI 评估桥接 — 纯世界模型 + 多时间尺度异常检测。

短时 Z-score（突发） + 中时趋势（漂移） + 长时抑制（静止）。
不依赖规则引擎/姿态分类。体征异常用医学阈值（非启发式规则）。

用法:
  python -m simulator.eval run scenarios/elderly_day_replay.yaml --dataset <npz>
  python -m simulator.eval batch scenarios/ --dataset <npz> --runs 5
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as torch_F

from simulator.scenario import ScenarioEngine
from simulator.types import FrameGroup
from ai.shared.types import AnomalyResult
from ai.encoder.encoder_model import EncoderModel
from ai.predictor.model import TinyPredictor

ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "ai", "checkpoints", "encoder_best.pt")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "ai", "checkpoints", "predictor_best.pt")

# ── 生理阈值（医学知识，非启发式规则）─────────────────
HR_CRITICAL_LOW, HR_CRITICAL_HIGH = 40, 150
RR_CRITICAL_LOW, RR_CRITICAL_HIGH = 6, 35
STILLNESS_MIN_S = 60      # 连续无运动才报警
OFFLINE_GAP_S = 30         # 断连超过此时间报警


# ── 指标结构 ─────────────────────────────────────────

@dataclass
class EvalMetrics:
    scenario_name: str = ""
    total_frames: int = 0
    total_gt_anomalies: int = 0
    detected_anomalies: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_latencies_ms: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    by_type: dict = field(default_factory=dict)
    model_mode: str = "world_model_multi_scale"

    def compute(self):
        self.true_positives = len(self.detection_latencies_ms)
        self.false_positives = max(0, self.detected_anomalies - self.true_positives)
        self.false_negatives = max(0, self.total_gt_anomalies - self.true_positives)
        if self.detected_anomalies > 0:
            self.precision = self.true_positives / self.detected_anomalies
        if self.total_gt_anomalies > 0:
            self.recall = self.true_positives / self.total_gt_anomalies
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        if self.detection_latencies_ms:
            self.avg_latency_ms = sum(self.detection_latencies_ms) / len(self.detection_latencies_ms)

    def report(self) -> str:
        lines = [
            "=" * 60,
            f"  评估报告: {self.scenario_name}",
            "=" * 60,
            f"  模式:              {self.model_mode}",
            f"  总帧数:            {self.total_frames}",
            f"  GT 异常段:         {self.total_gt_anomalies}",
            f"  检测异常:          {self.detected_anomalies}",
            f"  正确检测 (TP):     {self.true_positives}",
            f"  误报 (FP):         {self.false_positives}",
            f"  漏报 (FN):         {self.false_negatives}",
            f"  ─────────────────────────────",
            f"  精确率 (Precision): {self.precision:.1%}",
            f"  召回率 (Recall):    {self.recall:.1%}",
            f"  F1:                 {self.f1:.3f}",
            f"  平均检测延迟:       {self.avg_latency_ms:.0f} ms",
        ]
        if self.by_type:
            lines.append(f"  按类型: {json.dumps(self.by_type, ensure_ascii=False)}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── GT 异常段提取 ────────────────────────────────────

def extract_gt_anomalies(frames: list[FrameGroup]) -> list[dict]:
    anomalies = []
    current = None
    for f in frames:
        if f.gt and f.gt.anomaly_type:
            if current is None:
                current = {"type": f.gt.anomaly_type, "severity": f.gt.anomaly_severity,
                           "start_ts": f.ts, "end_ts": f.ts, "start_frame": f.frame_id}
            else:
                current["end_ts"] = f.ts
        elif current is not None:
            anomalies.append(current)
            current = None
    if current is not None:
        anomalies.append(current)
    return anomalies


# ── 匹配逻辑 ─────────────────────────────────────────

MATCH_WINDOW_MS = 5000


def match_detections(gt_anomalies: list[dict], ai_results: list[AnomalyResult]) -> tuple[int, list[float]]:
    matched, latencies, matched_ai = 0, [], set()
    for gt in gt_anomalies:
        for ai_idx, result in enumerate(ai_results):
            if ai_idx in matched_ai:
                continue
            if abs(result.ts - gt["start_ts"]) <= MATCH_WINDOW_MS or \
               abs(result.ts - gt["end_ts"]) <= MATCH_WINDOW_MS:
                matched += 1
                latencies.append(result.ts - gt["start_ts"])
                matched_ai.add(ai_idx)
                break
    return matched, latencies


# ── 世界模型加载 ─────────────────────────────────────

def load_world_model() -> tuple[EncoderModel | None, TinyPredictor | None, bool]:
    if not os.path.exists(ENCODER_PATH) or not os.path.exists(PREDICTOR_PATH):
        print("[Eval] 模型文件缺失")
        return None, None, False
    try:
        encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
        ckpt = torch.load(ENCODER_PATH, map_location="cpu")
        encoder.load_state_dict(ckpt.get("model_state_dict", ckpt))
        encoder.eval()

        predictor = TinyPredictor(in_dim=256, hid=128)
        p_ckpt = torch.load(PREDICTOR_PATH, map_location="cpu")
        predictor.load_state_dict(p_ckpt.get("model_state_dict", p_ckpt))
        predictor.eval()

        enc_m = sum(p.numel() for p in encoder.parameters()) / 1e6
        pred_m = sum(p.numel() for p in predictor.parameters()) / 1e6
        print(f"[Eval] 世界模型已加载  encoder={enc_m:.1f}M  predictor={pred_m:.1f}M")
        return encoder, predictor, True
    except Exception as e:
        print(f"[Eval] 模型加载失败: {e}")
        return None, None, False


# ── 世界模型推理器（多时间尺度）──────────────────────

class WorldModelInference:
    """双时间尺度异常检测:
      - 短时 (30帧=3s):  Z-score 突变 → falls / 突发异常
      - 长时 (750帧=75s):误差持续极低 → 静止超时（世界模型"无聊"了）
    中时尺度已移除——实验证明产生大量 FP 且无独立 TP。
    """

    WINDOW = 32
    MAX_POINTS = 64
    WARMUP_WINDOWS = 80

    LOCAL_W = 30
    Z_SHORT = 3.0

    STILLNESS_W = 750
    STILLNESS_LOW_PCT = 10
    STILLNESS_Z_THRESH = 1.5  # cos_sim 的 Z-score 阈值（相对自身基线）

    def __init__(self, encoder: EncoderModel, predictor: TinyPredictor):
        self.encoder = encoder
        self.predictor = predictor
        self._frame_buf: list[np.ndarray] = []
        self._errors: deque[float] = deque(maxlen=self.STILLNESS_W)
        self._cos_sims: deque[float] = deque(maxlen=self.STILLNESS_W)
        self._ready = False
        self._error_p10: float = 0.0
        self._cos_mean: float = 0.0
        self._cos_std: float = 1.0

    def process_frame(self, points: np.ndarray) -> list[dict]:
        """返回触发的异常信号列表 [{"type": str, "score": float, "detail": str}]"""
        n = len(points) if points is not None else 0
        if n < 4:
            self._frame_buf.append(np.zeros((self.MAX_POINTS, 5), dtype=np.float32))
            return []

        # Pad to 64
        pts = self._pad_to_64(points)

        self._frame_buf.append(pts.astype(np.float32))
        if len(self._frame_buf) > self.WINDOW:
            self._frame_buf = self._frame_buf[-self.WINDOW:]

        if len(self._frame_buf) < self.WINDOW:
            return []

        # 训练路径批量推理
        seq = np.stack(self._frame_buf)
        seq_t = torch.from_numpy(seq).unsqueeze(0)

        with torch.no_grad():
            out = self.encoder(seq_t)
            S = out["S"]
            pred_out = self.predictor(S[:, :-1])
            error = 1.0 - torch_F.cosine_similarity(pred_out, S[:, -1], dim=-1)
            error = float(error.item())

        self._errors.append(error)

        # 隐状态稳定性：S_t 和 S_{t-1} 的 cosine similarity
        S_prev = S[0, -2]
        S_curr = S[0, -1]
        cos_sim = float(torch_F.cosine_similarity(S_prev.unsqueeze(0), S_curr.unsqueeze(0), dim=-1))
        self._cos_sims.append(cos_sim)

        if len(self._errors) < self.WARMUP_WINDOWS:
            return []

        if not self._ready:
            self._error_p10 = float(np.percentile(list(self._errors), self.STILLNESS_LOW_PCT))
            cos_all = np.array(list(self._cos_sims))
            self._cos_mean = cos_all.mean()
            self._cos_std = cos_all.std() or 1.0
            self._ready = True

        alerts = []
        errs = np.array(list(self._errors))

        # ── 短时：Z-score 突变 → fall / 突发异常 ──
        local = errs[-self.LOCAL_W:]
        mu, sigma = local.mean(), local.std()
        if sigma > 1e-6:
            z = (error - mu) / sigma
            if z > self.Z_SHORT:
                alerts.append({"type": "sudden_spike", "score": min(z / 6, 1.0),
                               "detail": f"z={z:.1f}"})

        # ── 长时：静止检测 —
        #     隐状态稳定性 Z-score > 阈值（帧间比平时更相似）
        #     AND 预测误差低（世界模型不惊讶）
        if len(errs) >= self.STILLNESS_W:
            cos_arr = np.array(list(self._cos_sims)[-self.STILLNESS_W:])
            cos_z = (cos_arr.mean() - self._cos_mean) / self._cos_std

            low_count = int(np.sum(errs[-self.STILLNESS_W:] < self._error_p10))
            low_ratio = low_count / self.STILLNESS_W

            if cos_z > self.STILLNESS_Z_THRESH and low_ratio > 0.50:
                alerts.append({"type": "stillness",
                               "score": min(cos_z / 4, 1.0),
                               "detail": f"cos_z={cos_z:.1f} low_err={low_ratio:.2f}"})

        return alerts

    def _pad_to_64(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        if n < self.MAX_POINTS:
            extra = self.MAX_POINTS - n
            idx = np.random.randint(0, n, size=extra)
            jitter = np.random.randn(extra, 5).astype(np.float32) * 0.01
            return np.vstack([points, points[idx] + jitter])
        elif n > self.MAX_POINTS:
            return points[np.random.choice(n, self.MAX_POINTS, replace=False)]
        return points


# ── 主评估流程 ───────────────────────────────────────

def evaluate_scenario(yaml_path: str, speed: float = 1.0,
                      params_path: str | None = None,
                      dataset_path: str | None = None) -> EvalMetrics:

    engine = ScenarioEngine(yaml_path, speed=speed,
                            params_path=params_path, dataset_path=dataset_path)
    frames = engine.run()
    gt_anomalies = extract_gt_anomalies(frames)

    encoder, predictor, ok = load_world_model()
    if not ok:
        return EvalMetrics(scenario_name=engine._scenario_name, model_mode="unavailable")
    wm = WorldModelInference(encoder, predictor)

    ai_results: list[AnomalyResult] = []
    last_vital_alert_ts = 0
    last_offline_alert_ts = 0
    offline_gap_start = 0
    NoVit = type("NoVit", (), {"heart_rate": None, "resp_rate": None, "quality": None})

    for fg in frames:
        vit = fg.vitals or NoVit

        # ── 世界模型：点云异常 ──
        wm_points = fg.raw_points if fg.raw_points is not None else fg.points
        if wm_points is not None and len(wm_points) > 0:
            wm_alerts = wm.process_frame(wm_points)
            for a in wm_alerts:
                ai_results.append(AnomalyResult(
                    ts=fg.ts, device_id=fg.device_id, room=fg.room,
                    anomaly_score=a["score"],
                    anomaly_type={"sudden_spike": "fall",
                                  "sustained_shift": "pattern_deviation",
                                  "stillness": "stillness"}.get(a["type"], "pattern_deviation"),
                    severity="warning",
                    source="world_model",
                    details=a,
                ))

        # ── 生命体征阈值（医学知识）──
        hr, rr = vit.heart_rate, vit.resp_rate
        if hr is not None and rr is not None:
            if (hr <= HR_CRITICAL_LOW or hr >= HR_CRITICAL_HIGH or
                rr <= RR_CRITICAL_LOW or rr >= RR_CRITICAL_HIGH):
                if fg.ts - last_vital_alert_ts > 10000:  # 10s 内不重复
                    ai_results.append(AnomalyResult(
                        ts=fg.ts, device_id=fg.device_id, room=fg.room,
                        anomaly_score=0.9, anomaly_type="vital_anomaly",
                        severity="critical", source="vital_threshold",
                        details={"heart_rate": hr, "resp_rate": rr},
                    ))
                    last_vital_alert_ts = fg.ts

        # ── 离线检测 ──
        if fg.points is None and fg.vitals is None:
            if offline_gap_start == 0:
                offline_gap_start = fg.ts
            elif fg.ts - offline_gap_start >= OFFLINE_GAP_S * 1000:
                if fg.ts - last_offline_alert_ts > 30000:
                    ai_results.append(AnomalyResult(
                        ts=fg.ts, device_id=fg.device_id, room=fg.room,
                        anomaly_score=1.0, anomaly_type="offline",
                        severity="critical", source="infra",
                        details={"gap_s": (fg.ts - offline_gap_start) / 1000},
                    ))
                    last_offline_alert_ts = fg.ts
        else:
            offline_gap_start = 0

    # 匹配 + 指标
    matched, latencies = match_detections(gt_anomalies, ai_results)

    type_counts = {}
    for r in ai_results:
        type_counts[r.anomaly_type] = type_counts.get(r.anomaly_type, 0) + 1

    metrics = EvalMetrics(
        scenario_name=engine._scenario_name,
        total_frames=len(frames),
        total_gt_anomalies=len(gt_anomalies),
        detected_anomalies=len(ai_results),
        detection_latencies_ms=latencies,
        by_type=type_counts,
    )
    metrics.compute()
    return metrics


# ── 批量实验 ─────────────────────────────────────────

def run_batch(scenarios_dir: str, speed: float = 10.0, runs: int = 3,
              params_path: str | None = None, dataset_path: str | None = None) -> list[EvalMetrics]:
    results = []
    for sf in sorted(Path(scenarios_dir).glob("*.yaml")):
        scenario_metrics = []
        for run_i in range(runs):
            print(f"\n{'─'*40}\n场景: {sf.stem} (run {run_i+1}/{runs})")
            metrics = evaluate_scenario(str(sf), speed=speed,
                                        params_path=params_path, dataset_path=dataset_path)
            print(metrics.report())
            scenario_metrics.append(metrics)

        if runs > 1:
            avg_r = np.mean([m.recall for m in scenario_metrics])
            avg_p = np.mean([m.precision for m in scenario_metrics])
            avg_f = np.mean([m.f1 for m in scenario_metrics])
            print(f"  [{sf.stem}] {runs} runs avg: Recall={avg_r:.1%}  Prec={avg_p:.1%}  F1={avg_f:.3f}")

        results.extend(scenario_metrics)
    return results


# ── 诊断：原始 predictor error 分布 ──────────────────

def diagnose_predictor(yaml_path: str, speed: float = 10.0,
                       params_path: str | None = None,
                       dataset_path: str | None = None) -> dict:
    """
    逐帧输出 predictor error 和 GT 标签，用于评估 predictor 本身的区分度。
    不依赖 Z-score 检测逻辑。
    """
    engine = ScenarioEngine(yaml_path, speed=speed,
                            params_path=params_path, dataset_path=dataset_path)
    frames = engine.run()

    encoder, predictor, ok = load_world_model()
    if not ok:
        return {"error": "model_unavailable"}

    wm = WorldModelInference(encoder, predictor)
    errors = []
    gt_types = []
    timestamps = []

    for fg in frames:
        wm_points = fg.raw_points if fg.raw_points is not None else fg.points
        if wm_points is not None and len(wm_points) > 0:
            # 手动推进 WorldModelInference 但不依赖其 alert 逻辑
            n = len(wm_points)
            if n >= 4:
                pts = wm._pad_to_64(wm_points)
                wm._frame_buf.append(pts.astype(np.float32))
                if len(wm._frame_buf) > wm.WINDOW:
                    wm._frame_buf = wm._frame_buf[-wm.WINDOW:]

                if len(wm._frame_buf) == wm.WINDOW:
                    seq = np.stack(wm._frame_buf)
                    seq_t = torch.from_numpy(seq).unsqueeze(0)
                    with torch.no_grad():
                        out = encoder(seq_t)
                        S = out["S"]
                        pred_out = predictor(S[:, :-1])
                        error = 1.0 - torch_F.cosine_similarity(pred_out, S[:, -1], dim=-1)
                        errors.append(float(error.item()))
                        gt_types.append(fg.gt.anomaly_type if fg.gt else None)
                        timestamps.append(fg.ts)

    errors = np.array(errors)
    gt_types_arr = np.array(gt_types)
    is_anomaly = np.array([t is not None for t in gt_types_arr])

    normal_err = errors[~is_anomaly] if (~is_anomaly).any() else np.array([])
    anomaly_err = errors[is_anomaly] if is_anomaly.any() else np.array([])

    from sklearn.metrics import roc_auc_score as _roc_auc
    auc = _roc_auc(is_anomaly.astype(int), errors) if is_anomaly.any() and (~is_anomaly).any() else 0.0

    return {
        "scenario": engine._scenario_name,
        "n_frames": len(errors),
        "n_normal": int((~is_anomaly).sum()),
        "n_anomaly": int(is_anomaly.sum()),
        "error_mean": float(errors.mean()),
        "error_std": float(errors.std()),
        "normal_error_mean": float(normal_err.mean()) if len(normal_err) > 0 else 0,
        "normal_error_std": float(normal_err.std()) if len(normal_err) > 0 else 0,
        "anomaly_error_mean": float(anomaly_err.mean()) if len(anomaly_err) > 0 else 0,
        "anomaly_error_std": float(anomaly_err.std()) if len(anomaly_err) > 0 else 0,
        "auc": auc,
        "ratio": float(anomaly_err.mean() / (normal_err.mean() + 1e-10)) if len(anomaly_err) > 0 and len(normal_err) > 0 else 0,
    }


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI 评估 — 多时间尺度世界模型")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run")
    p_run.add_argument("scenario")
    p_run.add_argument("--speed", type=float, default=10.0)
    p_run.add_argument("--params")
    p_run.add_argument("--dataset")
    p_run.add_argument("--json", action="store_true")

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("scenarios_dir")
    p_batch.add_argument("--speed", type=float, default=10.0)
    p_batch.add_argument("--runs", type=int, default=3)
    p_batch.add_argument("--params")
    p_batch.add_argument("--dataset")

    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("scenario")
    p_diag.add_argument("--speed", type=float, default=10.0)
    p_diag.add_argument("--params")
    p_diag.add_argument("--dataset")

    args = parser.parse_args()

    if args.command == "run":
        metrics = evaluate_scenario(args.scenario, speed=args.speed,
                                    params_path=args.params, dataset_path=args.dataset)
        if args.json:
            print(json.dumps({"scenario": metrics.scenario_name, "recall": metrics.recall,
                              "precision": metrics.precision, "f1": metrics.f1,
                              "by_type": metrics.by_type}, ensure_ascii=False))
        else:
            print(metrics.report())

    elif args.command == "batch":
        run_batch(args.scenarios_dir, speed=args.speed, runs=args.runs,
                  params_path=args.params, dataset_path=args.dataset)

    elif args.command == "diagnose":
        result = diagnose_predictor(args.scenario, speed=args.speed,
                                    params_path=args.params, dataset_path=args.dataset)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
