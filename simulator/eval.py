"""AI 评估桥接 — 纯世界模型异常检测。

不依赖规则引擎、体征、姿态分类。Encoder → Predictor 预测误差 → 自适应阈值 → 异常。

用法:
  python -m simulator.eval run scenarios/elderly_day_replay.yaml --dataset <npz>
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
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
    model_mode: str = "world_model"

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
            "=" * 60,
        ]
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
    matched = 0
    latencies = []
    matched_ai = set()
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

        enc_m = sum(p.numel() for p in encoder.parameters())
        pred_m = sum(p.numel() for p in predictor.parameters())
        print(f"[Eval] 世界模型已加载  encoder={enc_m/1e6:.1f}M  predictor={pred_m/1e6:.1f}M")
        return encoder, predictor, True
    except Exception as e:
        print(f"[Eval] 模型加载失败: {e}")
        return None, None, False


# ── 世界模型推理器 ───────────────────────────────────

class WorldModelInference:
    """训练路径推理 + 局部突变检测。
    异常 = predictor error 突然飙升超过局部滚动基线的 3σ。"""

    WINDOW = 32
    MAX_POINTS = 64
    WARMUP_WINDOWS = 60    # 需要这么多窗口才能开始检测
    LOCAL_WINDOW = 30      # 局部基线窗口
    Z_THRESHOLD = 3.0      # Z-score 阈值

    def __init__(self, encoder: EncoderModel, predictor: TinyPredictor):
        self.encoder = encoder
        self.predictor = predictor
        self._frame_buf: list[np.ndarray] = []
        self._errors: list[float] = []
        self._ready = False

    def process_frame(self, points: np.ndarray) -> float:
        """返回 Z-score。> Z_THRESHOLD 表示局部异常突变"""
        n = len(points) if points is not None else 0
        if n < 4:
            self._frame_buf.append(np.zeros((self.MAX_POINTS, 5), dtype=np.float32))
            return 0.0

        if n < self.MAX_POINTS:
            extra = self.MAX_POINTS - n
            idx = np.random.randint(0, n, size=extra)
            jitter = np.random.randn(extra, 5).astype(np.float32) * 0.01
            pts = np.vstack([points, points[idx] + jitter])
        elif n > self.MAX_POINTS:
            pts = points[np.random.choice(n, self.MAX_POINTS, replace=False)]
        else:
            pts = points

        self._frame_buf.append(pts.astype(np.float32))
        if len(self._frame_buf) > self.WINDOW:
            self._frame_buf = self._frame_buf[-self.WINDOW:]

        if len(self._frame_buf) < self.WINDOW:
            return 0.0

        seq = np.stack(self._frame_buf)
        seq_t = torch.from_numpy(seq).unsqueeze(0)

        with torch.no_grad():
            out = self.encoder(seq_t)
            S = out["S"]
            pred_out = self.predictor(S[:, :-1])
            error = 1.0 - torch_F.cosine_similarity(pred_out, S[:, -1], dim=-1)
            error = float(error.item())

        self._errors.append(error)

        if len(self._errors) < self.WARMUP_WINDOWS:
            return 0.0

        self._ready = True

        # 局部 Z-score：当前误差 vs 最近 LOCAL_WINDOW 帧的分布
        local = np.array(self._errors[-self.LOCAL_WINDOW:])
        mu, sigma = local.mean(), local.std()
        if sigma < 1e-6:
            return 0.0

        z = (error - mu) / sigma
        return z if z > self.Z_THRESHOLD else 0.0


# ── 主评估流程 ───────────────────────────────────────

def evaluate_scenario(yaml_path: str, speed: float = 1.0,
                      params_path: str | None = None,
                      dataset_path: str | None = None) -> EvalMetrics:

    # 1. 生成场景帧
    engine = ScenarioEngine(yaml_path, speed=speed,
                            params_path=params_path, dataset_path=dataset_path)
    frames = engine.run()

    # 2. GT 异常
    gt_anomalies = extract_gt_anomalies(frames)

    # 3. 世界模型
    encoder, predictor, ok = load_world_model()
    if not ok:
        return EvalMetrics(scenario_name=engine._scenario_name, model_mode="unavailable")
    wm = WorldModelInference(encoder, predictor)

    # 4. 纯世界模型：Encoder → Predictor 误差 → 阈值
    ai_results: list[AnomalyResult] = []

    for fg in frames:
        wm_points = fg.raw_points if fg.raw_points is not None else fg.points
        if wm_points is None or len(wm_points) == 0:
            continue

        z_score = wm.process_frame(wm_points)
        if z_score > 0:
            ai_results.append(AnomalyResult(
                ts=fg.ts, device_id=fg.device_id, room=fg.room,
                anomaly_score=min(z_score / (wm.Z_THRESHOLD * 2), 1.0),
                anomaly_type="pattern_deviation",
                severity="warning" if z_score > wm.Z_THRESHOLD * 1.5 else "info",
                source="world_model",
                details={"z_score": round(z_score, 4)},
            ))

    # 5. 匹配 + 指标
    matched, latencies = match_detections(gt_anomalies, ai_results)
    metrics = EvalMetrics(
        scenario_name=engine._scenario_name,
        total_frames=len(frames),
        total_gt_anomalies=len(gt_anomalies),
        detected_anomalies=len(ai_results),
        detection_latencies_ms=latencies,
        model_mode="world_model",
    )
    metrics.compute()
    return metrics


# ── 批量实验 ─────────────────────────────────────────

def run_batch(scenarios_dir: str, speed: float = 10.0, runs: int = 3,
              params_path: str | None = None, dataset_path: str | None = None) -> list[EvalMetrics]:
    results = []
    for sf in sorted(Path(scenarios_dir).glob("*.yaml")):
        for run_i in range(runs):
            print(f"\n{'─'*40}\n场景: {sf.stem} (run {run_i+1}/{runs})")
            metrics = evaluate_scenario(str(sf), speed=speed,
                                        params_path=params_path, dataset_path=dataset_path)
            print(metrics.report())
            results.append(metrics)

    if len(results) > 1:
        avg_r = np.mean([r.recall for r in results])
        avg_p = np.mean([r.precision for r in results])
        avg_f = np.mean([r.f1 for r in results])
        print(f"\n{'='*60}\n  汇总 ({len(results)} 次): "
              f"Recall={avg_r:.1%}  Prec={avg_p:.1%}  F1={avg_f:.3f}")
    return results


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI 评估 — 纯世界模型异常检测")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="评估单场景")
    p_run.add_argument("scenario")
    p_run.add_argument("--speed", type=float, default=10.0)
    p_run.add_argument("--params")
    p_run.add_argument("--dataset")
    p_run.add_argument("--json", action="store_true")

    p_batch = sub.add_parser("batch", help="批量评估")
    p_batch.add_argument("scenarios_dir")
    p_batch.add_argument("--speed", type=float, default=10.0)
    p_batch.add_argument("--runs", type=int, default=3)
    p_batch.add_argument("--params")
    p_batch.add_argument("--dataset")

    args = parser.parse_args()

    if args.command == "run":
        metrics = evaluate_scenario(args.scenario, speed=args.speed,
                                    params_path=args.params, dataset_path=args.dataset)
        if args.json:
            print(json.dumps({"scenario": metrics.scenario_name, "recall": metrics.recall,
                              "precision": metrics.precision, "f1": metrics.f1,
                              "tp": metrics.true_positives, "fp": metrics.false_positives,
                              "fn": metrics.false_negatives}, ensure_ascii=False))
        else:
            print(metrics.report())

    elif args.command == "batch":
        run_batch(args.scenarios_dir, speed=args.speed, runs=args.runs,
                  params_path=args.params, dataset_path=args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
