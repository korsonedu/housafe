"""AI 评估桥接 — 仿真场景 → AI pipeline → ground truth 对比 → 指标报告

纯 in-process 执行，不需要 Redis/backend。比走 Redis 快 100x+，适合 CI 和批量实验。

用法:
  python -m simulator.eval run scenarios/elderly_day_normal.yaml
  python -m simulator.eval run scenarios/elderly_day_normal.yaml --speed 10 --params calibrated.npz
  python -m simulator.eval batch scenarios/ --runs 3
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

from simulator.scenario import ScenarioEngine
from simulator.types import FrameGroup, GroundTruth

# AI pipeline 组件（与 ai/main.py 共享）
from ai.shared.types import (
    FeatureVector, AnomalyResult, PointCloudFrame, VitalFrame, time_encode, POSTURES,
)
from ai.fallback.rules import FallbackEngine
from ai.latent_space.graph import SpatialGraph
from ai.baseline.gmm_baseline import PersonalBaseline
from ai.decoders.anomaly import AnomalyDecoder
from ai.decoders.notification import NotificationDecider


# ── 指标结构 ─────────────────────────────────────────

@dataclass
class EvalMetrics:
    scenario_name: str = ""
    total_frames: int = 0
    total_gt_anomalies: int = 0       # ground truth 异常段数
    detected_anomalies: int = 0        # AI 检测到的异常段数
    true_positives: int = 0            # 匹配到的 GT 异常
    false_positives: int = 0           # AI 报警但无 GT 匹配
    false_negatives: int = 0           # GT 异常但 AI 未报警
    detection_latencies_ms: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

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
            f"  总帧数:           {self.total_frames}",
            f"  GT 异常段:        {self.total_gt_anomalies}",
            f"  AI 检测异常:      {self.detected_anomalies}",
            f"  正确检测 (TP):    {self.true_positives}",
            f"  误报 (FP):        {self.false_positives}",
            f"  漏报 (FN):        {self.false_negatives}",
            f"  ─────────────────────────────",
            f"  精确率 (Precision): {self.precision:.1%}",
            f"  召回率 (Recall):    {self.recall:.1%}",
            f"  F1:                 {self.f1:.3f}",
            f"  平均检测延迟:       {self.avg_latency_ms:.0f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── 特征提取（从 ai/main.py 移植）────────────────────

def extract_features_from_points(ts: int, device_id: str, room: str,
                                  points: np.ndarray) -> FeatureVector:
    """从点云数组提取特征向量"""
    n = len(points) if points is not None else 0

    if n == 0 or points is None:
        return FeatureVector(
            ts=ts, device_id=device_id, room=room,
            posture="stand", posture_confidence=0.0, presence=False,
            moving=False, centroid=(0.0, 0.0, 0.0), height=0.0, n_points=0,
            occupancy_estimate=0, resp_rate=None, heart_rate=None,
            vital_quality=None, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.0,
        )

    centroid = tuple(points[:, :3].mean(axis=0).astype(float))
    zs = points[:, 2]
    height = float(zs.max() - zs.min())

    if height < 0.3:
        posture = "lie"
    elif height < 0.8:
        posture = "sit"
    elif height >= 1.8:
        posture = "walk"
    else:
        posture = "stand"

    velocities = points[:, 3]
    moving = bool(np.mean(np.abs(velocities)) > 0.1)
    velocity_variance = float(np.var(velocities))

    occupancy_estimate = 1 if n >= 5 else 0

    intensities = points[:, 4]
    confidence = min(float(np.mean(intensities)), 1.0) if n > 0 else 0.0

    dt = datetime.fromtimestamp(ts / 1000.0, tz=None)
    # Use UTC-equivalent local time for hour/weekday encoding
    hour = dt.hour + dt.minute / 60.0
    h_sin, h_cos = time_encode(hour)
    weekday = dt.weekday()

    return FeatureVector(
        ts=ts, device_id=device_id, room=room,
        posture=posture, posture_confidence=round(confidence, 3),
        presence=True, moving=moving,
        centroid=centroid, height=round(height, 3),
        n_points=n, occupancy_estimate=occupancy_estimate,
        resp_rate=None, heart_rate=None, vital_quality=None,
        hour_sin=h_sin, hour_cos=h_cos, weekday=weekday,
        velocity_variance=round(velocity_variance, 3),
    )


def merge_vitals(fv: FeatureVector, heart_rate: float | None,
                 resp_rate: float | None, quality: float | None):
    """将体征数据合并到已有 FeatureVector"""
    fv.resp_rate = resp_rate
    fv.heart_rate = heart_rate
    fv.vital_quality = quality


# ── GT 异常段提取 ────────────────────────────────────

def extract_gt_anomalies(frames: list[FrameGroup]) -> list[dict]:
    """从帧序列提取 ground truth 异常段"""
    anomalies = []
    current = None
    for f in frames:
        if f.gt and f.gt.anomaly_type:
            if current is None:
                current = {
                    "type": f.gt.anomaly_type,
                    "severity": f.gt.anomaly_severity,
                    "start_ts": f.ts,
                    "end_ts": f.ts,
                    "start_frame": f.frame_id,
                }
            else:
                current["end_ts"] = f.ts
        else:
            if current is not None:
                anomalies.append(current)
                current = None
    if current is not None:
        anomalies.append(current)
    return anomalies


# ── 匹配逻辑 ─────────────────────────────────────────

MATCH_WINDOW_MS = 5000  # ±5s 时间窗口内算匹配


def match_detections(gt_anomalies: list[dict], ai_results: list[AnomalyResult]) -> tuple[int, list[float]]:
    """
    将 AI 检测结果与 GT 异常匹配。
    Returns: (matched_count, latencies_ms)
    """
    matched = 0
    latencies = []
    matched_gt = set()
    matched_ai = set()

    for gi, gt in enumerate(gt_anomalies):
        for ai_idx, result in enumerate(ai_results):
            if ai_idx in matched_ai:
                continue
            # 时间窗口匹配：AI 检测时间在 GT 时间窗口内或附近
            ts_diff = abs(result.ts - gt["start_ts"])
            ts_diff_end = abs(result.ts - gt["end_ts"])
            if ts_diff <= MATCH_WINDOW_MS or ts_diff_end <= MATCH_WINDOW_MS:
                matched += 1
                latencies.append(result.ts - gt["start_ts"])
                matched_gt.add(gi)
                matched_ai.add(ai_idx)
                break

    return matched, latencies


# ── 主评估流程 ───────────────────────────────────────

def evaluate_scenario(yaml_path: str, speed: float = 1.0,
                      params_path: str | None = None,
                      dataset_path: str | None = None) -> EvalMetrics:
    """
    运行完整评估流程：
    1. 加载场景 → 生成帧序列
    2. 帧 → 特征提取 → AnomalyDecoder
    3. AI 检测结果 vs Ground Truth
    4. 返回指标
    """
    # 1. 生成场景帧
    engine = ScenarioEngine(yaml_path, speed=speed,
                            params_path=params_path, dataset_path=dataset_path)
    frames = engine.run()
    scenario_name = engine._scenario_name

    # 2. 提取 GT 异常
    gt_anomalies = extract_gt_anomalies(frames)

    # 3. 初始化 AI pipeline
    fallback = FallbackEngine()
    graph = SpatialGraph()
    baseline = PersonalBaseline()
    anomaly_decoder = AnomalyDecoder(fallback, baseline, graph)
    notifier = NotificationDecider()

    # 4. 逐帧处理
    feature_buffer: deque[FeatureVector] = deque(maxlen=2000)
    recent_decisions: deque = deque(maxlen=100)
    ai_results: list[AnomalyResult] = []
    ai_anomaly_count = 0
    prev_feature: FeatureVector | None = None

    for fg in frames:
        # 点云 → FeatureVector
        if fg.points is not None and len(fg.points) > 0:
            fv = extract_features_from_points(fg.ts, fg.device_id, fg.room, fg.points)
        elif prev_feature is not None:
            # 离线帧：复用上一帧特征
            fv = FeatureVector(
                ts=fg.ts, device_id=fg.device_id, room=fg.room,
                posture=prev_feature.posture,
                posture_confidence=prev_feature.posture_confidence,
                presence=False, moving=False,
                centroid=prev_feature.centroid, height=prev_feature.height,
                n_points=0, occupancy_estimate=0,
                resp_rate=prev_feature.resp_rate, heart_rate=prev_feature.heart_rate,
                vital_quality=prev_feature.vital_quality,
                hour_sin=prev_feature.hour_sin, hour_cos=prev_feature.hour_cos,
                weekday=prev_feature.weekday, velocity_variance=0.0,
            )
        else:
            continue

        # 合并体征
        if fg.vitals:
            merge_vitals(fv, fg.vitals.heart_rate, fg.vitals.resp_rate, fg.vitals.quality)

        prev_feature = fv

        # 更新 buffer
        feature_buffer.append(fv)
        history = list(feature_buffer)

        # 异常检测
        anomaly = anomaly_decoder.evaluate(fv, history)
        if anomaly is not None:
            ai_results.append(anomaly)

            decision = notifier.decide(anomaly, list(recent_decisions))
            recent_decisions.append(decision)

            if decision.level != "none":
                ai_anomaly_count += 1

    # 5. 匹配
    matched, latencies = match_detections(gt_anomalies, ai_results)

    # 6. 指标
    metrics = EvalMetrics(
        scenario_name=scenario_name,
        total_frames=len(frames),
        total_gt_anomalies=len(gt_anomalies),
        detected_anomalies=ai_anomaly_count,
        detection_latencies_ms=latencies,
    )
    metrics.compute()

    return metrics


# ── 批量实验 ─────────────────────────────────────────

def run_batch(scenarios_dir: str, speed: float = 10.0, runs: int = 3,
              params_path: str | None = None,
              dataset_path: str | None = None) -> list[EvalMetrics]:
    """批量运行场景目录下所有 YAML"""
    results = []
    scenario_files = sorted(Path(scenarios_dir).glob("*.yaml"))
    if not scenario_files:
        print(f"无场景文件: {scenarios_dir}")
        return results

    for sf in scenario_files:
        for run_i in range(runs):
            print(f"\n{'─'*40}")
            print(f"场景: {sf.stem} (run {run_i+1}/{runs})")
            start = time.monotonic()
            metrics = evaluate_scenario(str(sf), speed=speed,
                                        params_path=params_path,
                                        dataset_path=dataset_path)
            elapsed = time.monotonic() - start
            metrics.scenario_name = f"{sf.stem} (run {run_i+1})"
            print(metrics.report())
            print(f"  评估耗时: {elapsed:.1f}s")
            results.append(metrics)

    # 汇总
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  汇总 ({len(results)} 次实验)")
        print(f"{'='*60}")
        avg_recall = np.mean([r.recall for r in results])
        avg_precision = np.mean([r.precision for r in results])
        avg_f1 = np.mean([r.f1 for r in results])
        avg_latency = np.mean([r.avg_latency_ms for r in results if r.avg_latency_ms > 0])
        print(f"  平均召回率:    {avg_recall:.1%}")
        print(f"  平均精确率:    {avg_precision:.1%}")
        print(f"  平均 F1:       {avg_f1:.3f}")
        print(f"  平均延迟:      {avg_latency:.0f} ms")

    return results


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI 评估桥接 — 仿真→AI→GT 对比")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="评估单个场景")
    p_run.add_argument("scenario", help="场景 YAML 路径")
    p_run.add_argument("--speed", type=float, default=10.0,
                       help="时间加速倍率 (default: 10)")
    p_run.add_argument("--params", help="校准参数 npz")
    p_run.add_argument("--dataset", help="3DPCHM 数据集（replay 模式）")
    p_run.add_argument("--json", action="store_true", help="输出 JSON 格式")

    # batch
    p_batch = sub.add_parser("batch", help="批量评估")
    p_batch.add_argument("scenarios_dir", help="场景目录")
    p_batch.add_argument("--speed", type=float, default=10.0)
    p_batch.add_argument("--runs", type=int, default=3)
    p_batch.add_argument("--params", help="校准参数 npz")
    p_batch.add_argument("--dataset", help="3DPCHM 数据集")

    args = parser.parse_args()

    if args.command == "run":
        metrics = evaluate_scenario(args.scenario, speed=args.speed,
                                    params_path=args.params,
                                    dataset_path=args.dataset)
        if args.json:
            print(json.dumps({
                "scenario": metrics.scenario_name,
                "recall": metrics.recall,
                "precision": metrics.precision,
                "f1": metrics.f1,
                "avg_latency_ms": metrics.avg_latency_ms,
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "fn": metrics.false_negatives,
            }, ensure_ascii=False))
        else:
            print(metrics.report())

    elif args.command == "batch":
        run_batch(args.scenarios_dir, speed=args.speed, runs=args.runs,
                  params_path=args.params, dataset_path=args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
