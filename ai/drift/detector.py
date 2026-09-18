"""事前健康管理 — 跨时间尺度行为漂移检测。

Direction A: DistributionDriftDetector
  时段分布比较 — 从"这个 S_t 正常吗"到"今天的分布正常吗"
  在 LatentBaseline per-context GMM 之上，用日均 NLL 量化分布漂移。

Direction C: ExplainableDriftDetector
  可解释漂移分解 — "走路时间减少 40%，久坐时间增加 60%"
  用 Encoder 分类头给 S_t 打动作标签，对比基线与当日动作分布。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from ai.baseline.latent_baseline import LatentBaseline, context_key, _utc_ms_to_datetime


def _mahalanobis_dist(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    try:
        inv_cov = np.linalg.inv(cov)
        diff = x - mean
        return float(np.sqrt(diff @ inv_cov @ diff))
    except np.linalg.LinAlgError:
        return 10.0


# ── 共享类型 ───────────────────────────────────────────

@dataclass
class DayDriftResult:
    """单日分布漂移检测结果 (Direction A)"""
    overall_score: float = 0.0          # 0-1, 0=与基线一致 1=严重漂移
    per_context: dict[str, float] = field(default_factory=dict)  # ctx → drift score
    context_n_frames: dict[str, int] = field(default_factory=dict)
    dominant_context: str = ""           # 漂移最大的上下文
    dominant_score: float = 0.0
    n_total_frames: int = 0
    mean_nll: float = 0.0               # 当日平均 NLL
    baseline_nll_mean: float = 0.0      # 基线平均 NLL（跨上下文加权）


@dataclass
class ActionDriftResult:
    """可解释动作分布漂移 (Direction C)"""
    overall_drift: float = 0.0           # JS 距离 0-1
    baseline_dist: dict[str, float] = field(default_factory=dict)  # action → prob
    current_dist: dict[str, float] = field(default_factory=dict)
    top_changes: list[dict] = field(default_factory=list)  # [{action, delta, direction}]
    summary: str = ""                    # 人类可读摘要
    per_context: dict[str, dict] = field(default_factory=dict)  # ctx → ActionDriftResult


# ── Action labels ──────────────────────────────────────

ACTION_LABELS = [
    "stand", "walk", "sit", "lie", "squat",
    "lean_left", "lean_right", "fall_forward",
    "fall_backward", "fall_left", "fall_right", "fall_up",
]

# 健康相关动作组
HEALTH_RELEVANT = {
    "mobility": ["walk", "stand", "squat"],
    "sedentary": ["sit", "lie"],
    "balance": ["lean_left", "lean_right"],
    "fall": ["fall_forward", "fall_backward", "fall_left", "fall_right", "fall_up"],
}


# ── Direction A: 分布漂移检测 ──────────────────────────

class DistributionDriftDetector:
    """在 LatentBaseline 之上做时段分布漂移检测。

    核心思路：
      - 基线：LatentBaseline 的 per-context GMM 已知
      - 当日：所有帧在该 context GMM 下的 NLL 均值
      - 漂移分 = 当日 mean NLL 在基线 NLL 分布中的 percentile

    与 LatentBaseline.score() 的区别：
      - score()：单帧 → "这个 S_t 正常吗"（秒级）
      - score_day()：当日所有帧 → "今天的分布正常吗"（天级）
    """

    def __init__(self, baseline: LatentBaseline, use_nll: bool = True):
        self.baseline = baseline
        self.use_nll = use_nll

    @property
    def is_ready(self) -> bool:
        return self.baseline.is_ready

    def score_frames(self, S: np.ndarray, timestamps_ms: np.ndarray) -> dict:
        """对一组帧计算 NLL 统计。不聚合到天级——由上层决定聚合窗口。

        Args:
            S: (N, 256) float32
            timestamps_ms: (N,) int64

        Returns:
            {
                "mean_nll": 当日平均 NLL,
                "median_nll": 中位数 NLL,
                "per_context": {ctx: {"mean_nll", "n", "drift_score"}},
                "overall_drift_score": 0-1,
            }
        """
        if not self.is_ready or len(S) == 0:
            return {"mean_nll": 0, "median_nll": 0, "per_context": {},
                    "overall_drift_score": 0.0, "n_frames": 0}

        # PCA transform
        reduced = self.baseline._pca.transform(S)  # (N, pca_dim)

        # 分上下文
        ctx_groups: dict[str, list[int]] = defaultdict(list)
        for i, ts in enumerate(timestamps_ms):
            dt = _utc_ms_to_datetime(ts)
            ctx = context_key(dt["weekday"], dt["hour"])
            ctx_groups[ctx].append(i)

        all_nlls = []
        per_context = {}
        context_drift_scores = []

        for ctx, indices in ctx_groups.items():
            gmm = self.baseline._gmms.get(ctx)
            if gmm is None:
                nearest = self.baseline._find_nearest_context(ctx)
                if nearest is None:
                    continue
                gmm = self.baseline._gmms[nearest]
                ctx_used = nearest
            else:
                ctx_used = ctx

            ctx_reduced = reduced[indices]  # (n, pca_dim)

            if self.use_nll:
                log_probs = gmm.score_samples(ctx_reduced)
                scores_per_frame = -log_probs  # NLL
            else:
                # Mahalanobis distance to nearest component
                scores_per_frame = np.array([
                    min(_mahalanobis_dist(ctx_reduced[i], gmm.means_[k], gmm.covariances_[k])
                        for k in range(gmm.n_components))
                    for i in range(len(ctx_reduced))
                ])

            all_nlls.extend(scores_per_frame.tolist())
            mean_score = float(np.mean(scores_per_frame))

            if self.use_nll:
                stats = self.baseline._nll_stats.get(ctx_used, {})
                baseline_mean = stats.get("mean", mean_score)
                baseline_std = stats.get("std", 1.0)
            else:
                # Mahalanobis 的基线统计用 historical scores
                hist = self.baseline._scores.get(ctx_used, [])
                if hist:
                    baseline_mean = float(np.mean(hist))
                    baseline_std = float(np.std(hist)) if np.std(hist) > 0 else 1.0
                else:
                    baseline_mean = mean_score
                    baseline_std = 1.0

            # Z-score → drift score（单边）
            if baseline_std > 1e-10:
                z = max(0.0, (mean_score - baseline_mean) / baseline_std)
                drift = float(np.clip(z / 3.0, 0.0, 1.0))
            else:
                drift = 0.0

            per_context[ctx] = {
                "mean_nll": mean_score,
                "n": len(indices),
                "drift_score": drift,
                "baseline_nll_mean": baseline_mean,
            }
            context_drift_scores.append((drift, len(indices), ctx))

        if not all_nlls:
            return {"mean_nll": 0, "median_nll": 0, "per_context": {},
                    "overall_drift_score": 0.0, "n_frames": 0}

        all_nlls = np.array(all_nlls)
        mean_nll = float(np.mean(all_nlls))
        median_nll = float(np.median(all_nlls))

        # 整体漂移分：各 context drift score 的帧数加权平均
        total_n = sum(n for _, n, _ in context_drift_scores)
        if total_n > 0:
            overall = sum(s * n for s, n, _ in context_drift_scores) / total_n
        else:
            overall = 0.0

        # 找漂移最大的 context
        dominant_ctx = max(context_drift_scores, key=lambda x: x[0]) if context_drift_scores else ("", 0, "")

        return {
            "mean_nll": mean_nll,
            "median_nll": median_nll,
            "per_context": per_context,
            "overall_drift_score": overall,
            "n_frames": len(S),
            "dominant_context": dominant_ctx[2],
            "dominant_score": dominant_ctx[0],
        }

    def score_day(self, S: np.ndarray, timestamps_ms: np.ndarray) -> DayDriftResult:
        """计算单日漂移分。对 score_frames() 的包装，返回结构化结果。"""
        raw = self.score_frames(S, timestamps_ms)

        # 计算加权 baseline NLL mean
        baseline_nlls = []
        for ctx, info in raw["per_context"].items():
            baseline_nlls.append(info.get("baseline_nll_mean", 0))
        baseline_mean = float(np.mean(baseline_nlls)) if baseline_nlls else 0.0

        return DayDriftResult(
            overall_score=raw["overall_drift_score"],
            per_context={ctx: info["drift_score"] for ctx, info in raw["per_context"].items()},
            context_n_frames={ctx: info["n"] for ctx, info in raw["per_context"].items()},
            dominant_context=raw["dominant_context"],
            dominant_score=raw["dominant_score"],
            n_total_frames=raw["n_frames"],
            mean_nll=raw["mean_nll"],
            baseline_nll_mean=baseline_mean,
        )

    def score_window(self, S: np.ndarray, timestamps_ms: np.ndarray,
                     window_hours: float = 24.0) -> list[dict]:
        """滑动窗口漂移检测。用于连续监控。

        将数据按 window_hours 切分，每段计算漂移分。
        返回按时间排序的漂移分序列。
        """
        if len(S) == 0:
            return []

        ts_sorted = np.argsort(timestamps_ms)
        S_sorted = S[ts_sorted]
        ts_sorted = timestamps_ms[ts_sorted]

        window_ms = window_hours * 3600 * 1000
        results = []

        # 简化：不重叠的窗口
        start_ts = ts_sorted[0]
        window_start = start_ts
        window_start_idx = 0

        for i, ts in enumerate(ts_sorted):
            if ts - window_start >= window_ms:
                result = self.score_frames(
                    S_sorted[window_start_idx:i],
                    ts_sorted[window_start_idx:i],
                )
                result["window_start_ts"] = int(window_start)
                result["window_end_ts"] = int(ts)
                results.append(result)
                window_start = ts
                window_start_idx = i

        # 最后一段
        if window_start_idx < len(ts_sorted):
            result = self.score_frames(
                S_sorted[window_start_idx:],
                ts_sorted[window_start_idx:],
            )
            result["window_start_ts"] = int(window_start)
            result["window_end_ts"] = int(ts_sorted[-1])
            results.append(result)

        return results


# ── Direction C: 可解释漂移分解 ────────────────────────

class ExplainableDriftDetector:
    """用 Encoder 分类头给 S_t 打动作标签，对比基线 vs 当日动作分布。

    不是 "漂移了 0.3"，而是 "walk 占有率从 45% 降到 28%，sit 从 30% 升到 52%"。

    Usage:
      edd = ExplainableDriftDetector(encoder)
      edd.fit_baseline(S_baseline, timestamps_baseline)  # 记录基线动作分布
      result = edd.compare(S_today, timestamps_today)     # 与基线比较
      print(result.summary)  # 人类可读
    """

    def __init__(self, encoder, window_size: int = 32, device: str = "cpu"):
        """
        Args:
            encoder: EncoderModel（已加载权重，eval 模式）
            window_size: 分类时取均值的 S_t 窗口大小（匹配训练时的 T）
            device: torch device
        """
        self.encoder = encoder
        self.window_size = window_size
        self.device = device
        self._baseline_action_dist: dict[str, np.ndarray] | None = None  # ctx → (12,) prob
        self._baseline_counts: dict[str, int] | None = None
        self._baseline_global_dist: np.ndarray | None = None  # (12,) global prob

    def classify_frames(self, S: np.ndarray) -> np.ndarray:
        """对 S_t 序列做动作分类。

        S: (N, 256) float32
        Returns: (N, 12) action probabilities

        分类头期望 S_mean (mean over T frames)。
        这里用滑动窗口取均值后分类。
        """
        import torch

        N = S.shape[0]
        if N == 0:
            return np.array([])

        probs = np.zeros((N, 12), dtype=np.float32)

        half = self.window_size // 2
        for i in range(N):
            start = max(0, i - half)
            end = min(N, i + half)
            window = S[start:end]  # (w, 256)
            # pad if needed
            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                window = np.vstack([window, np.tile(window[-1], (pad_len, 1))])

            S_mean = window.mean(axis=0)  # (256,)
            S_mean_t = torch.from_numpy(S_mean).unsqueeze(0).to(self.device)  # (1, 256)

            with torch.no_grad():
                logits = self.encoder.classifier(S_mean_t)  # (1, 12)
                p = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            probs[i] = p

        return probs

    def _action_distribution(self, S: np.ndarray, timestamps_ms: np.ndarray,
                             per_context: bool = False) -> dict:
        """从 S_t 序列计算动作分布。

        Returns:
            如果 per_context=False: {"global": (12,) prob, "n_frames": int}
            如果 per_context=True: 上述 + {"by_context": {ctx: (12,) prob}}
        """
        if len(S) == 0:
            return {"global": np.zeros(12), "n_frames": 0}

        probs = self.classify_frames(S)  # (N, 12)
        global_dist = probs.mean(axis=0)  # (12,)

        result = {"global": global_dist, "n_frames": len(S)}

        if per_context:
            by_context: dict[str, list[np.ndarray]] = defaultdict(list)
            for i, ts in enumerate(timestamps_ms):
                dt = _utc_ms_to_datetime(ts)
                ctx = context_key(dt["weekday"], dt["hour"])
                by_context[ctx].append(probs[i])

            ctx_dists = {}
            for ctx, ctx_probs in by_context.items():
                ctx_dists[ctx] = np.mean(ctx_probs, axis=0)

            result["by_context"] = ctx_dists

        return result

    def fit_baseline(self, S: np.ndarray, timestamps_ms: np.ndarray):
        """记录基线动作分布。"""
        dist = self._action_distribution(S, timestamps_ms, per_context=True)
        self._baseline_global_dist = dist["global"]
        self._baseline_action_dist = dist.get("by_context", {})
        self._baseline_counts = {"total": dist["n_frames"]}

    def compare(self, S: np.ndarray, timestamps_ms: np.ndarray) -> ActionDriftResult:
        """比较当日动作分布与基线分布，生成可解释漂移报告。"""
        if self._baseline_global_dist is None:
            return ActionDriftResult(summary="基线未就绪，请先调用 fit_baseline()")

        current = self._action_distribution(S, timestamps_ms, per_context=True)
        current_global = current["global"]
        baseline_global = self._baseline_global_dist

        # JS 距离（对称 KL）
        js_dist = _js_divergence(baseline_global, current_global)

        # 每动作变化
        changes = []
        for i, name in enumerate(ACTION_LABELS):
            delta = float(current_global[i] - baseline_global[i])
            if abs(delta) > 0.005:  # 过滤 0.5pp 以下的变化
                changes.append({
                    "action": name,
                    "baseline_pct": round(float(baseline_global[i]) * 100, 1),
                    "current_pct": round(float(current_global[i]) * 100, 1),
                    "delta_pct": round(delta * 100, 1),
                    "direction": "increase" if delta > 0 else "decrease",
                })

        changes.sort(key=lambda c: abs(c["delta_pct"]), reverse=True)

        # 健康相关分组变化
        group_changes = {}
        for group, actions in HEALTH_RELEVANT.items():
            idxs = [ACTION_LABELS.index(a) for a in actions if a in ACTION_LABELS]
            base_sum = float(sum(baseline_global[i] for i in idxs))
            curr_sum = float(sum(current_global[i] for i in idxs))
            group_changes[group] = {
                "baseline_pct": round(base_sum * 100, 1),
                "current_pct": round(curr_sum * 100, 1),
                "delta_pct": round((curr_sum - base_sum) * 100, 1),
            }

        # 生成摘要
        summary = _generate_summary(changes, group_changes, js_dist)

        # Per-context 比较
        per_context = {}
        for ctx in self._baseline_action_dist:
            if ctx in current.get("by_context", {}):
                base_d = self._baseline_action_dist[ctx]
                curr_d = current["by_context"][ctx]
                ctx_js = _js_divergence(base_d, curr_d)
                ctx_changes = []
                for i, name in enumerate(ACTION_LABELS):
                    d = float(curr_d[i] - base_d[i])
                    if abs(d) > 0.01:
                        ctx_changes.append({
                            "action": name,
                            "delta_pct": round(d * 100, 1),
                            "direction": "increase" if d > 0 else "decrease",
                        })
                ctx_changes.sort(key=lambda c: abs(c["delta_pct"]), reverse=True)
                per_context[ctx] = {
                    "js_distance": float(ctx_js),
                    "top_changes": ctx_changes[:3],
                }

        return ActionDriftResult(
            overall_drift=float(js_dist),
            baseline_dist={ACTION_LABELS[i]: round(float(baseline_global[i]), 4) for i in range(12)},
            current_dist={ACTION_LABELS[i]: round(float(current_global[i]), 4) for i in range(12)},
            top_changes=changes[:5],
            summary=summary,
            per_context=per_context,
        )


# ── 辅助函数 ───────────────────────────────────────────

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions. 0-1 range."""
    # 平滑
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    # normalize: max JS = ln(2)
    return float(min(js / np.log(2), 1.0))


def _generate_summary(changes: list[dict], group_changes: dict,
                      js_dist: float) -> str:
    """生成人类可读的漂移摘要。"""
    parts = []

    # 总体漂移级别
    if js_dist < 0.02:
        parts.append("行为模式与基线基本一致")
    elif js_dist < 0.05:
        parts.append("行为模式有轻微变化")
    elif js_dist < 0.10:
        parts.append("行为模式有明显漂移")
    else:
        parts.append("行为模式发生显著变化")

    parts.append(f"(JS距离={js_dist:.3f})")

    # 健康相关分组
    significant_groups = []
    for group, info in group_changes.items():
        if abs(info["delta_pct"]) >= 3:
            direction = "增加" if info["delta_pct"] > 0 else "减少"
            significant_groups.append(
                f"{_group_label(group)}{direction}{abs(info['delta_pct']):.0f}%"
            )

    if significant_groups:
        parts.append("；".join(significant_groups))

    # 具体动作变化 top 2
    if changes:
        top2 = changes[:2]
        action_parts = []
        for c in top2:
            direction = "↑" if c["direction"] == "increase" else "↓"
            action_parts.append(
                f"{c['action']} {c['baseline_pct']}%→{c['current_pct']}% ({direction}{abs(c['delta_pct']):.1f}pp)"
            )
        parts.append(" | ".join(action_parts))

    return "。".join(parts)


def _group_label(group: str) -> str:
    labels = {
        "mobility": "活动量",
        "sedentary": "静坐/卧",
        "balance": "平衡动作",
        "fall": "跌倒风险动作",
    }
    return labels.get(group, group)
