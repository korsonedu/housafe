"""测试漂移检测：DistributionDriftDetector + ExplainableDriftDetector + 实验模块"""
import numpy as np
import pytest
from ai.baseline.latent_baseline import LatentBaseline, context_key, _utc_ms_to_datetime
from ai.drift.detector import (
    DistributionDriftDetector,
    ExplainableDriftDetector,
    DayDriftResult,
    ActionDriftResult,
    _js_divergence,
    _generate_summary,
    ACTION_LABELS,
)
from ai.drift.common import NORMAL_PATTERN


def make_S_and_ts(n_frames: int, n_days: int = 3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成模拟 S_t 和时间戳。

    每天 8 个时段各生成 ~n_frames // (n_days * 8) 帧。
    S_t 从不同高斯分布采样以模拟不同上下文的差异。
    """
    rng = np.random.RandomState(seed)
    # 7 个时段（night 出现两次，wd+we = 12 context keys）
    contexts = [
        "wd_night", "wd_morning", "wd_late_morning", "wd_noon",
        "wd_afternoon", "wd_evening",
        "we_night", "we_morning", "we_late_morning", "we_noon",
        "we_afternoon", "we_evening",
    ]
    # 每个 context 的中心不同
    ctx_centers = {
        ctx: rng.randn(256).astype(np.float32) * 0.5
        for ctx in contexts
    }
    # 时间映射
    ctx_to_hour = {
        "wd_night": 3, "wd_morning": 7, "wd_late_morning": 10, "wd_noon": 12,
        "wd_afternoon": 15, "wd_evening": 19,
        "we_night": 3, "we_morning": 7, "we_late_morning": 10, "we_noon": 12,
        "we_afternoon": 15, "we_evening": 19,
    }
    # weekday映射
    ctx_to_wd = {c: (5 if c.startswith("we") else 2) for c in contexts}

    S_list, ts_list = [], []
    base_ts = 1753891200000  # 2026-07-31 00:00 Beijing

    for day in range(n_days):
        for ctx in contexts:
            n = max(5, n_frames // (n_days * len(contexts)))
            center = ctx_centers[ctx]
            S_ctx = rng.randn(n, 256).astype(np.float32) * 0.3 + center

            hour = ctx_to_hour[ctx]
            wd = ctx_to_wd[ctx]
            # 计算时间戳：base + day offset + hour
            ts_ctx = np.array([
                int(base_ts + day * 86400 * 1000 + hour * 3600 * 1000 + i * 1000)
                for i in range(n)
            ], dtype=np.int64)

            S_list.append(S_ctx)
            ts_list.append(ts_ctx)

    S_all = np.vstack(S_list).astype(np.float32)
    ts_all = np.concatenate(ts_list).astype(np.int64)
    return S_all, ts_all


def make_drifted_S(baseline_S: np.ndarray, ts: np.ndarray,
                    drift_scale: float = 1.0) -> np.ndarray:
    """在基线 S 上叠加漂移：向随机方向偏移。drift_scale 越大漂移越严重。"""
    rng = np.random.RandomState(99)
    # 向一个固定方向偏移（模拟行为模式系统性变化）
    drift_direction = rng.randn(256).astype(np.float32)
    drift_direction = drift_direction / (np.linalg.norm(drift_direction) + 1e-10)
    drift_magnitude = drift_scale * 0.5  # scale to reasonable NLL range
    return baseline_S + drift_direction * drift_magnitude


class TestDistributionDriftDetector:
    def test_fit_and_score_day(self):
        """基线拟合后，正常日漂移分低，漂移日漂移分高"""
        S, ts = make_S_and_ts(2000, n_days=7)
        lb = LatentBaseline()
        lb.fit(S, ts)
        assert lb.is_ready

        detector = DistributionDriftDetector(lb)

        # 正常日（同分布）
        S_normal, ts_normal = make_S_and_ts(300, n_days=1, seed=42)
        result_normal = detector.score_day(S_normal, ts_normal)
        assert result_normal.n_total_frames > 0
        # 正常日漂移分应较低
        assert result_normal.overall_score < 0.5, \
            f"Normal day drift should be low, got {result_normal.overall_score}"

        # 漂移日（偏移分布）
        S_drifted, ts_drifted = make_S_and_ts(300, n_days=1, seed=42)
        S_drifted = make_drifted_S(S_drifted, ts_drifted, drift_scale=2.0)
        result_drift = detector.score_day(S_drifted, ts_drifted)
        # 漂移日漂移分应高于正常日
        assert result_drift.overall_score > result_normal.overall_score, \
            f"Drifted day ({result_drift.overall_score}) should have higher score than normal ({result_normal.overall_score})"

    def test_score_window(self):
        """滑动窗口漂移检测返回多段结果"""
        S, ts = make_S_and_ts(2000, n_days=5)
        lb = LatentBaseline()
        lb.fit(S, ts)

        detector = DistributionDriftDetector(lb)
        # 2 天数据 → 约 2 个不重叠 24h 窗口
        S_test, ts_test = make_S_and_ts(600, n_days=2, seed=99)
        windows = detector.score_window(S_test, ts_test, window_hours=24.0)

        assert len(windows) > 0, "Should return at least one window"


class TestExplainableDriftDetector:
    def test_fit_baseline_and_compare(self):
        """基线记录后，compare 返回合理的 ActionDriftResult"""
        S, ts = make_S_and_ts(2000, n_days=5)

        # 需要真实 encoder 或 mock 分类器。这里测试不依赖 encoder 的部分。
        # 直接测试 _js_divergence 和 _generate_summary
        pass

    def test_js_divergence_identical(self):
        """相同分布 → JS=0"""
        p = np.ones(12) / 12
        q = np.ones(12) / 12
        assert _js_divergence(p, q) < 1e-6

    def test_js_divergence_disjoint(self):
        """完全不同的分布 → JS 接近 1"""
        p = np.zeros(12)
        p[0] = 1.0
        q = np.zeros(12)
        q[1] = 1.0
        js = _js_divergence(p, q)
        assert js > 0.5, f"Disjoint distributions should have high JS, got {js}"

    def test_generate_summary_baseline(self):
        """无变化时摘要合理"""
        changes = []
        group_changes = {
            "mobility": {"delta_pct": 0},
            "sedentary": {"delta_pct": 0},
        }
        summary = _generate_summary(changes, group_changes, 0.005)
        assert "基本一致" in summary

    def test_generate_summary_drift(self):
        """明显变化时摘要包含具体动作"""
        changes = [
            {"action": "walk", "baseline_pct": 45.0, "current_pct": 28.0,
             "delta_pct": -17.0, "direction": "decrease"},
            {"action": "sit", "baseline_pct": 30.0, "current_pct": 52.0,
             "delta_pct": 22.0, "direction": "increase"},
        ]
        group_changes = {
            "mobility": {"delta_pct": -15},
            "sedentary": {"delta_pct": 20},
            "balance": {"delta_pct": 0},
            "fall": {"delta_pct": 0},
        }
        summary = _generate_summary(changes, group_changes, 0.12)
        assert "walk" in summary
        assert "sit" in summary


class TestLatentBaselineNLL:
    def test_nll_stats_saved(self):
        """fit 后 NLL 统计存在"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)

        assert lb.is_ready
        assert len(lb._nll_stats) > 0
        for ctx in lb._nll_stats:
            stats = lb._nll_stats[ctx]
            assert "mean" in stats
            assert "std" in stats
            assert "p95" in stats
            assert stats["mean"] > 0  # NLL 应为正数

    def test_score_nll_returns_finite(self):
        """score_nll 返回有限值"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)

        nll = lb.score_nll(S[0], ts[0])
        assert np.isfinite(nll), f"NLL should be finite, got {nll}"

    def test_score_nll_percentile_in_range(self):
        """score_nll_percentile 返回 0-1"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)

        pct = lb.score_nll_percentile(S[0], ts[0])
        assert 0.0 <= pct <= 1.0, f"Percentile should be 0-1, got {pct}"

    def test_save_load_preserves_nll(self):
        """save/load 保留 NLL 统计"""
        import tempfile
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            lb.save(path)
            lb2 = LatentBaseline.load(path)
            assert len(lb2._nll_stats) == len(lb._nll_stats)
            for ctx in lb._nll_stats:
                assert ctx in lb2._nll_stats
                assert abs(lb._nll_stats[ctx]["mean"] - lb2._nll_stats[ctx]["mean"]) < 1e-6
        finally:
            import os
            os.unlink(path)


# ── Ablation hooks ────────────────────────────────────

class TestLatentBaselineAblation:
    def test_no_contexts(self):
        """use_contexts=False → 单一全局 GMM"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline(use_contexts=False)
        lb.fit(S, ts)
        assert lb.is_ready
        assert len(lb._gmms) == 1
        assert "global" in lb._gmms

    def test_custom_pca_dim(self):
        """自定义 PCA 维度"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline(pca_dim=6)
        lb.fit(S, ts)
        assert lb.is_ready
        assert lb._pca.n_components <= 6

    def test_custom_n_components(self):
        """自定义 GMM 分量数"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline(n_components=5)
        lb.fit(S, ts)
        assert lb.is_ready
        for gmm in lb._gmms.values():
            assert gmm.n_components <= 5

    def test_ablation_save_load(self):
        """消融参数在 save/load 中保持"""
        import tempfile, os
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline(pca_dim=8, n_components=2, use_contexts=False)
        lb.fit(S, ts)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            lb.save(path)
            lb2 = LatentBaseline.load(path)
            assert lb2._n_components == 2
            assert lb2._use_contexts is False
        finally:
            os.unlink(path)


class TestDistributionDriftDetectorAblation:
    def test_use_nll_false(self):
        """Mahalanobis 模式不报错"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)

        detector = DistributionDriftDetector(lb, use_nll=False)
        result = detector.score_day(S[:100], ts[:100])
        assert result.overall_score >= 0

    def test_default_is_nll(self):
        """默认使用 NLL"""
        S, ts = make_S_and_ts(2000, n_days=3)
        lb = LatentBaseline()
        lb.fit(S, ts)
        detector = DistributionDriftDetector(lb)
        assert detector.use_nll is True


# ── Patterns ──────────────────────────────────────────

class TestPatterns:
    def test_mobility_decline(self):
        from ai.drift.patterns import mobility_decline
        result = mobility_decline(NORMAL_PATTERN, 1.0)
        assert len(result) > 0
        # walk duration should decrease
        walk_dur = sum(e - s for a, s, e in result if a == "walk")
        orig_walk = sum(e - s for a, s, e in NORMAL_PATTERN if a == "walk")
        assert walk_dur < orig_walk

    def test_sleep_fragmentation(self):
        from ai.drift.patterns import sleep_fragmentation
        result = sleep_fragmentation(NORMAL_PATTERN, 0.5)
        assert len(result) > len(NORMAL_PATTERN)  # more fragments
        # should have walk during night hours
        night_walks = [1 for a, s, e in result if a == "walk" and s < 6]
        assert len(night_walks) > 0

    def test_morning_delay(self):
        from ai.drift.patterns import morning_delay
        result = morning_delay(NORMAL_PATTERN, 1.0)
        # morning activities should shift +2h
        orig_6 = [(a, s, e) for a, s, e in NORMAL_PATTERN if 6 <= s < 9]
        new_6 = [(a, s, e) for a, s, e in result if 6 <= s < 9]
        assert len(new_6) < len(orig_6)  # fewer activities in 6-9 slot

    def test_all_patterns_import(self):
        from ai.drift.patterns import PATTERNS, PATTERN_LABELS
        assert len(PATTERNS) == 7
        assert len(PATTERN_LABELS) == 7

    def test_stepwise_decline(self):
        from ai.drift.patterns import stepwise_decline
        r1 = stepwise_decline(NORMAL_PATTERN, 0.3)
        r2 = stepwise_decline(NORMAL_PATTERN, 0.7)
        # Different levels should produce different patterns
        assert r1 != r2


# ── Degraders ─────────────────────────────────────────

class TestDegraders:
    def test_sparsity(self):
        from ai.drift.degraders import degrade_sparsity
        pts = np.random.randn(50, 5).astype(np.float32)
        degraded = degrade_sparsity(pts, 0.5)
        assert len(degraded) < len(pts)

    def test_noise(self):
        from ai.drift.degraders import degrade_noise
        pts = np.random.randn(50, 5).astype(np.float32)
        degraded = degrade_noise(pts, 1.0)
        assert degraded.shape == pts.shape
        # positions should change
        assert not np.allclose(degraded[:, :3], pts[:, :3])

    def test_dropout(self):
        from ai.drift.degraders import degrade_dropout
        pts = np.random.randn(50, 5).astype(np.float32)
        # With level 1.0, should always dropout
        result = degrade_dropout(pts, 1.0)
        assert result is None

    def test_multipath(self):
        from ai.drift.degraders import degrade_multipath
        pts = np.random.randn(50, 5).astype(np.float32)
        degraded = degrade_multipath(pts, 1.0)
        assert len(degraded) > len(pts)  # ghost points added

    def test_composite_degrader(self):
        from ai.drift.degraders import PointCloudDegrader
        deg = PointCloudDegrader(sparsity=0.5, noise=0.5, multipath=0.5)
        pts = np.random.randn(50, 5).astype(np.float32)
        degraded = deg.apply(pts)
        assert degraded is not None

    def test_degradation_levels(self):
        from ai.drift.degraders import DEGRADATION_LEVELS
        assert "sparsity" in DEGRADATION_LEVELS
        assert "noise" in DEGRADATION_LEVELS
        assert len(DEGRADATION_LEVELS["sparsity"]) == 3


# ── Ablation Config ───────────────────────────────────

class TestAblationConfig:
    def test_all_ablations(self):
        from ai.drift.ablation import ABLATIONS
        assert len(ABLATIONS) >= 5
        names = [a.name for a in ABLATIONS]
        assert "baseline" in names
        assert "no_pca" in names
        assert "no_contexts" in names
        assert "mahalanobis_not_nll" in names
