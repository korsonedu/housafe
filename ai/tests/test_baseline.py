"""测试个人基线：GMM 拟合、异常分输出、增量更新"""
import numpy as np
from ai.shared.types import FeatureVector
from ai.baseline.gmm_baseline import PersonalBaseline, context_key


def make_feature_vectors(
    n: int,
    posture: str = "sit",
    heart_rate: float = 72.0,
    resp_rate: float = 16.0,
    centroid: tuple = (1.0, 1.0, 0.5),
    height: float = 0.8,
    weekday: int = 0,
    hour: float = 12.0,
    noise_scale: float = 0.02,
) -> list[FeatureVector]:
    """生成带微小噪声的批量特征向量"""
    from ai.shared.types import time_encode

    rng = np.random.RandomState(42)
    fvs = []
    for i in range(n):
        h = hour  # 固定在同一 context，避免跨 context 稀释样本
        h_sin, h_cos = time_encode(h)
        fvs.append(FeatureVector(
            ts=1000 + i * 2000,
            device_id="r1", room="bedroom",
            posture=posture, posture_confidence=0.9,
            presence=True, moving=False,
            centroid=tuple(c + rng.normal(0, noise_scale) for c in centroid),
            height=height + rng.normal(0, noise_scale),
            n_points=15, occupancy_estimate=1,
            resp_rate=resp_rate + rng.normal(0, noise_scale * 5),
            heart_rate=heart_rate + rng.normal(0, noise_scale * 10),
            vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
            weekday=weekday, velocity_variance=0.01,
        ))
    return fvs


class TestContextKey:
    def test_weekday_morning(self):
        assert context_key(0, 8) == "weekday_morning"

    def test_weekend_night(self):
        assert context_key(5, 2) == "weekend_night"

    def test_weekday_afternoon(self):
        assert context_key(2, 14) == "weekday_afternoon"


class TestPersonalBaseline:
    def test_fit_and_score_normal(self):
        """在 14 天"正常"数据上拟合 → 相似数据应得低异常分"""
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)  # ~1/3 天数据
        baseline.fit(fvs)

        # 和训练数据分布相同的样本 → 异常分应低
        test_fv = make_feature_vectors(1)[0]
        score = baseline.score(test_fv)
        assert score < 0.5, f"Expected low score for normal data, got {score}"

    def test_anomaly_scores_high(self):
        """偏离基线 → 高异常分"""
        baseline = PersonalBaseline()
        normal = make_feature_vectors(500, posture="sit", heart_rate=72)
        baseline.fit(normal)

        # 构造异常：心率极高 + 躺卧（和训练数据完全不同）
        anomaly = make_feature_vectors(1, posture="lie", heart_rate=140)[0]
        score = baseline.score(anomaly)
        assert score > 0.5, f"Expected high score for anomaly, got {score}"

    def test_insufficient_data_returns_zero(self):
        """数据不足时 → 返回 0（基线未建立，不告警）"""
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(10)  # 太少
        baseline.fit(fvs)
        score = baseline.score(fvs[0])
        assert score == 0.0

    def test_update_adds_new_data(self):
        """增量更新后新数据也纳入基线"""
        baseline = PersonalBaseline()
        day1 = make_feature_vectors(200, heart_rate=72, hour=10)
        baseline.fit(day1)

        # 新数据：心率模式不同
        day2 = make_feature_vectors(200, heart_rate=68, hour=10)
        baseline.update(day2)

        # 更新后，心率 70 应该在基线内（介于 68-72）
        test = make_feature_vectors(1, heart_rate=70, hour=10)[0]
        score = baseline.score(test)
        assert score < 0.5, f"After update, mid-range should be normal, got {score}"

    def test_save_and_load(self, tmp_path):
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)
        baseline.fit(fvs)

        path = str(tmp_path / "baseline.pkl")
        baseline.save(path)

        b2 = PersonalBaseline.load(path)
        assert b2.is_ready is True
        # 同样的测试数据应得到相近的分数
        test = make_feature_vectors(1)[0]
        assert abs(baseline.score(test) - b2.score(test)) < 0.01

    def test_is_ready_false_before_fit(self):
        baseline = PersonalBaseline()
        assert baseline.is_ready is False

    def test_is_ready_true_after_sufficient_data(self):
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)
        baseline.fit(fvs)
        assert baseline.is_ready is True
