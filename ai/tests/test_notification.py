"""测试通知决策规则"""
from ai.shared.types import AnomalyResult, NotificationDecision
from ai.decoders.notification import NotificationDecider


class TestNotificationDecider:
    def test_fall_critical_calls(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.95, anomaly_type="fall",
            severity="critical", source="fallback",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level == "call"

    def test_fall_warning_sms(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.75, anomaly_type="fall",
            severity="warning", source="fallback",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level == "sms"

    def test_stillness_critical_sms(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.9, anomaly_type="stillness",
            severity="critical", source="fallback",
            details={"duration_s": 600},
        )
        decision = d.decide(ar, [])
        assert decision.level == "sms"

    def test_vital_warning_push(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.6, anomaly_type="vital_anomaly",
            severity="warning", source="fallback",
            details={"heart_rate": 135},
        )
        decision = d.decide(ar, [])
        assert decision.level == "push"

    def test_pattern_deviation_info_push(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.5, anomaly_type="pattern_deviation",
            severity="info", source="baseline",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level in ("push", "none")

    def test_suppress_duplicates(self):
        """短时间内相同异常类型 → 抑制重复通知"""
        d = NotificationDecider(suppress_window_s=300)
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.7, anomaly_type="stillness",
            severity="warning", source="fallback",
            details={},
        )
        # 第一条正常通知
        d1 = d.decide(ar, [])
        # 模拟 history: 刚发过同类通知
        d2 = d.decide(ar, [d1])
        assert d2.level == "none"
