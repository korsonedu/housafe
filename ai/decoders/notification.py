"""通知 Decoder — 异常 → 通知等级（不提/推送/短信/电话）"""
from ai.shared.types import AnomalyResult, NotificationDecision


class NotificationDecider:
    """MVP 版本：规则决策表。演进版本：MLP 4-class 从用户反馈学习。"""

    # 通知决策矩阵: (anomaly_type, severity) → level
    DECISION_MATRIX = {
        ("fall", "critical"): "call",
        ("fall", "warning"): "sms",
        ("fall", "info"): "push",
        ("stillness", "critical"): "sms",
        ("stillness", "warning"): "push",
        ("stillness", "info"): "push",
        ("vital_anomaly", "critical"): "sms",
        ("vital_anomaly", "warning"): "push",
        ("vital_anomaly", "info"): "push",
        ("pattern_deviation", "critical"): "sms",
        ("pattern_deviation", "warning"): "push",
        ("pattern_deviation", "info"): "none",
        ("offline", "critical"): "call",
        ("offline", "warning"): "push",
        ("offline", "info"): "push",
    }

    def __init__(self, suppress_window_s: float = 300.0):
        """
        Args:
            suppress_window_s: 同类型通知抑制窗口（秒）。窗口内重复通知降级为 none。
        """
        self.suppress_window_s = suppress_window_s

    def decide(
        self, anomaly: AnomalyResult, recent_decisions: list[NotificationDecision]
    ) -> NotificationDecision:
        """
        决策通知等级。
        Args:
            anomaly: 当前异常
            recent_decisions: 最近的决策历史（用于抑制重复）
        """
        level = self.DECISION_MATRIX.get(
            (anomaly.anomaly_type, anomaly.severity), "push"
        )

        # 抑制窗口：同类型同房间的重复通知
        if level != "none" and self._is_duplicate(anomaly, recent_decisions):
            level = "none"

        return NotificationDecision(
            level=level,
            reason=f"{anomaly.anomaly_type} in {anomaly.room} (score={anomaly.anomaly_score:.2f})",
            anomaly=anomaly,
        )

    def _is_duplicate(
        self, anomaly: AnomalyResult, recent: list[NotificationDecision]
    ) -> bool:
        for d in recent:
            if d.level == "none":
                continue
            a = d.anomaly
            if (a.anomaly_type == anomaly.anomaly_type
                and a.room == anomaly.room
                and a.device_id == anomaly.device_id):
                dt_s = (anomaly.ts - a.ts) / 1000.0
                if 0 <= dt_s <= self.suppress_window_s:
                    return True
        return False
