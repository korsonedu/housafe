### Task 8: 通知 Decoder

**Files:**
- Create: `ai/decoders/notification.py`
- Create: `ai/tests/test_notification.py`

**Interfaces:**
- Consumes: `AnomalyResult` from Task 1, `NotificationDecision` from Task 1
- Produces: `NotificationDecider` class — `decide(anomaly, history) -> NotificationDecision`

- [ ] **Step 1: Write tests**

`ai/tests/test_notification.py`:

```python
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
```

- [ ] **Step 2: Run test (verify failure)** then implement

- [ ] **Step 3: Implement `ai/decoders/notification.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_notification.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ai/decoders/notification.py ai/tests/test_notification.py
git commit -m "feat(ai): add NotificationDecider — anomaly → notification level mapping

Rule matrix + duplicate suppression window. MLP upgrade on user feedback (P2).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

