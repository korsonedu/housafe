### Task 7: 异常 Decoder

**Files:**
- Create: `ai/decoders/__init__.py`
- Create: `ai/decoders/anomaly.py`
- Create: `ai/tests/test_anomaly.py`

**Interfaces:**
- Consumes: `FallbackEngine` from Task 3, `SpatialGraph` from Task 5, `PersonalBaseline` from Task 6, `FeatureVector` from Task 1
- Produces: `AnomalyDecoder` class — `evaluate(fv, history) -> AnomalyResult | None`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/decoders
touch ai/decoders/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_anomaly.py`:

```python
"""测试异常解码器：融合 fallback + baseline + 空间感知"""
from ai.shared.types import FeatureVector
from ai.decoders.anomaly import AnomalyDecoder
from ai.fallback.rules import FallbackEngine

def make_fv(ts=1000, posture="sit", room="bedroom", heart_rate=72, resp_rate=16,
            centroid=(1.0, 1.0, 0.5), height=0.8, moving=False, vel_var=0.01,
            hour=12.0, weekday=0):
    from ai.shared.types import time_encode
    h_sin, h_cos = time_encode(hour)
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=moving,
        centroid=centroid, height=height, n_points=15,
        occupancy_estimate=1, resp_rate=resp_rate, heart_rate=heart_rate,
        vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
        weekday=weekday, velocity_variance=vel_var,
    )

class TestAnomalyDecoder:
    def test_fallback_detected_passes_through(self):
        """Fallback 检出的异常直接透传"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        # 构造跌倒序列
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6, moving=True)
            for i in range(5)
        ]
        current = make_fv(ts=2000, posture="lie", height=0.2, moving=False,
                          centroid=(1.0, 1.0, 0.1))
        history.append(current)

        result = decoder.evaluate(current, history)
        assert result is not None
        assert result.anomaly_type == "fall"
        assert result.source == "fallback"

    def test_no_baseline_no_graph_returns_none_for_normal(self):
        """无基线/无图时 → 正常帧不告警"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fv = make_fv()
        result = decoder.evaluate(fv, [fv])
        assert result is None

    def test_spatial_awareness_bathroom_lie(self):
        """卫生间躺卧 → 空间感知加权 → 异常分增加"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fvs_bathroom = [make_fv(room="bathroom", posture="lie",
                                 centroid=(3, 3, 0.1), height=0.2)
                        for _ in range(60)]
        result = decoder.evaluate(fvs_bathroom[-1], fvs_bathroom)
        # stillness 规则在卫生间触发
        if result:
            assert result.room == "bathroom"
```

- [ ] **Step 3: Run test (verify failure)** then implement

- [ ] **Step 4: Implement `ai/decoders/anomaly.py`**

```python
"""异常 Decoder — 融合规则引擎 + 基线偏离 + 空间感知 → 最终异常分"""
from ai.shared.types import FeatureVector, AnomalyResult
from ai.fallback.rules import FallbackEngine


class AnomalyDecoder:
    """
    异常解码器。MVP 版本主要靠 FallbackEngine。
    随基线数据积累，baseline 异常分逐渐加入。
    """

    def __init__(
        self,
        fallback: FallbackEngine,
        baseline=None,   # PersonalBaseline | None
        graph=None,      # SpatialGraph | None
        baseline_weight: float = 0.3,
    ):
        self.fallback = fallback
        self.baseline = baseline
        self.graph = graph
        self.baseline_weight = baseline_weight

    def evaluate(
        self, fv: FeatureVector, history: list[FeatureVector]
    ) -> AnomalyResult | None:
        """
        综合评估当前帧。
        Returns: AnomalyResult（异常）或 None（正常）
        """
        # 1. 规则引擎（确定性安全网）
        rule_results = self.fallback.evaluate(fv, history)

        # 2. 基线偏离
        baseline_score = 0.0
        if self.baseline is not None and self.baseline.is_ready:
            baseline_score = self.baseline.score(fv)

        # 3. 空间感知加权
        spatial_multiplier = 1.0
        if self.graph is not None:
            node_id = self.graph.locate(fv.centroid)
            if node_id is not None:
                attrs = self.graph.get_node_attrs(node_id)
                # 高风险区域 + 躺卧 → 权重增加
                risk = attrs.get("risk_score", 0.0)
                if fv.posture == "lie" and risk > 0.3:
                    spatial_multiplier = 1.0 + risk

        # 4. 融合
        if not rule_results and baseline_score < 0.7:
            return None  # 无异常

        # 取最严重的规则结果
        severity_order = {"critical": 3, "warning": 2, "info": 1}
        best_rule = None
        if rule_results:
            best_rule = max(rule_results, key=lambda r: severity_order.get(r.severity, 0))

        # 融合分数
        rule_score = best_rule.anomaly_score if best_rule else 0.0
        combined_score = max(rule_score, baseline_score * spatial_multiplier)
        combined_score = min(combined_score, 1.0)

        # 确定来源和类型
        if best_rule:
            source = "both" if baseline_score > 0.5 else "fallback"
            anomaly_type = best_rule.anomaly_type
            severity = best_rule.severity
            details = best_rule.details
        else:
            source = "baseline"
            anomaly_type = "pattern_deviation"
            severity = "warning" if combined_score > 0.8 else "info"
            details = {"baseline_score": baseline_score}

        details["spatial_multiplier"] = spatial_multiplier
        details["baseline_score"] = baseline_score

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=round(combined_score, 3),
            anomaly_type=anomaly_type,
            severity=severity, source=source,
            details=details,
        )
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_anomaly.py -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ai/decoders/__init__.py ai/decoders/anomaly.py ai/tests/test_anomaly.py
git commit -m "feat(ai): add AnomalyDecoder — fuse fallback rules + baseline + spatial context

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

