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
        self, fv: FeatureVector, history: list[FeatureVector],
        predictor_error: float = 0.0,
    ) -> AnomalyResult | None:
        """
        综合评估当前帧。
        predictor_error: 世界模型预测误差 (0-1)，0 表示模型不可用
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
        if not rule_results and baseline_score < 0.7 and predictor_error < 0.3:
            return None  # 无异常

        # 取最严重的规则结果
        severity_order = {"critical": 3, "warning": 2, "info": 1}
        best_rule = None
        if rule_results:
            best_rule = max(rule_results, key=lambda r: severity_order.get(r.severity, 0))

        # 融合分数（规则 + 基线 + 预测误差）
        rule_score = best_rule.anomaly_score if best_rule else 0.0
        combined_score = max(rule_score, baseline_score * spatial_multiplier)

        # 世界模型预测误差：训练正常误差均值 0.081，3x ≈ 0.25 为阈值
        if predictor_error > 0.25:
            combined_score = max(combined_score, min(predictor_error, 1.0))

        combined_score = min(combined_score, 1.0)

        # 确定来源和类型
        if best_rule:
            source = "both" if baseline_score > 0.5 else "fallback"
            anomaly_type = best_rule.anomaly_type
            severity = best_rule.severity
            details = best_rule.details
        elif predictor_error > 0.25:
            source = "world_model"
            anomaly_type = "pattern_deviation"
            severity = "warning" if predictor_error > 0.5 else "info"
            details = {}
        else:
            source = "baseline"
            anomaly_type = "pattern_deviation"
            severity = "warning" if combined_score > 0.8 else "info"
            details = {"baseline_score": baseline_score}

        details["spatial_multiplier"] = spatial_multiplier
        details["baseline_score"] = baseline_score
        details["predictor_error"] = round(predictor_error, 4)

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=round(combined_score, 3),
            anomaly_type=anomaly_type,
            severity=severity, source=source,
            details=details,
        )
