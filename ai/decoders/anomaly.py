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
        latent_score: float = 0.0,
    ) -> AnomalyResult | None:
        """
        综合评估当前帧。
        predictor_error: 世界模型预测误差 (0-1)，0 表示模型不可用
        latent_score: LatentBaseline 密度异常分 (0-1)，0 表示不可用
        Returns: AnomalyResult（异常）或 None（正常）
        """
        # 1. 规则引擎（确定性安全网）
        rule_results = self.fallback.evaluate(fv, history)

        # 2. FeatureVector 基线偏离（传统方法，基线未就绪时返回 0）
        baseline_score = 0.0
        if self.baseline is not None and self.baseline.is_ready:
            baseline_score = self.baseline.score(fv)

        # 3. 隐状态密度异常分 — 世界模型核心信号
        #    latent_score > 0.8: 行为模式显著偏离个人常轨
        LATENT_THRESHOLD = 0.8

        # 4. 空间感知加权
        spatial_multiplier = 1.0
        if self.graph is not None:
            node_id = self.graph.locate(fv.centroid)
            if node_id is not None:
                attrs = self.graph.get_node_attrs(node_id)
                risk = attrs.get("risk_score", 0.0)
                if fv.posture == "lie" and risk > 0.3:
                    spatial_multiplier = 1.0 + risk

        # 5. 融合：规则 + 传统基线 + 隐状态密度 + 预测误差
        if not rule_results and baseline_score < 0.7 and latent_score < LATENT_THRESHOLD and predictor_error < 0.3:
            return None  # 无异常

        severity_order = {"critical": 3, "warning": 2, "info": 1}
        best_rule = None
        if rule_results:
            best_rule = max(rule_results, key=lambda r: severity_order.get(r.severity, 0))

        rule_score = best_rule.anomaly_score if best_rule else 0.0
        combined_score = max(rule_score, baseline_score * spatial_multiplier, latent_score)

        # Predictor error 辅助
        if predictor_error > 0.25:
            combined_score = max(combined_score, min(predictor_error, 1.0))

        combined_score = min(combined_score, 1.0)

        # 确定来源和类型
        if best_rule:
            source = "both" if (latent_score > LATENT_THRESHOLD or baseline_score > 0.5) else "fallback"
            anomaly_type = best_rule.anomaly_type
            severity = best_rule.severity
            details = best_rule.details
        elif latent_score > LATENT_THRESHOLD:
            source = "world_model"
            anomaly_type = "pattern_deviation"
            severity = "warning" if latent_score > 0.9 else "info"
            details = {}
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
        details["latent_score"] = round(latent_score, 4)
        details["predictor_error"] = round(predictor_error, 4)

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=round(combined_score, 3),
            anomaly_type=anomaly_type,
            severity=severity, source=source,
            details=details,
        )
