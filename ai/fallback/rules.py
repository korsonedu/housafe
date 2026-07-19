"""Fallback 规则引擎 — 世界模型的安全网。每条规则独立、可配置、确定性。"""

from collections import deque
from ai.shared.types import FeatureVector, AnomalyResult

# 候选阈值（设计文档标注 "候选·待实验确定"）
FALL_HEIGHT_DROP_M = 0.8       # 高度骤降阈值
FALL_WINDOW_S = 1.0            # 骤降发生的时间窗口
FALL_CONFIRM_S = 5.0           # 跌落后持续观察时间
FALL_STILL_S = 10.0            # 跌落后未恢复站立

BATHROOM_STILLNESS_WARN_S = 60   # 卫生间静止 warning 阈值（候选·待实验确定）
BATHROOM_STILLNESS_CRITICAL_S = 120
GENERAL_STILLNESS_INFO_S = 1800   # 其他区域静止 info 阈值

HR_CRITICAL_LOW = 30
HR_WARN_LOW = 40
HR_WARN_HIGH = 130
HR_CRITICAL_HIGH = 150

RR_CRITICAL_LOW = 4
RR_WARN_LOW = 6
RR_WARN_HIGH = 30
RR_CRITICAL_HIGH = 35

OFFLINE_WARN_S = 60
OFFLINE_CRITICAL_S = 300


class FallbackEngine:
    """确定性规则引擎。每个规则可独立开关。"""

    def __init__(
        self,
        enable_fall: bool = True,
        enable_stillness: bool = True,
        enable_vital: bool = True,
        enable_offline: bool = True,
    ):
        self.enable_fall = enable_fall
        self.enable_stillness = enable_stillness
        self.enable_vital = enable_vital
        self.enable_offline = enable_offline

    def evaluate(self, fv: FeatureVector, history: list[FeatureVector]) -> list[AnomalyResult]:
        """评估当前帧 + 历史 → 返回触发的异常列表"""
        results = []

        if self.enable_fall:
            fall = self._detect_fall(fv, history)
            if fall:
                results.append(fall)

        if self.enable_stillness:
            still = self._detect_stillness(fv, history)
            if still:
                results.append(still)

        if self.enable_vital:
            vital = self._check_vital_thresholds(fv)
            if vital:
                results.append(vital)

        # offline 规则需要心跳数据，由外层 main.py 触发，不在此处基于 FeatureVector 判断
        return results

    # ── 跌倒检测 ──────────────────────────────────────

    def _detect_fall(self, fv: FeatureVector, history: list[FeatureVector]) -> AnomalyResult | None:
        if len(history) < 5:
            return None

        # 找 1s 窗口内的最高高度
        window_s = FALL_WINDOW_S
        recent = [h for h in history if 0 <= (fv.ts - h.ts) <= window_s * 1000]
        if not recent:
            recent = history[-5:]

        max_height = max(h.height for h in recent)
        height_drop = max_height - fv.height

        if height_drop < FALL_HEIGHT_DROP_M:
            return None

        # 确认：当前姿势是躺卧
        if fv.posture != "lie":
            return None

        # 确认：跌落后持续躺卧
        after_fall = [h for h in history if h.ts >= fv.ts and (h.ts - fv.ts) <= FALL_CONFIRM_S * 1000]
        if not after_fall:
            after_fall = [fv]

        still_lying = all(h.posture == "lie" for h in after_fall)
        if not still_lying:
            return None

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=min(0.8 + height_drop * 0.1, 1.0),
            anomaly_type="fall", severity="critical", source="fallback",
            details={"height_drop_m": round(height_drop, 2), "max_height": max_height},
        )

    # ── 静止超时 ──────────────────────────────────────

    def _detect_stillness(self, fv: FeatureVector, history: list[FeatureVector]) -> AnomalyResult | None:
        if not fv.presence or fv.moving:
            return None

        # 计算连续静止时长
        still_duration_s = self._continuous_stillness(fv, history)

        if fv.room in ("bathroom", "卫生间"):
            if fv.posture == "lie" and still_duration_s >= BATHROOM_STILLNESS_CRITICAL_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.9, anomaly_type="stillness",
                    severity="critical", source="fallback",
                    details={"duration_s": still_duration_s},
                )
            elif fv.posture == "lie" and still_duration_s >= BATHROOM_STILLNESS_WARN_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.7, anomaly_type="stillness",
                    severity="warning", source="fallback",
                    details={"duration_s": still_duration_s},
                )
        else:
            # 非卫生间/卧室：长时间静止
            if fv.room != "bedroom" and fv.room != "卧室" and still_duration_s >= GENERAL_STILLNESS_INFO_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.5, anomaly_type="stillness",
                    severity="info", source="fallback",
                    details={"duration_s": still_duration_s},
                )

        return None

    def _continuous_stillness(self, fv: FeatureVector, history: list[FeatureVector]) -> float:
        """计算从最近一次 moving=True 到当前的连续静止秒数"""
        sorted_h = sorted(history, key=lambda h: h.ts, reverse=True)
        for h in sorted_h:
            if h.device_id != fv.device_id:
                continue
            if h.moving:
                return (fv.ts - h.ts) / 1000.0
        return (fv.ts - sorted_h[-1].ts) / 1000.0 if sorted_h else 0.0

    # ── 生命体征阈值 ──────────────────────────────────

    def _check_vital_thresholds(self, fv: FeatureVector) -> AnomalyResult | None:
        if fv.heart_rate is None or fv.resp_rate is None:
            return None

        hr = fv.heart_rate
        rr = fv.resp_rate

        if hr <= HR_CRITICAL_LOW or hr >= HR_CRITICAL_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.9, anomaly_type="vital_anomaly",
                severity="critical", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if rr <= RR_CRITICAL_LOW or rr >= RR_CRITICAL_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.9, anomaly_type="vital_anomaly",
                severity="critical", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if hr <= HR_WARN_LOW or hr >= HR_WARN_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.6, anomaly_type="vital_anomaly",
                severity="warning", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if rr <= RR_WARN_LOW or rr >= RR_WARN_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.6, anomaly_type="vital_anomaly",
                severity="warning", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        return None
