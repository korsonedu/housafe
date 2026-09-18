"""报告 Decoder — 从隐状态轨迹生成结构化统计数据"""
from collections import Counter
from ai.shared.types import FeatureVector


class ReportDecoder:
    """MVP 版本：统计聚合。P2 接 LLM 生成自然语言报告。"""

    def generate_daily_summary(self, features: list[FeatureVector]) -> dict:
        """
        生成单日摘要。
        Args:
            features: 约 1440 帧（1/min 采样）的当天特征
        Returns:
            dict with posture_distribution, hours_active, avg_vitals, room_usage
        """
        if not features:
            return {
                "posture_distribution": {},
                "hours_active": 0,
                "avg_heart_rate": None,
                "avg_resp_rate": None,
                "room_usage": {},
                "n_frames": 0,
            }

        # 姿态分布
        posture_counts = Counter(f.posture for f in features)
        total = len(features)
        posture_dist = {k: round(v / total, 3) for k, v in posture_counts.items()}

        # 活跃时长（moving=True 且有走动）
        active_frames = sum(1 for f in features if f.moving or f.posture == "walk")
        hours_active = round(active_frames / max(total, 1) * 24, 1)

        # 平均生命体征
        hrs = [f.heart_rate for f in features if f.heart_rate is not None]
        rrs = [f.resp_rate for f in features if f.resp_rate is not None]
        avg_hr = round(sum(hrs) / len(hrs), 1) if hrs else None
        avg_rr = round(sum(rrs) / len(rrs), 1) if rrs else None

        # 房间使用
        room_counts = Counter(f.room for f in features)
        room_usage = {k: round(v / total, 3) for k, v in room_counts.items()}

        return {
            "posture_distribution": posture_dist,
            "hours_active": hours_active,
            "avg_heart_rate": avg_hr,
            "avg_resp_rate": avg_rr,
            "room_usage": room_usage,
            "n_frames": total,
        }

    def weekly_comparison(self, this_week: dict, last_week: dict) -> dict:
        """本周 vs 上周趋势对比"""
        deltas = {}
        for key in ("hours_active", "avg_heart_rate", "avg_resp_rate"):
            if this_week.get(key) is not None and last_week.get(key) is not None:
                deltas[key] = round(this_week[key] - last_week[key], 2)
        return deltas
