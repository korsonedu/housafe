"""安静态判定 — 只有在安静态才适合提取生命体征"""
from ai.shared.types import FeatureVector

QUIET_DURATION_S = 5.0
VELOCITY_VAR_THRESHOLD = 0.05


def is_quiet(fv: FeatureVector, history: list[FeatureVector]) -> bool:
    """
    判断当前是否处于安静态。
    条件: posture in (sit, lie) + not moving + 持续 ≥5s + 速度方差低
    """
    if fv.posture not in ("sit", "lie"):
        return False
    if fv.moving:
        return False
    if fv.velocity_variance > VELOCITY_VAR_THRESHOLD:
        return False

    # 检查持续时间
    device_hist = [h for h in history if h.device_id == fv.device_id]
    device_hist.sort(key=lambda h: h.ts)

    # 从当前帧往回找连续静止的时长
    quiet_start = fv.ts
    for h in reversed(device_hist):
        if h.ts >= fv.ts:
            continue
        if h.moving or h.posture not in ("sit", "lie"):
            quiet_start = h.ts
            break
        quiet_start = h.ts

    duration_s = (fv.ts - quiet_start) / 1000.0
    return duration_s >= QUIET_DURATION_S
