"""退化模式库 — Experiment 3 的 7 种行为退化模式。

每种模式是一个函数，接受 base pattern 和配置，返回 modified pattern。
复用 `apply_drift` 的机制，但不限于线性衰减。
"""

import copy
import numpy as np
from ai.drift.common import NORMAL_PATTERN, DRIFT_MODIFIERS, apply_drift


def mobility_decline(pattern: list, level: float = 1.0) -> list:
    """活动量减少：walk/squat/stand ↓, sit ↑（已有默认模式）"""
    return apply_drift(pattern, level, DRIFT_MODIFIERS)


def sleep_fragmentation(pattern: list, level: float = 1.0,
                         insert_per_hour: float = 3.0) -> list:
    """睡眠破碎：凌晨 lie 时段随机插入 walk/stand 片段。

    level=1.0 时每小时插入 ~3 次短暂觉醒。
    """
    rng = np.random.RandomState(42)
    modified = []
    for action, start_h, end_h in pattern:
        if action == "lie" and level > 0:
            dur = end_h - start_h
            n_insertions = int(dur * insert_per_hour * level)
            # 在 lie 时段均匀插入 walk 片段
            insert_points = sorted(rng.uniform(start_h + 0.1, end_h - 0.1, n_insertions))
            current = start_h
            for ip in insert_points:
                if ip - current > 0.05:
                    modified.append(("lie", current, ip))
                modified.append(("walk", ip, ip + 0.03))  # 1.8min walk
                current = ip + 0.03
            if end_h - current > 0.05:
                modified.append(("lie", current, end_h))
        else:
            modified.append((action, start_h, end_h))
    return modified


def morning_delay(pattern: list, level: float = 1.0) -> list:
    """晨间启动延迟：morning 时段活动后移。level=1.0 → 后移 2h。"""
    shift_hours = level * 2.0
    modified = []
    for action, start_h, end_h in pattern:
        h = start_h
        # 只延迟 6-12 点的活动
        if 6 <= h < 12:
            modified.append((action, start_h + shift_hours, end_h + shift_hours))
        else:
            modified.append((action, start_h, end_h))
    return modified


def gait_instability(pattern: list, level: float = 1.0) -> list:
    """步态不稳：不改模式，在 common.generate_day 里通过
    point_cloud_degrader 对 walk 帧加高频噪声实现。
    这里只返回原 pattern，标记需要 degrader。
    """
    # gait_instability 不修改活动模式，通过破化器在点云层注入
    return pattern


def activity_fragmentation(pattern: list, level: float = 1.0) -> list:
    """活动碎片化：长 sit 段拆成 sit-walk-sit 短循环。

    level=1.0 时，>1h 的 sit 段被拆分为 20min sit + 2min walk 循环。
    """
    fragment_dur = max(0.05, 0.33 - level * 0.2)  # 20min at level=1
    walk_insert_dur = 0.03 + level * 0.03  # 2-4min walk

    modified = []
    for action, start_h, end_h in pattern:
        dur = end_h - start_h
        if action == "sit" and dur > 0.5 and level > 0.1:
            n_fragments = max(1, int(dur / fragment_dur))
            actual_frag = dur / n_fragments
            current = start_h
            for i in range(n_fragments):
                sit_end = current + actual_frag * 0.85
                walk_end = sit_end + walk_insert_dur
                if walk_end > end_h:
                    walk_end = end_h
                modified.append(("sit", current, sit_end))
                if walk_end > sit_end:
                    modified.append(("walk", sit_end, walk_end))
                current = walk_end
            if end_h - current > 0.02:
                modified.append(("sit", current, end_h))
        else:
            modified.append((action, start_h, end_h))
    return modified


def bathroom_prolong(pattern: list, level: float = 1.0) -> list:
    """卫生间停留延长：特定时段（上午 7-9、下午 13-15）sit 延长。

    level=1.0 时延长 3×。
    """
    target_hours = {(7, 9), (13, 15)}
    multiplier = 1.0 + level * 2.0  # 1× → 3×

    modified = []
    for action, start_h, end_h in pattern:
        dur = end_h - start_h
        in_target = any(t[0] <= start_h < t[1] or t[0] < end_h <= t[1]
                       for t in target_hours)
        if action == "sit" and in_target:
            new_dur = dur * multiplier
            modified.append((action, start_h, start_h + new_dur))
        else:
            modified.append((action, start_h, end_h))
    return modified


def stepwise_decline(pattern: list, level: float = 1.0,
                      step_points: list[float] | None = None) -> list:
    """阶梯式退化：突然退化 → 平台 → 再退化。

    不退化为线性。在 step_points（默认 [0.3, 0.7]）处跳跃。
    """
    if step_points is None:
        step_points = [0.3, 0.7]

    # 阶梯函数：level 映射到阶梯值
    stepped = 0.0
    for sp in step_points:
        if level >= sp:
            stepped = sp
    if level >= step_points[-1]:
        stepped = level  # 超出最后阶梯后线性

    return apply_drift(pattern, stepped * 1.5, DRIFT_MODIFIERS)


# ── 模式注册表 ────────────────────────────────────────

PATTERNS = {
    "mobility_decline": mobility_decline,
    "sleep_fragmentation": sleep_fragmentation,
    "morning_delay": morning_delay,
    "gait_instability": gait_instability,
    "activity_fragmentation": activity_fragmentation,
    "bathroom_prolong": bathroom_prolong,
    "stepwise_decline": stepwise_decline,
}

PATTERN_LABELS = {
    "mobility_decline": "活动量减少",
    "sleep_fragmentation": "睡眠破碎",
    "morning_delay": "晨间启动延迟",
    "gait_instability": "步态不稳",
    "activity_fragmentation": "活动碎片化",
    "bathroom_prolong": "卫生间停留延长",
    "stepwise_decline": "阶梯式退化",
}
