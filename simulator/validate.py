"""仿真有效性自检"""
import numpy as np
from simulator.types import FrameGroup

EXPECTED_Z = {
    "stand": (0.3, 1.8),
    "sit": (0.0, 1.2),
    "lie": (-0.1, 0.5),
    "walk": (0.3, 1.8),
    "fall": (-0.1, 0.5),
    "squat": (0.0, 1.3),
}


def validate_simulation(frames: list[FrameGroup]) -> dict:
    """对生成数据跑统计检查，返回检查结果字典"""
    checks = {}

    # 1. 点数范围：3-100（无校准时 fallback 生成器点少）
    n_points = [len(f.points) for f in frames
                if f.points is not None and len(f.points) > 0]
    if n_points:
        checks["point_count_range"] = {
            "passed": min(n_points) >= 3 and max(n_points) <= 100,
            "min": min(n_points), "max": max(n_points),
        }
    else:
        checks["point_count_range"] = {"passed": False, "detail": "no valid points"}

    # 2. z 坐标范围：-0.5m ~ 2.5m
    all_z = []
    for f in frames:
        if f.points is not None and len(f.points) > 0 and f.points.shape[1] >= 3:
            all_z.extend(f.points[:, 2].tolist())
    if all_z:
        z_arr = np.array(all_z)
        checks["z_range"] = {
            "passed": z_arr.min() >= -0.5 and z_arr.max() <= 2.5,
            "min": float(z_arr.min()), "max": float(z_arr.max()),
        }
    else:
        checks["z_range"] = {"passed": False, "detail": "no z data"}

    # 3. 时序连续性：相邻帧间隔 0-300ms
    if len(frames) >= 2:
        ts_diffs = np.diff([f.ts for f in frames])
        checks["temporal_continuity"] = {
            "passed": ts_diffs.min() > 0 and ts_diffs.max() < 300,
            "min_ms": int(ts_diffs.min()), "max_ms": int(ts_diffs.max()),
            "mean_ms": float(ts_diffs.mean()),
        }

    # 4. 姿态-z 一致性（允许 30% 不一致，过渡段正常）
    mismatches = []
    for f in frames:
        if (f.gt and f.gt.posture in EXPECTED_Z
                and f.points is not None and len(f.points) > 0):
            z_mean = float(f.points[:, 2].mean())
            z_lo, z_hi = EXPECTED_Z[f.gt.posture]
            if not (z_lo <= z_mean <= z_hi):
                mismatches.append({
                    "frame_id": f.frame_id,
                    "posture": f.gt.posture,
                    "z_mean": round(z_mean, 2),
                })
    checks["posture_z_consistency"] = {
        "passed": len(mismatches) <= len(frames) * 0.3,
        "mismatches": len(mismatches),
        "total": len(frames),
    }

    # 汇总
    checks["all_passed"] = all(
        v["passed"] if isinstance(v, dict) and "passed" in v else True
        for v in checks.values()
    )
    return checks


def print_validation_report(checks: dict):
    """打印可读的验证报告"""
    print("=" * 50)
    print("  仿真有效性自检报告")
    print("=" * 50)
    for name, result in checks.items():
        if name == "all_passed":
            continue
        passed = result.get("passed", False) if isinstance(result, dict) else result
        status = "✓" if passed else "✗"
        detail = ""
        if isinstance(result, dict):
            parts = []
            for k, v in result.items():
                if k != "passed":
                    parts.append(f"{k}={v}")
            detail = " | ".join(parts)
        print(f"  {status} {name}: {detail}")
    print(f"\n  整体: {'✓ 通过' if checks.get('all_passed') else '✗ 失败'}")
