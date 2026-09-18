"""Markdown 报告生成 — 所有实验输出统一格式的报告。"""

import os
from datetime import datetime
from typing import Any


REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """生成 Markdown 表格。"""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def report_header(title: str, description: str = "") -> str:
    """生成报告头部。"""
    lines = [
        f"# {title}",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 实验描述: {description}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def report_section(title: str, content: str) -> str:
    return f"## {title}\n\n{content}\n"


def write_report(filename: str, content: str):
    ensure_report_dir()
    path = os.path.join(REPORT_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"  报告已保存: {path}")
    return path


# ── 实验专用报告函数 ──────────────────────────────────


def report_exp1_cross_person(results: list[dict]) -> str:
    """Exp 1: 跨人泛化报告。

    results: [{"subject": int, "same_auc": float, "cross_auc": float,
               "cold_start": dict}, ...]
    """
    lines = [report_header("Experiment 1: 跨人泛化",
                           "留一被试交叉验证 + 冷启动分析")]

    # 冷启动曲线（如果存在）
    if results and results[0].get("cold_start"):
        cold_days = sorted(
            [int(k[1:]) for k in results[0]["cold_start"].keys()],
        )
        cs_headers = ["被试"] + [f"d{d}" for d in cold_days]
        cs_rows = []
        for r in results:
            cs = r.get("cold_start", {})
            cs_rows.append([r["subject"]] + [cs.get(f"d{d}", 0) for d in cold_days])
        # 平均行
        avg_row = ["**平均**"]
        for d in cold_days:
            vals = [r.get("cold_start", {}).get(f"d{d}", 0) for r in results]
            avg_row.append(sum(vals) / max(len(vals), 1))
        cs_rows.append(avg_row)

        lines.append(report_section("冷启动曲线 (累积个人数据天数 → AUC)",
                                    _md_table(cs_headers, cs_rows)))

        # 结论
        d0_avg = sum(r.get("cold_start", {}).get("d0", 0) for r in results) / max(len(results), 1)
        d14_avg = sum(r.get("cold_start", {}).get("d14", 0) for r in results) / max(len(results), 1)
        days_to_08 = None
        for d in cold_days:
            vals = [r.get("cold_start", {}).get(f"d{d}", 0) for r in results]
            if sum(vals) / max(len(vals), 1) >= 0.8:
                days_to_08 = d
                break
        lines.append(f"\n**0 天冷启动 (纯跨人) 平均 AUC**: {d0_avg:.4f}\n")
        lines.append(f"**14 天个人数据后平均 AUC**: {d14_avg:.4f}\n")
        if days_to_08 is not None:
            lines.append(f"**AUC 恢复到 0.8+ 所需天数**: ~{days_to_08} 天\n")
        else:
            lines.append(f"**AUC 恢复到 0.8+ 所需天数**: >14 天（或未达到）\n")

    # 主表
    headers = ["被试", "跨人 AUC (d0)", "同人 AUC (d14)", "Δ"]
    rows = []
    for r in results:
        d0 = r.get("cross_auc", 0)
        d14 = r.get("same_auc", 0)
        rows.append([r["subject"], d0, d14, d14 - d0])

    avg_d0 = sum(r.get("cross_auc", 0) for r in results) / max(len(results), 1)
    avg_d14 = sum(r.get("same_auc", 0) for r in results) / max(len(results), 1)
    rows.append(["**平均**", avg_d0, avg_d14, avg_d14 - avg_d0])

    lines.append(report_section("d0 vs d14 对比", _md_table(headers, rows)))

    return "\n".join(lines)


def report_exp2_lead_time(results: list[dict]) -> str:
    """Exp 2: 前置时间分析。

    results: [{"rate": str, "lead_time_days": float, "detection_rate": float,
               "false_alarm_days": int, "seed": int}, ...]
    """
    lines = [report_header("Experiment 2: 前置时间分析",
                           "不同退化速率下的预警前量")]

    # 聚合
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["rate"]].append(r)

    headers = ["退化速率", "平均前量(天)", "标准差(天)", "检出率", "误报天数"]
    rows = []
    for rate in ["fast", "medium", "slow"]:
        if rate not in agg:
            continue
        items = agg[rate]
        lts = [it["lead_time_days"] for it in items if it.get("lead_time_days") is not None]
        drs = [it.get("detection_rate", 0) for it in items]
        fas = [it.get("false_alarm_days", 0) for it in items]
        if lts:
            rows.append([
                rate, sum(lts)/len(lts), _std(lts),
                sum(drs)/len(drs), int(sum(fas)/len(fas)),
            ])
        else:
            rows.append([rate, "N/A", "N/A", "N/A", "N/A"])

    lines.append(report_section("前置时间汇总", _md_table(headers, rows)))

    # 逐种子详情
    lines.append(report_section("逐种子详情",
        _md_table(["速率", "种子", "前量(天)", "检出", "误报"],
                  [[r["rate"], r["seed"], r.get("lead_time_days", "N/A"),
                    r.get("detection_rate", 0), r.get("false_alarm_days", 0)]
                   for r in results])))

    return "\n".join(lines)


def report_exp3_patterns(results: list[dict]) -> str:
    """Exp 3: 退化模式库。

    results: [{"pattern": str, "seed": int, "drift_auc": float, "lead_time": float, ...}]
    """
    lines = [report_header("Experiment 3: 退化模式库",
                           "7 种临床相关退化模式的检测效果")]

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["pattern"]].append(r)

    headers = ["退化模式", "标签", "AUC (mean)", "AUC (std)", "前量(天)", "检出率"]
    rows = []
    from ai.drift.patterns import PATTERN_LABELS
    for name in PATTERN_LABELS:
        if name not in agg:
            continue
        items = agg[name]
        aucs = [it.get("drift_auc", 0) for it in items]
        lts = [it.get("lead_time_days", 0) for it in items if it.get("lead_time_days") is not None]
        drs = [it.get("detection_rate", 0) for it in items]
        rows.append([
            name, PATTERN_LABELS.get(name, name),
            sum(aucs)/len(aucs), _std(aucs),
            sum(lts)/len(lts) if lts else "N/A",
            sum(drs)/len(drs),
        ])

    lines.append(report_section("模式对比", _md_table(headers, rows)))
    return "\n".join(lines)


def report_exp4_false_positive(results: list[dict]) -> str:
    """Exp 4: 假阳性率。

    results: [{"scenario": str, "drift_score_95": float, "drift_score_max": float}, ...]
    """
    lines = [report_header("Experiment 4: 假阳性率分析",
                           "正常生活变化的漂移分分布")]

    headers = ["正常变化场景", "95分位漂移分", "最大漂移分", "建议阈值"]
    rows = []
    all_p95 = []
    for r in results:
        rows.append([r["scenario"], r.get("drift_score_95", 0),
                     r.get("drift_score_max", 0), ""])
        if r.get("drift_score_95", 0) > 0:
            all_p95.append(r["drift_score_95"])

    recommended = max(all_p95) if all_p95 else 0.15
    lines.append(report_section("正常波动漂移分",
        _md_table(headers, rows)))
    lines.append(f"\n**建议告警阈值**: {recommended:.4f} （正常波动 95 分位数的最大值）\n")

    return "\n".join(lines)


def report_exp5_ablation(results: list[dict]) -> str:
    """Exp 5: 消融实验。

    results: [{"name": str, "label": str, "drift_auc": float,
               "lead_time_days": float, ...}]
    """
    lines = [report_header("Experiment 5: 消融实验",
                           "每个设计选择对性能的贡献")]

    baseline_auc = None
    for r in results:
        if r["name"] == "baseline":
            baseline_auc = r.get("drift_auc", 0)
            break

    headers = ["消融项", "标签", "AUC", "Δ AUC", "前量(天)"]
    rows = []
    for r in results:
        delta = ""
        if baseline_auc is not None and r["name"] != "baseline":
            delta = r.get("drift_auc", 0) - baseline_auc
        rows.append([
            r["name"], r.get("label", ""),
            r.get("drift_auc", 0), delta,
            r.get("lead_time_days", "N/A"),
        ])

    lines.append(report_section("消融对比", _md_table(headers, rows)))
    return "\n".join(lines)


def report_exp6_robustness(results: list[dict]) -> str:
    """Exp 6: 传感器鲁棒性。

    results: [{"degradation": str, "level": str, "auc": float, "lead_time_days": float}, ...]
    """
    lines = [report_header("Experiment 6: 传感器鲁棒性",
                           "点云退化对检测性能的影响")]

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["degradation"]].append(r)

    for deg_type, items in sorted(agg.items()):
        headers = ["等级", "AUC", "前量(天)"]
        rows = []
        for it in items:
            rows.append([it["level"], it.get("auc", 0), it.get("lead_time_days", "N/A")])
        lines.append(report_section(f"{deg_type} 退化", _md_table(headers, rows)))

    return "\n".join(lines)


def report_exp2_gait(results: list[dict]) -> str:
    """Exp 2: 动作质量退化。

    results: [{"noise_sigma": float, "mean_nll": float, "delta_nll": float,
               "overall_score": float, "baseline_mean_nll": float}, ...]
    """
    lines = [report_header("Experiment 2: 动作质量退化 (Gait Degradation)",
                           "walk 点云渐增噪声 → S_t 偏移 → NLL 单调上升")]
    headers = ["噪声σ(m)", "mean NLL", "Δ baseline", "drift score", "检出"]
    rows = []
    for r in sorted(results, key=lambda x: x.get("noise_sigma", 0)):
        detected = "✅" if r.get("overall_score", 0) > 0.15 else "—"
        rows.append([
            f"{r['noise_sigma']:.3f}",
            f"{r['mean_nll']:.2f}",
            f"{r['delta_nll']:+.2f}",
            f"{r['overall_score']:.4f}",
            detected,
        ])
    lines.append(report_section("噪声等级 vs NLL", _md_table(headers, rows)))

    bl_mean = results[0].get("baseline_mean_nll", 0) if results else 0
    bl_std = results[0].get("baseline_std_nll", 0) if results else 0
    lines.append(report_section("基线统计",
                   f"mean NLL = {bl_mean:.2f}, std = {bl_std:.2f}"))
    return "\n".join(lines)


def report_exp3_anomaly(results: list[dict]) -> str:
    """Exp 3: 异常动作检测。

    results: [{"anomaly_type": str, "normal_nll_mean": float,
               "anomaly_nll_mean": float, "anomaly_nll_p95": float,
               "normal_nll_p95": float, "auc": float}, ...]
    """
    lines = [report_header("Experiment 3: 异常动作检测 (Anomaly Action)",
                           "插入异常帧 → S_t 进入 GMM 盲区 → NLL 尖峰")]
    headers = ["异常类型", "正常 NLL mean", "异常 NLL mean", "正常 NLL P95", "异常 NLL P95", "AUC"]
    rows = []
    for r in results:
        rows.append([
            r.get("anomaly_type", "?"),
            f"{r.get('normal_nll_mean', 0):.2f}",
            f"{r.get('anomaly_nll_mean', 0):.2f}",
            f"{r.get('normal_nll_p95', 0):.2f}",
            f"{r.get('anomaly_nll_p95', 0):.2f}",
            f"{r.get('auc', 0):.4f}",
        ])
    lines.append(report_section("异常帧 vs 正常帧 NLL", _md_table(headers, rows)))
    return "\n".join(lines)


def report_exp4_context(results: list[dict]) -> str:
    """Exp 4: Context 时段敏感性。

    results: [{"shift": str, "correct_nll": float, "shifted_nll": float,
               "delta_pct": float}, ...]
    """
    lines = [report_header("Experiment 4: Context 时段敏感性",
                           "同 S_t 匹配不同 context GMM → NLL 差异")]
    headers = ["timestamp 偏移", "正确 context NLL", "错位 context NLL", "NLL 升高%"]
    rows = []
    for r in results:
        rows.append([
            r.get("shift", "?"),
            f"{r.get('correct_nll', 0):.2f}",
            f"{r.get('shifted_nll', 0):.2f}",
            f"{r.get('delta_pct', 0):+.1f}%",
        ])
    lines.append(report_section("Context 错位效果", _md_table(headers, rows)))
    return "\n".join(lines)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    import math
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)
