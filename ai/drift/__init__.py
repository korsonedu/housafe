"""事前健康管理 — 跨时间尺度行为模式漂移检测。

Direction A: 时段分布比较 (DistributionDriftDetector)
  - 在 LatentBaseline 的 per-context GMM 之上，比较当日 S_t 分布 vs 基线分布
  - 用 NLL ratio 量化分布漂移

Direction C: 可解释漂移分解 (ExplainableDriftDetector)
  - 用 Encoder 分类头给每帧打动作标签
  - 对比基线与当日的动作分布变化
  - 生成人类可读的漂移解释
"""

from ai.drift.detector import (
    DistributionDriftDetector,
    ExplainableDriftDetector,
    DayDriftResult,
    ActionDriftResult,
)
