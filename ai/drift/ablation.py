"""消融实验配置 — Experiment 5。

每个消融项是 DistributionDriftDetector 的一组参数覆盖。
"""

from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """消融实验的一组配置。None = 使用默认值。"""
    name: str = ""
    label: str = ""  # 中文标签
    pca_dim: int | None = None        # None = 默认 12
    n_components: int | None = None   # None = 默认 3
    use_nll: bool = True
    use_contexts: bool = True
    description: str = ""


ABLATIONS = [
    AblationConfig(
        name="baseline",
        label="基线（当前最优）",
        description="PCA 12d + GMM 3 + NLL + 6 contexts",
    ),
    AblationConfig(
        name="no_pca",
        label="无 PCA",
        pca_dim=256,
        description="256d 原始 S_t 直接 GMM",
    ),
    AblationConfig(
        name="no_contexts",
        label="无时间上下文",
        use_contexts=False,
        description="单一全局 GMM，不分时段",
    ),
    AblationConfig(
        name="pca_6",
        label="PCA 6d",
        pca_dim=6,
        description="PCA 降到 6 维",
    ),
    AblationConfig(
        name="pca_24",
        label="PCA 24d",
        pca_dim=24,
        description="PCA 降到 24 维",
    ),
    AblationConfig(
        name="n_components_1",
        label="GMM K=1",
        n_components=1,
        description="单高斯分布",
    ),
    AblationConfig(
        name="n_components_5",
        label="GMM K=5",
        n_components=5,
        description="5 个高斯分量",
    ),
    AblationConfig(
        name="mahalanobis_not_nll",
        label="Mahalanobis 距离",
        use_nll=False,
        description="用 mean Mahalanobis 代替 mean NLL 做漂移分",
    ),
]
