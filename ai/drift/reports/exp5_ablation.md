# Experiment 5: 消融实验

> 生成时间: 2026-08-09 19:27:25
> 实验描述: 每个设计选择对性能的贡献

---

## 消融对比

| 消融项 | 标签 | AUC | Δ AUC | 前量(天) |
| --- | --- | --- | --- | --- |
| baseline | 基线（当前最优） | 0.7143 |  | None |
| no_pca | 无 PCA | 1.0000 | 0.2857 | None |
| no_contexts | 无时间上下文 | 0.1429 | -0.5714 | None |
| pca_6 | PCA 6d | 0.7143 | 0.0000 | None |
| pca_24 | PCA 24d | 1.0000 | 0.2857 | None |
| n_components_1 | GMM K=1 | 0.7143 | 0.0000 | None |
| n_components_5 | GMM K=5 | 0.7143 | 0.0000 | None |
| mahalanobis_not_nll | Mahalanobis 距离 | 0.7143 | 0.0000 | None |
