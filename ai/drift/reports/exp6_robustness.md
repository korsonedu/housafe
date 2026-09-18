# Experiment 6: 传感器鲁棒性

> 生成时间: 2026-08-09 22:29:45
> 实验描述: 点云退化对检测性能的影响

---

## dropout 退化

| 等级 | AUC | 前量(天) |
| --- | --- | --- |
| dropout_5s | 1.0000 | N/A |
| dropout_30s | 0.6667 | N/A |
| dropout_60s | 0.3333 | N/A |

## multipath 退化

| 等级 | AUC | 前量(天) |
| --- | --- | --- |
| multipath_mild | 0.6667 | N/A |
| multipath_moderate | 0.1111 | N/A |
| multipath_severe | 0.6667 | N/A |

## noise 退化

| 等级 | AUC | 前量(天) |
| --- | --- | --- |
| noise_mild | 0.8889 | N/A |
| noise_moderate | 0.6667 | N/A |
| noise_severe | 0.3333 | N/A |

## sparsity 退化

| 等级 | AUC | 前量(天) |
| --- | --- | --- |
| sparsity_mild | 0.7778 | N/A |
| sparsity_moderate | 0.5556 | N/A |
| sparsity_severe | 0.2222 | N/A |
