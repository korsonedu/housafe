# 家安 AI 技术白皮书

> 给投资人 / 技术合作方的计算原理与数学推导

---

## 0. 问题定义

输入：60GHz 毫米波雷达点云流（每帧 10-50 个点，10fps）
输出：
- **事后**：跌倒 / 生命体征异常 / 离线告警（秒级）
- **事前**：行为模式漂移预警（天/周级），提前 2-4 周提示健康退化

核心技术路线一句话：
> PointNet++ 将点云映射到语义隐空间 → GMM 对个人行为常轨做概率密度建模 → 分布漂移量化为健康退化信号

---

## 1. 点云 → 空间特征：PointNet++ Backbone

### 1.1 输入表示

每帧雷达点云：$\mathcal{P} = \{p_i\}_{i=1}^{N}$，每个点 $p_i \in \mathbb{R}^5 = (x, y, z, v_{doppler}, I_{intensity})$

$N \in [10, 50]$，典型值 ~32，为适配 batch 补齐至 64 点。

关键特点：毫米波点云不同于 LiDAR——点数少（几十 vs 几万）、噪声大（多径）、密度不均匀。

### 1.2 层级结构

PointNet++ 在度量空间上递归编码局部几何。3 层 Set Abstraction（SA）：

$$
\begin{aligned}
\text{SA}_1&: (N, 5) \xrightarrow{\text{FPS}} (N/4, 3) \xrightarrow{\text{BallQuery}(r=0.4\text{m}, k=16)} (N/4, 16, 3+2) \xrightarrow{\text{MLP+Max}} (N/4, 128) \\
\text{SA}_2&: (N/4, 128+3) \xrightarrow{\text{FPS}} (N/16, 3) \xrightarrow{\text{BallQuery}(r=0.8\text{m}, k=32)} \xrightarrow{\text{MLP+Max}} (N/16, 256) \\
\text{SA}_3&: (N/16, 256+3) \xrightarrow{\text{Global BallQuery}} \xrightarrow{\text{MaxPool}} (1, 1024)
\end{aligned}
$$

**Farthest Point Sampling (FPS)** 保证采样点均匀覆盖空间，不会在密集区域过采样。从 $N$ 个点中选出 $k$ 个采样点的算法：

1. 随机选初始点 $c_1$
2. For $i = 2, \dots, k$: $c_i = \arg\max_{p \notin \{c_1,\dots,c_{i-1}\}} \min_{j < i} \|p - c_j\|_2$

复杂度 $O(kN)$。$k=N/4$ 时每帧约 512 次距离计算，CPU 上 ~0.1ms。

**Ball Query** 在每个采样点半径 $r$ 内取 $k$ 个最近邻，在局部邻域上执行 PointNet（MLP + max-pooling），实现**置换不变性**（点的输入顺序不影响输出）。

### 1.3 为什么 PointNet++ 对这个场景合适

毫米波雷达没有 RGB 像素网格，点云是无序、非结构化的。CNN 需要体素化，会丢失精度且计算爆炸。PointNet++ 直接在点集上操作：
- 用空间距离（而非像素邻接）定义邻域 → 天然适配稀疏 3D 点
- 层级聚合从局部→全局，类似于 CNN 的感受野递增
- max-pooling 对遮挡/缺失点鲁棒

---

## 2. 时序建模：因果空洞卷积 (TCN)

### 2.1 为什么不是 RNN/LSTM

- 雷达 10fps，1 秒 10 帧。LSTM 串行计算慢，且 32 帧窗口不需要跨 100 帧的长期记忆
- TCN 可并行计算整段时间窗，训练快 5-10×，推理单帧 < 0.1ms

### 2.2 感受野设计

4 层因果卷积，膨胀率 $d \in \{1, 2, 4, 8\}$，kernel size 3：

```
Layer 1 (d=1):  ● ● ●           ← 3 帧
Layer 2 (d=2):  ● . ● . ●       ← 5 帧等效
Layer 3 (d=4):  ● . . . ● . . . ●   ← 9 帧等效
Layer 4 (d=8):  ● . . . . . . . ● . . . . . . . ●  ← 17 帧等效
----------------------------------------------
总感受野:       1 + 2×(1+2+4+8) = 31 帧 ≈ 3 秒
```

32 帧的滑动窗口完整覆盖 3 秒的行为上下文。跌倒从失稳到触地 ~1 秒，3 秒窗口足够捕捉完整的动作过程。

**因果约束**：padding 只在左侧（`F.pad(x, (kernel_size-1)*dilation, 0)`），$t$ 时刻的输出只看 $t$ 及之前的帧，不偷看未来。生产环境中不需要等"未来帧到达"才能推理。

数学上，第 $l$ 层在时间 $t$ 的输出：

$$
h_t^{(l)} = \text{ReLU}\left( \sum_{i=0}^{k-1} W_i^{(l)} h_{t - d_l \cdot i}^{(l-1)} + b^{(l)} \right)
$$

其中 $d_l = 2^{l-1}$，每层带残差连接：$h^{(l)} = h^{(l)} + h^{(l-1)}$（对齐维度时）。

### 2.3 输出

TCN 输出 $S_t \in \mathbb{R}^{256}$ —— 这是整个系统的核心表示：

$$
S_t = \text{TCN}(\phi(\mathcal{P}_{t-31}), \phi(\mathcal{P}_{t-30}), \dots, \phi(\mathcal{P}_t))
$$

其中 $\phi: \mathbb{R}^{N \times 5} \to \mathbb{R}^{1024}$ 是 PointNet++ backbone。

---

## 3. 训练目标：联合多任务学习

### 3.1 总损失

$$
\mathcal{L} = \mathcal{L}_{\text{contrastive}} + \alpha \mathcal{L}_{\text{aux}} + \beta \mathcal{L}_{\text{smooth}} + \gamma \mathcal{L}_{\text{var}}
$$

其中 $\alpha=0.05, \beta=0.1, \gamma=0.01$。

### 3.2 对比损失 (InfoNCE) —— 主信号

将每条 32 帧序列内的帧作为正样本对（同一个人、同一连贯动作），不同序列的帧作为负样本：

$$
\mathcal{L}_{\text{contrastive}} = -\frac{1}{BT}\sum_{i=1}^{BT} \log \frac{\sum_{j \in \mathcal{P}(i)} \exp(z_i \cdot z_j / \tau)}{\sum_{k \neq i} \exp(z_i \cdot z_k / \tau)}
$$

其中 $z_i = \frac{\text{proj}(S_i)}{\|\text{proj}(S_i)\|_2} \in \mathbb{S}^{127}$（128 维投影，L2 归一化到球面），$\tau=0.1$ 是温度参数，$\mathcal{P}(i)$ 是与 $i$ 同序列的所有帧。

**温度 $\tau$ 的作用**：$\tau$ 越小，相似度分布越尖锐，模型被迫更精细地区分正负样本。$\tau=0.1$ 意味着两个帧的 cosine similarity 需要从 0.9 降到 0.8 才能把 softmax 概率从 99% 降到 50% —— 这是一个很强的约束。

**为什么是对比学习**：人工标注"这个人的正常行为"是不可能的（每个人的"正常"不同）。对比学习不需要行为标签——它只要求同一个人相邻时间的行为在隐空间里邻近，不同人的行为分离。这天然适配"个人常轨"的概念。

### 3.3 辅助分类损失

对比学习学到的表示可能在语义上缺乏可解释性。附加一个分类头，用序列均值做动作识别：

$$
S_{\text{mean}} = \frac{1}{T}\sum_{t=1}^{T} S_t, \quad \hat{y} = \text{softmax}(W_c \cdot \text{ReLU}(W_h S_{\text{mean}}))
$$

训练时仅占 5% 权重。结果：12 动作分类准确率 99.98%，说明 **S_t 空间里 walk/sit/fall 等动作是线性可分的**——这是方向 C（可解释漂移分解）能工作的前提。

### 3.4 平滑正则

相邻帧的行为状态不应突变（人不会在 100ms 内从"走"变成"躺"）：

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|S_{t+1} - S_t\|_2^2
$$

### 3.5 方差正则

防止隐状态坍缩到常向量（所有输入映射到同一点）：

$$
\mathcal{L}_{\text{var}} = \left( \frac{1}{256}\sum_{d=1}^{256} \text{Var}[S_{:,d}] - 1 \right)^2
$$

目标方差=1，每个维度被鼓励在训练样本间有单位方差。

---

## 4. 隐空间性质与个人常轨

### 4.1 S_t 空间的语义结构

训练后，S_t 空间具有以下性质（实验验证）：

| 性质 | 证据 |
|------|------|
| 动作线性可分 | 分类头 99.98% acc |
| 时序连续性 | smooth loss 收敛到 0.01 |
| 方差规整 | 各维度方差 ≈1, 无坍缩维度 |
| 语义邻近 | walk 帧和 stand 帧的 cosine sim > walk 和 fall |

这 4 点合起来意味着：**S_t 空间里，行为在语义上是连续的**。一个人的"正常状态"在这个空间里占据一个流形区域，正常行为在这个区域上连续演化，异常行为会跳出去。

### 4.2 为什么 S_t 优于手写特征

手写特征（高度/速度/姿态规则）的问题：

- "高度 < 0.3m" 不能区分"躺床上睡觉"和"跌倒在地上"—— 两类帧的特征向量几乎相同，但它们是人类行为认知里最不同的两类事件
- 老年人缓慢蹲下捡东西 → 高度下降 40cm、持续 3 秒 → 规则引擎可能报"跌倒"，但 S_t 的 smooth trajectory 会自然地停留在 "squat 区域"

Encoder 把"这种运动模式"映射到 S_t，而不是"这个高度值"。这是对比学习的核心能力。

---

## 5. 异常检测：LatentBaseline

### 5.1 PCA 降维

256 维太高 —— 欧氏距离在 256 维空间中失效（任意两点距离趋近，即"维度灾难"）：

$$
\lim_{d \to \infty} \frac{\max_i \|x_i - x_j\| - \min_i \|x_i - x_j\|}{\min_i \|x_i - x_j\|} \to 0
$$

用 PCA 降到 $d' = 12$ 维。选择 12 而不是 32 的理由：GMM 的协方差矩阵 $d' \times d'$ 有 $d'(d'+1)/2$ 个自由参数，12 维 → 78 个参数，3 分量 → 234 个参数，在 1000+ 样本下可靠估计。32 维 → 528 个参数/分量 → 过拟合（实验证实：v1 用 32d，AUC 0.77；v2 用 12d，AUC 0.83）。

$$
\tilde{S}_t = W_{\text{PCA}}^\top (S_t - \mu_{\text{PCA}}) \in \mathbb{R}^{12}
$$

### 5.2 时间上下文

同一个隐状态在不同时间的含义不同：

- $\tilde{S}_t$ 对应的行为是"凌晨 3 点在卧室平躺" → 正常
- $\tilde{S}_t$ 对应的行为是"下午 3 点在厨房平躺" → 异常

定义上下文函数：$\text{ctx}(t) = (\text{weekday\_or\_weekend}, \text{time\_bucket})$

6 个时间段 × 2 种日类型 = 12 个上下文。当前基线使用合并版（weekday 和 weekend 共用 morning/afternoon 等时段名），实际有效 10 个。

每个上下文独立训练一个 GMM：

$$
p(\tilde{S} \mid \text{ctx} = c) = \sum_{k=1}^{K} \pi_k^{(c)} \cdot \mathcal{N}(\tilde{S} \mid \mu_k^{(c)}, \Sigma_k^{(c)})
$$

$K=3$（每个上下文 3 种典型行为模式），协方差类型为 full（允许椭圆高斯，不假设特征独立）。

GMM 参数的 EM 估计：

**E-step** — 计算每个样本对每个分量的责任度：

$$
\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
$$

**M-step** — 重新估计参数：

$$
\mu_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^{N} \gamma_{nk} x_n, \quad
\Sigma_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^{N} \gamma_{nk} (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^\top, \quad
\pi_k^{\text{new}} = \frac{N_k}{N}
$$

其中 $N_k = \sum_n \gamma_{nk}$。

### 5.3 单帧异常打分

对于新帧 $S_t$，做 PCA 变换后，用 Mahalanobis 距离到最近分量的中心：

$$
d_M(\tilde{S}_t, \mu_k, \Sigma_k) = \sqrt{(\tilde{S}_t - \mu_k)^\top \Sigma_k^{-1} (\tilde{S}_t - \mu_k)}
$$

异常分 = 该距离在基线历史分数分布中的 percentile：

$$
\text{score}(S_t, t) = \min_{c} \frac{|\{h \in \mathcal{H}_c : h < \max_k d_M(\tilde{S}_t, \mu_k^{(c)}, \Sigma_k^{(c)})\}|}{|\mathcal{H}_c|}
$$

取 min 跨上下文（找最匹配的分布），值域 $[0, 1]$，0=正常，1=极端异常。

**为什么是 Mahalanobis 而不是欧氏距离**：Mahalanobis 考虑了各维度方差和相关性。在 walk 区域（沿走廊方向方差大，垂直方向方差小），一个偏离走廊 30° 的点在欧氏距离下可能看起来"不远"，但 Mahalanobis 距离会很大 —— 因为 walk 模式几乎从不在那个方向发生。

---

## 6. 事前预警：分布漂移检测

### 6.1 从单帧到单日

单帧异常检测问："这个 $S_t$ 正常吗？"
分布漂移检测问："今天的 $S_t$ 分布和基线周有什么变化？"

对于一天的 $M$ 帧（同上下文 $c$），定义日均 NLL（负对数似然）：

$$
\overline{\text{NLL}}_c^{\text{day}} = -\frac{1}{M}\sum_{t \in \text{day}, \text{ctx}(t)=c} \log p(\tilde{S}_t \mid c)
$$

其中 $p(\tilde{S} \mid c)$ 是基线 GMM。如果今天的帧在基线模型下"平均看起来不太像"——即平均 NLL 偏高——那就意味着分布发生了漂移。

### 6.2 标准化与漂移分

基线拟合时，对每个上下文 $c$ 记录基线数据的 NLL 经验分布 $\{\text{NLL}_n^{(c)}\}_{n=1}^{N_c}$，提取统计量 $\mu_c, \sigma_c$。

当日漂移分 = 当日 mean NLL 偏离基线 mean NLL 的 Z-score，归一化到 $[0, 1]$：

$$
\text{drift}_c^{\text{day}} = \text{clip}\left( \max\left(0, \frac{\overline{\text{NLL}}_c^{\text{day}} - \mu_c}{\sigma_c} \right) / 3, \; 0, 1 \right)
$$

跨上下文的整体漂移分 = 帧数加权平均：

$$
\text{drift}^{\text{day}} = \frac{\sum_c M_c \cdot \text{drift}_c^{\text{day}}}{\sum_c M_c}
$$

其中 $M_c$ 是当日落入上下文 $c$ 的帧数。

**阈值参考**：$\text{drift} < 0.15$ 正常，$0.15$ – $0.4$ 轻度，$>0.4$ 显著。在仿真退化数据上，正常日 vs 漂移日的 AUC = 0.81。

### 6.3 为什么用 NLL 而不用 KL / Wasserstein

真实环境下一个上下文一天可能只有几十帧。在几十个样本上 fit 一个 GMM 是不可靠的。NLL 不需要在当日数据上建模——它把每一帧直接送进基线 GMM 打分，然后取均值。数学上等价于用蒙特卡洛采样的交叉熵：

$$
H(p_{\text{day}}, p_{\text{baseline}}) = -\mathbb{E}_{x \sim p_{\text{day}}}[\log p_{\text{baseline}}(x)] \approx \overline{\text{NLL}}^{\text{day}}
$$

这个近似在 $M \geq 20$ 时稳定。对比 KL 散度需要的 $\mathbb{E}_{p_{\text{day}}}[\log p_{\text{day}}]$ ——当日熵的估计不稳定。

---

## 7. 可解释性：动作分布漂移

### 7.1 动作标注

Encoder 的分类头对序列均值 $S_{\text{mean}}$ 做 12 类动作预测。对滑动窗口取均值获得 per-frame 标签：

$$
\hat{y}_t = \arg\max \text{softmax}\left(W_c \cdot \text{ReLU}\left(W_h \cdot \frac{1}{W}\sum_{i=-W/2}^{W/2} S_{t+i}\right)\right)
$$

### 7.2 分布比较

基线动作分布：$p_{\text{base}}(a) = \frac{1}{N_{\text{base}}}\sum_{t \in \text{baseline}} \mathbf{1}[\hat{y}_t = a]$

当日动作分布：$p_{\text{day}}(a) = \frac{1}{N_{\text{day}}}\sum_{t \in \text{day}} \mathbf{1}[\hat{y}_t = a]$

漂移量 = Jensen-Shannon 散度：

$$
\text{JS}(p \| q) = \frac{1}{2}\text{KL}(p \| m) + \frac{1}{2}\text{KL}(q \| m), \quad m = \frac{p+q}{2}
$$

JS 散度对称、有界 ($[0, \ln 2]$)，适合解释。每个动作的 $\Delta$ 转化为自然语言：

> "walk 占有率从 45% 降到 28% (−17pp)，sit 从 30% 升到 52% (+22pp)"

---

## 8. 融合层：AnomalyDecoder

最终异常分 = 4 个信号的 max 融合（OR 逻辑）：

$$
\text{score}_{\text{final}} = \max(\text{score}_{\text{rules}}, \; \text{score}_{\text{baseline}} \cdot w_{\text{spatial}}, \; \text{score}_{\text{latent}}, \; \text{score}_{\text{predictor}})
$$

其中 $w_{\text{spatial}}$ 来自 SpatialGraph 的房间风险权重（如卫生间风险 > 卧室）。

**融合原则**：规则引擎 100% 敏感（宁可误报不可漏报），GMM 提供特异性（排除规则误报）。二者取 max 保证：任一信号认为异常就告警。事后告警用 max，事前预警用 NLL drift trend（需要连续多天漂移才触发 — 尚未部署到生产）。

---

## 9. 关键指标汇总

| 组件 | 参数量 | 关键指标 |
|------|--------|---------|
| Encoder (PointNet++ + TCN) | 4.7M | 动作分类 99.98% |
| Predictor (GRU) | 1.3M | AUC 0.68 (弱信号) |
| LatentBaseline (PCA + GMM) | ~234 params/GMM | Fall AUC 0.83 |
| DistributionDriftDetector | 0 (复用 GMM) | 基线 vs 漂移 AUC 0.81 |
| 推理延迟 (CPU, 单帧) | — | ~1ms (backbone) + 0.1ms (TCN) |

---

## 10. 与相关工作的关系

| 工作 | 我们的关系 |
|------|-----------|
| PointNet++ (Qi et al. 2017) | 直接采用 backbone |
| SimCLR (Chen et al. 2020) | 对比学习框架参考，适配时序场景 |
| Deep SVDD (Ruff et al. 2018) | 同属 one-class 异常检测范式，我们用 GMM 代替球心距离 |
| 行为节律研究 (Alzheimer's 领域) | 概念参考：昼夜节律紊乱 → 认知衰退早期信号 |
