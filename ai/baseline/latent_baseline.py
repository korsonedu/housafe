"""LatentBaseline — 在 Encoder 256-d 隐状态空间上做分上下文 GMM 密度估计。

这是世界模型的核心异常检测器：
  1. Encoder 将点云映射到 256-d 动作语义空间
  2. 按 (weekday, hour) 分 10 个上下文，每个上下文独立 GMM
  3. 异常 = 当前 S_t 在其上下文 GMM 中的 Mahalanobis 距离 percentile

与 PersonalBaseline (FeatureVector) 的区别：
  - 不依赖手写特征（posture/height/centroid），直接用 Encoder 学到的表示
  - 256-d → PCA → 32-d → GMM，保留语义信息同时保证数值稳定
  - 这是规则引擎做不到的：检测"这个人的行为模式偏离了个人常轨"
"""

import math
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

MIN_SAMPLES_PER_CONTEXT = 50
MIN_TOTAL_SAMPLES = 200
N_COMPONENTS = 3
PCA_DIM = 12


def context_key(weekday: int, hour: float) -> str:
    """(weekday, hour) → 上下文 bucket。hour 为 0-24 浮点数。"""
    is_weekend = weekday >= 5
    prefix = "we" if is_weekend else "wd"
    h = int(hour)
    if h < 6:
        return f"{prefix}_night"
    elif h < 9:
        return f"{prefix}_morning"
    elif h < 12:
        return f"{prefix}_late_morning"
    elif h < 14:
        return f"{prefix}_noon"
    elif h < 18:
        return f"{prefix}_afternoon"
    elif h < 21:
        return f"{prefix}_evening"
    else:
        return f"{prefix}_night"


class LatentBaseline:
    """分上下文 GMM 在 Encoder latent space 上做个人常轨建模。

    Usage:
      # 拟合
      lb = LatentBaseline()
      lb.fit(S_sequences, timestamps)  # S: (N, 256), ts: (N,) ms

      # 检测
      score = lb.score(S_t, ts)  # → 0-1, 0=正常 1=极端异常

      # 持久化
      lb.save("baseline.pkl")
      lb = LatentBaseline.load("baseline.pkl")
    """

    def __init__(self, pca_dim: int = PCA_DIM, n_components: int = N_COMPONENTS,
                 use_contexts: bool = True):
        self._gmms: dict[str, GaussianMixture] = {}
        self._pca: PCA | None = None
        self._scores: dict[str, list[float]] = {}  # 历史 Mahalanobis 分数用于标准化
        self._nll_stats: dict[str, dict] = {}  # per-context NLL 分布: {mean, std, p50, p75, p95}
        self._pca_dim: int = pca_dim
        self._n_components: int = n_components
        self._use_contexts: bool = use_contexts
        self.is_ready: bool = False

    def fit(self, S: np.ndarray, timestamps_ms: np.ndarray):
        """
        拟合个人基线。

        Args:
            S: (N, 256) float32, Encoder 输出的 latent states
            timestamps_ms: (N,) int64, 每帧的 UTC 毫秒时间戳
        """
        N = len(S)
        if N < MIN_TOTAL_SAMPLES:
            return

        # 时间上下文
        contexts = []
        for ts in timestamps_ms:
            dt = _utc_ms_to_datetime(ts)
            contexts.append(context_key(dt["weekday"], dt["hour"]))
        contexts = np.array(contexts)

        # 分上下文分组
        groups: dict[str, np.ndarray] = {}
        if self._use_contexts:
            for ctx in np.unique(contexts):
                mask = contexts == ctx
                if mask.sum() >= MIN_SAMPLES_PER_CONTEXT:
                    groups[ctx] = S[mask]
        else:
            # 消融模式：单一全局上下文
            groups["global"] = S

        if not groups:
            return

        # PCA on all data
        all_data = np.vstack(list(groups.values()))
        pca_dim = min(self._pca_dim, all_data.shape[1], all_data.shape[0] // 20)
        self._pca = PCA(n_components=pca_dim, random_state=42)
        all_reduced = self._pca.fit_transform(all_data)

        # 分上下文 GMM
        self._gmms = {}
        self._scores = {}
        self._nll_stats = {}
        offset = 0
        for ctx, data in groups.items():
            n = len(data)
            reduced = all_reduced[offset:offset + n]
            offset += n

            n_comp = min(self._n_components, max(1, n // 30))
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                random_state=42,
                reg_covar=1e-4,
            )
            gmm.fit(reduced)
            self._gmms[ctx] = gmm

            # 记录历史 Mahalanobis 分数
            scores = []
            for i in range(n):
                comp = gmm.predict(reduced[i:i + 1])[0]
                d = _mahalanobis(reduced[i], gmm.means_[comp], gmm.covariances_[comp])
                scores.append(d)
            self._scores[ctx] = scores

            # 记录 NLL 分布（用于漂移检测）
            log_probs = gmm.score_samples(reduced)  # (n,) log-likelihood
            nlls = -log_probs  # negative log-likelihood
            self._nll_stats[ctx] = {
                "mean": float(np.mean(nlls)),
                "std": float(np.std(nlls)),
                "p50": float(np.percentile(nlls, 50)),
                "p75": float(np.percentile(nlls, 75)),
                "p95": float(np.percentile(nlls, 95)),
                "p99": float(np.percentile(nlls, 99)),
            }

        self.is_ready = True

    def score(self, S_t: np.ndarray, ts_ms: int) -> float:
        """
        返回异常分 0-1。0=正常，1=极端异常。
        在所有上下文中取最低分（最匹配的上下文）。
        基线未就绪时返回 0。
        """
        if not self.is_ready or self._pca is None:
            return 0.0

        reduced = self._pca.transform(S_t.reshape(1, -1))[0]

        # 在所有上下文中找最低异常分（即最匹配的分布）
        best_score = 1.0
        for ctx, gmm in self._gmms.items():
            comp = gmm.predict(reduced.reshape(1, -1))[0]
            dist = _mahalanobis(reduced, gmm.means_[comp], gmm.covariances_[comp])
            hist = self._scores.get(ctx, [])
            if hist:
                pct = sum(1 for h in hist if h < dist) / len(hist)
            else:
                pct = min(dist / 10.0, 1.0)
            best_score = min(best_score, float(np.clip(pct, 0.0, 1.0)))

        return best_score

    def score_nll(self, S_t: np.ndarray, ts_ms: int) -> float:
        """
        返回当前帧在其匹配上下文 GMM 下的 NLL（负对数似然）。
        值越低=越正常。基线未就绪时返回 inf。
        """
        if not self.is_ready or self._pca is None:
            return float("inf")

        reduced = self._pca.transform(S_t.reshape(1, -1))[0]

        # 找匹配上下文
        dt = _utc_ms_to_datetime(ts_ms)
        ctx = context_key(dt["weekday"], dt["hour"])

        gmm = self._gmms.get(ctx)
        if gmm is None:
            nearest = self._find_nearest_context(ctx)
            if nearest is None:
                return float("inf")
            gmm = self._gmms[nearest]
            ctx = nearest

        log_prob = gmm.score_samples(reduced.reshape(1, -1))[0]
        return float(-log_prob)

    def score_nll_percentile(self, S_t: np.ndarray, ts_ms: int) -> float:
        """
        返回当前帧 NLL 在基线 NLL 分布中的 percentile (0-1)。
        高值=异常偏离。基线未就绪时返回 0。
        """
        nll = self.score_nll(S_t, ts_ms)
        if nll == float("inf"):
            return 0.0

        dt = _utc_ms_to_datetime(ts_ms)
        ctx = context_key(dt["weekday"], dt["hour"])

        stats = self._nll_stats.get(ctx)
        if stats is None:
            return 0.0

        mean, std = stats["mean"], stats["std"]
        if std < 1e-10:
            return 0.0

        # Z-score → percentile (单边：只关心 NLL 高于基线)
        z = max(0.0, (nll - mean) / std)
        from math import erf
        pct = 0.5 * (1.0 + erf(z / np.sqrt(2)))
        return float(np.clip(2.0 * pct - 1.0, 0.0, 1.0))  # 映射到 0-1

    def get_nll_stats(self, ctx: str) -> dict | None:
        """获取指定上下文的 NLL 统计。"""
        return self._nll_stats.get(ctx)

    def _find_nearest_context(self, ctx: str) -> str | None:
        """找不到精确上下文时，找最近的。"""
        if not self._gmms:
            return None
        # 优先同 day type (weekday/weekend)
        same_type = [k for k in self._gmms if k[:2] == ctx[:2]]
        if same_type:
            return same_type[0]
        return list(self._gmms.keys())[0]

    def update(self, S_new: np.ndarray, timestamps_ms: np.ndarray,
               S_old: np.ndarray, ts_old: np.ndarray):
        """增量更新 — 追加数据后重新拟合。
        S_old, ts_old: 之前的训练数据（调用方负责保存和传入）"""
        S_all = np.vstack([S_old, S_new])
        ts_all = np.concatenate([ts_old, timestamps_ms])
        self.fit(S_all, ts_all)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "gmms": self._gmms,
                "pca": self._pca,
                "scores": self._scores,
                "nll_stats": self._nll_stats,
                "pca_dim": self._pca_dim,
                "n_components": self._n_components,
                "use_contexts": self._use_contexts,
                "is_ready": self.is_ready,
            }, f)

    @classmethod
    def load(cls, path: str) -> "LatentBaseline":
        lb = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        lb._gmms = data["gmms"]
        lb._pca = data["pca"]
        lb._scores = data["scores"]
        lb._pca_dim = data.get("pca_dim", PCA_DIM)
        lb._n_components = data.get("n_components", N_COMPONENTS)
        lb._use_contexts = data.get("use_contexts", True)
        lb._nll_stats = data.get("nll_stats", {})
        lb.is_ready = data["is_ready"]
        return lb


def _mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    try:
        inv_cov = np.linalg.inv(cov)
        diff = x - mean
        return float(np.sqrt(diff @ inv_cov @ diff))
    except np.linalg.LinAlgError:
        return 10.0


def _utc_ms_to_datetime(ts_ms: int) -> dict:
    """UTC 毫秒 → weekday + hour（float）"""
    import datetime
    # 简化：用本地时区（+8），产品中应存 UTC 偏移
    dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000.0) + datetime.timedelta(hours=8)
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return {"weekday": dt.weekday(), "hour": hour}
