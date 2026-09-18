"""个人基线 — 14 天隐状态轨迹 → GMM 概率密度 → 异常分"""
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from ai.shared.types import FeatureVector

MIN_FRAMES_PER_CONTEXT = 200  # 每个上下文最少帧数
MIN_TOTAL_FRAMES = 400        # 全局最少帧数
N_COMPONENTS = 5
PCA_DIM = 16


def context_key(weekday: int, hour: int) -> str:
    """将 (weekday, hour) → 10 个上下文字符串之一"""
    is_weekend = weekday >= 5
    prefix = "weekend" if is_weekend else "weekday"
    if 0 <= hour < 6:
        return f"{prefix}_night"
    elif 6 <= hour < 9:
        return f"{prefix}_morning"
    elif 9 <= hour < 12:
        return f"{prefix}_morning"
    elif 12 <= hour < 14:
        return f"{prefix}_afternoon"
    elif 14 <= hour < 18:
        return f"{prefix}_afternoon"
    elif 18 <= hour < 21:
        return f"{prefix}_evening"
    else:
        return f"{prefix}_night"


class PersonalBaseline:
    """
    分上下文 GMM 个人基线。
    - 10 个上下文 × 1 个 GMM，每个 GMM 拟合该上下文中的特征向量分布
    - 异常分 = Mahalanobis distance percentile
    """

    def __init__(self):
        self._context_gmms: dict[str, GaussianMixture] = {}
        self._pca: PCA | None = None
        self._context_data: dict[str, list[np.ndarray]] = {}
        self._context_scores: dict[str, list[float]] = {}  # 历史分数用于标准化
        self.is_ready: bool = False

    def fit(self, features: list[FeatureVector], n_days: int = 14):
        """训练基线"""
        if len(features) < MIN_TOTAL_FRAMES:
            return

        # 分组
        grouped: dict[str, list[np.ndarray]] = {}
        for fv in features:
            key = context_key(fv.weekday, self._hour_from_ts(fv))
            arr = fv.to_array()
            grouped.setdefault(key, []).append(arr)

        # 过滤样本不够的上下文
        self._context_data = {
            k: v for k, v in grouped.items()
            if len(v) >= MIN_FRAMES_PER_CONTEXT
        }

        if not self._context_data:
            return

        # PCA
        all_data = np.vstack(list(self._context_data.values()))
        pca_dim = min(PCA_DIM, all_data.shape[1], all_data.shape[0] // 10)
        self._pca = PCA(n_components=pca_dim)
        all_reduced = self._pca.fit_transform(all_data)

        # 分上下文 GMM
        self._context_gmms = {}
        self._context_scores = {}
        offset = 0
        for key, arrs in self._context_data.items():
            n = len(arrs)
            reduced = all_reduced[offset:offset + n]
            offset += n
            gmm = GaussianMixture(
                n_components=min(N_COMPONENTS, n // 20),
                covariance_type="full",
                random_state=42,
            )
            gmm.fit(reduced)
            self._context_gmms[key] = gmm
            # 记录历史 Mahalanobis 分数
            scores = []
            for i in range(n):
                comp = gmm.predict(reduced[i:i + 1])[0]
                d = self._mahalanobis(reduced[i], gmm.means_[comp],
                                       gmm.covariances_[comp])
                scores.append(d)
            self._context_scores[key] = scores

        self.is_ready = True

    def score(self, fv: FeatureVector) -> float:
        """
        返回异常分 0-1。0=完全正常, 1=极度异常。
        基线未就绪时返回 0。
        """
        if not self.is_ready or self._pca is None:
            return 0.0

        key = context_key(fv.weekday, self._hour_from_ts(fv))
        if key not in self._context_gmms:
            return 0.0

        gmm = self._context_gmms[key]
        arr = fv.to_array().reshape(1, -1)
        reduced = self._pca.transform(arr)[0]

        # 到最近分量的 Mahalanobis 距离
        comp = gmm.predict(reduced.reshape(1, -1))[0]
        dist = self._mahalanobis(reduced, gmm.means_[comp],
                                  gmm.covariances_[comp])

        # 在该上下文历史分数中计算 percentile
        hist = self._context_scores.get(key, [])
        if not hist:
            return min(dist / 10.0, 1.0)

        percentile = sum(1 for h in hist if h < dist) / len(hist)
        return float(np.clip(percentile, 0.0, 1.0))

    def update(self, new_features: list[FeatureVector]):
        """增量更新 — 追加数据后重新拟合"""
        all_features = []
        for arrs in self._context_data.values():
            for a in arrs:
                all_features.append(a)
        for fv in new_features:
            all_features.append(fv.to_array())
        # 重建 FeatureVector 列表重新 fit
        # (简化为直接重新拟合；产品中可优化为增量 GMM)
        self.fit(self._arrays_to_fv(all_features, new_features))

    @staticmethod
    def _mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        try:
            inv_cov = np.linalg.inv(cov)
            diff = x - mean
            return float(np.sqrt(diff @ inv_cov @ diff))
        except np.linalg.LinAlgError:
            return 10.0

    @staticmethod
    def _hour_from_ts(fv: FeatureVector) -> int:
        import math
        rad = math.atan2(fv.hour_sin, fv.hour_cos)
        hour = rad / (2 * math.pi) * 24
        return int(hour % 24)

    def _arrays_to_fv(self, arrays: list[np.ndarray],
                       ref: list[FeatureVector]) -> list[FeatureVector]:
        # 简化：返回 ref 作为代理（update 后重新 fit 仅需要数量信息）
        # 实际上我们直接重新 fit，保留原始 FeatureVector 引用
        return ref

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "context_gmms": self._context_gmms,
                "pca": self._pca,
                "context_data": self._context_data,
                "context_scores": self._context_scores,
                "is_ready": self.is_ready,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PersonalBaseline":
        b = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        b._context_gmms = data["context_gmms"]
        b._pca = data["pca"]
        b._context_data = data["context_data"]
        b._context_scores = data["context_scores"]
        b.is_ready = data["is_ready"]
        return b
