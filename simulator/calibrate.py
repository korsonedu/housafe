"""从 3DPCHM 数据集拟合 GMM 参数 — 离线一次性校准"""
import argparse
import numpy as np
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture

# 3DPCHM 全部 12 个动作类（与 ai/data/public/stats.py 保持一致）
ACTION_NAMES = [
    "stand", "sit", "fall", "walk",
    "punch", "jump", "wave_left", "lean_left",
    "open_arms", "wave_right", "lean_right", "squat",
]

# 场景系统暴露的常用动作
SCENE_ACTIONS = {"stand", "sit", "fall", "walk", "squat", "lie"}

GMM_K_DEFAULT = 4
RANDOM_SEED = 42


@dataclass
class GMMParams:
    """单个动作的统计参数"""
    action: str
    gmm_weights: np.ndarray       # (K,)
    gmm_means: np.ndarray         # (K, 3) 空间均值
    gmm_covariances: np.ndarray   # (K, 3, 3) 协方差矩阵
    point_count_lambda: float     # Poisson λ
    velocity_mean: float
    velocity_std: float
    intensity_alpha: float        # Beta α
    intensity_beta: float         # Beta β
    max_displacement: float       # 质心每秒最大位移 (m/s)


def fit_action_gmm(points_list: list[np.ndarray], k: int = GMM_K_DEFAULT) -> GaussianMixture:
    """对单动作的所有帧点云拟合 GMM"""
    all_points = np.vstack([p[:, :3] for p in points_list if len(p) > 0])
    n_components = min(k, len(all_points) // 10)
    n_components = max(1, n_components)
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_SEED,
                          covariance_type="full", max_iter=200,
                          reg_covar=1e-4)
    gmm.fit(all_points)
    return gmm


def fit_point_count_lambda(points_list: list[np.ndarray]) -> float:
    """Poisson λ = 每帧点数均值"""
    counts = [len(p) for p in points_list]
    return float(np.mean(counts))


def fit_velocity_stats(points_list: list[np.ndarray]) -> tuple[float, float]:
    """拟合速度分布（点云第4列）"""
    velocities = []
    for p in points_list:
        if p.shape[1] >= 4:
            velocities.extend(p[:, 3].tolist())
    if not velocities:
        return 0.0, 0.5
    return float(np.mean(velocities)), float(np.std(velocities))


def fit_intensity_beta(points_list: list[np.ndarray]) -> tuple[float, float]:
    """拟合强度 Beta 分布（点云第5列），method of moments"""
    intensities = []
    for p in points_list:
        if p.shape[1] >= 5:
            intensities.extend(p[:, 4].tolist())
    if not intensities:
        return 1.0, 1.0
    arr = np.array(intensities)
    arr = arr[(arr > 0) & (arr < 1)]
    if len(arr) < 2:
        return 1.0, 1.0
    mu = arr.mean()
    var = arr.var()
    if var == 0:
        return 1.0, 1.0
    common = mu * (1 - mu) / var - 1
    alpha = max(mu * common, 0.01)
    beta_val = max((1 - mu) * common, 0.01)
    return float(alpha), float(beta_val)


def fit_max_displacement(points_list: list[np.ndarray], fps: float = 10.0) -> float:
    """从相邻帧质心位移估算 max_displacement (m/s)"""
    centroids = [p[:, :3].mean(axis=0) for p in points_list if len(p) > 0]
    if len(centroids) < 2:
        return 1.5
    displacements = [np.linalg.norm(centroids[i] - centroids[i - 1])
                     for i in range(1, len(centroids))]
    if not displacements:
        return 1.5
    return float(np.percentile(displacements, 95) * fps * 1.5)


def calibrate(dataset_path: str, k: int = GMM_K_DEFAULT) -> dict[str, GMMParams]:
    """主校准流程：加载数据集 → 逐动作拟合 → 返回参数字典"""
    data = np.load(dataset_path, allow_pickle=True)
    pts_list = data["points"]
    action_labels = data["action_labels"].astype(np.int32)

    params = {}
    for aid, name in enumerate(ACTION_NAMES):
        mask = action_labels == aid
        n_frames = mask.sum()
        if n_frames < 10:
            print(f"  [WARN] {name}: 仅 {n_frames} 帧，跳过")
            continue
        action_points = [pts_list[i] for i in np.where(mask)[0]]
        gmm = fit_action_gmm(action_points, k)
        params[name] = GMMParams(
            action=name,
            gmm_weights=gmm.weights_,
            gmm_means=gmm.means_,
            gmm_covariances=gmm.covariances_,
            point_count_lambda=fit_point_count_lambda(action_points),
            velocity_mean=fit_velocity_stats(action_points)[0],
            velocity_std=fit_velocity_stats(action_points)[1],
            intensity_alpha=fit_intensity_beta(action_points)[0],
            intensity_beta=fit_intensity_beta(action_points)[1],
            max_displacement=fit_max_displacement(action_points),
        )
        print(f"  {name}: {n_frames} 帧, λ={params[name].point_count_lambda:.1f}, "
              f"max_disp={params[name].max_displacement:.2f} m/s")
    return params


def save_params(params: dict[str, GMMParams], out_path: str):
    """保存参数到 npz"""
    save_dict = {}
    for name, p in params.items():
        save_dict[f"{name}_weights"] = p.gmm_weights
        save_dict[f"{name}_means"] = p.gmm_means
        save_dict[f"{name}_covs"] = p.gmm_covariances
        save_dict[f"{name}_pt_lambda"] = np.array(p.point_count_lambda)
        save_dict[f"{name}_vel_mean"] = np.array(p.velocity_mean)
        save_dict[f"{name}_vel_std"] = np.array(p.velocity_std)
        save_dict[f"{name}_int_alpha"] = np.array(p.intensity_alpha)
        save_dict[f"{name}_int_beta"] = np.array(p.intensity_beta)
        save_dict[f"{name}_max_disp"] = np.array(p.max_displacement)
    np.savez_compressed(out_path, **save_dict)


def load_params(npz_path: str) -> dict[str, GMMParams]:
    """从 npz 加载参数"""
    data = np.load(npz_path)
    params = {}
    for name in ACTION_NAMES:
        key = f"{name}_weights"
        if key not in data:
            continue
        params[name] = GMMParams(
            action=name,
            gmm_weights=data[f"{name}_weights"],
            gmm_means=data[f"{name}_means"],
            gmm_covariances=data[f"{name}_covs"],
            point_count_lambda=float(data[f"{name}_pt_lambda"]),
            velocity_mean=float(data[f"{name}_vel_mean"]),
            velocity_std=float(data[f"{name}_vel_std"]),
            intensity_alpha=float(data[f"{name}_int_alpha"]),
            intensity_beta=float(data[f"{name}_int_beta"]),
            max_displacement=float(data[f"{name}_max_disp"]),
        )
    return params


def main():
    parser = argparse.ArgumentParser(description="校准仿真参数（从 3DPCHM 数据集）")
    parser.add_argument("--dataset", required=True, help="3DPCHM 预处理 npz 路径")
    parser.add_argument("--out", required=True, help="输出参数文件路径")
    parser.add_argument("--k", type=int, default=GMM_K_DEFAULT, help="GMM 分量数 (default: 4)")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    params = calibrate(args.dataset, k=args.k)
    if not params:
        print("[ERROR] 没有拟合出任何动作参数")
        return
    save_params(params, args.out)
    print(f"\n参数已保存到: {args.out} ({len(params)} 个动作)")


if __name__ == "__main__":
    main()
