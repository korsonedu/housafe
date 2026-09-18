"""从雷达相位信号提取呼吸率和心率 — 经典 FFT 频谱分析"""
import numpy as np


def _has_band_energy(signal: np.ndarray, fs: float,
                     lowcut: float, highcut: float,
                     min_ratio: float = 0.02) -> bool:
    """检查原始信号的 FFT 在 [lowcut, highcut] 带内是否有足够能量"""
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band_mask = (freqs >= lowcut) & (freqs <= highcut)
    total_e = float(np.sum(fft)) + 1e-12
    band_e = float(np.sum(fft[band_mask]))
    return band_e / total_e >= min_ratio


def _extract_peak_frequency(signal: np.ndarray, fs: float,
                             lowcut: float, highcut: float) -> tuple[float, float]:
    """
    对滤波后信号加窗 → FFT → 在 [lowcut, highcut] 频带内找最高峰。
    Returns: (peak_freq_hz, quality=0~1)
    """
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1 / fs)

    band_mask = (freqs >= lowcut) & (freqs <= highcut)
    if not np.any(band_mask):
        return 0.0, 0.0

    band_fft = fft[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = np.argmax(band_fft)
    peak_freq = band_freqs[peak_idx]

    # SNR: 峰值 vs band 内其他 bin 的均值（排除峰值±2 bin 避免频谱泄漏）
    exclude = set(range(max(0, peak_idx - 2), min(len(band_fft), peak_idx + 3)))
    noise_bins = [band_fft[i] for i in range(len(band_fft)) if i not in exclude]
    if not noise_bins:
        return peak_freq, 1.0
    noise_floor = float(np.mean(noise_bins)) + 1e-10
    snr = float(band_fft[peak_idx]) / noise_floor
    # SNR=1 → 0, SNR=3 → ~0.3, SNR=8+ → 1.0
    normalized = min(max((snr - 1.0) / 7.0, 0.0), 1.0)

    return peak_freq, normalized


def extract_breath_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取呼吸率。
    Args:
        phase_signal: 相位序列
        fs: 采样率 (Hz)
    Returns:
        (breath_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    # 在原始信号上检查带内能量（避免滤波瞬态干扰）
    if not _has_band_energy(phase_signal, fs, 0.1, 0.5, min_ratio=0.02):
        return 0.0, 0.0

    filtered = apply_bandpass_filter(phase_signal, 0.1, 0.5, fs)
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.1, 0.5)
    return round(freq_hz * 60, 1), round(quality, 3)


def extract_heart_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取心率。
    Args:
        phase_signal: 相位序列
        fs: 采样率 (Hz)
    Returns:
        (heart_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    if not _has_band_energy(phase_signal, fs, 0.8, 2.5, min_ratio=0.02):
        return 0.0, 0.0

    filtered = apply_bandpass_filter(phase_signal, 0.8, 2.5, fs)
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.8, 2.5)
    return round(freq_hz * 60, 1), round(quality, 3)
