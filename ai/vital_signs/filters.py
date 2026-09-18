"""数字滤波器 — 呼吸/心率提取的带通滤波"""
import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """设计 Butterworth 带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                           fs: float, order: int = 4) -> np.ndarray:
    """零相位带通滤波"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)
