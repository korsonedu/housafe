"""生命体征提取测试 — 用合成正弦波验证信号处理管线"""
import numpy as np
from ai.vital_signs.filters import butter_bandpass, apply_bandpass_filter
from ai.vital_signs.extractor import extract_breath_rate, extract_heart_rate
from ai.vital_signs.quiet_detector import is_quiet
from ai.shared.types import FeatureVector


def make_test_signal(freq_hz: float, duration_s: float, fs: float,
                      noise_std: float = 0.01) -> np.ndarray:
    """生成带噪声的合成正弦波信号"""
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    signal = np.sin(2 * np.pi * freq_hz * t)
    noise = np.random.normal(0, noise_std, len(t))
    return signal + noise


class TestFilters:
    def test_bandpass_shape(self):
        b, a = butter_bandpass(0.1, 0.5, fs=20.0, order=4)
        assert len(b) == 9  # 4th-order bandpass → 2N+1 = 9 coefs（bandpass 阶数翻倍）

    def test_apply_bandpass_no_crash(self):
        signal = make_test_signal(0.3, 20, 20.0)
        result = apply_bandpass_filter(signal, 0.1, 0.5, fs=20.0)
        assert len(result) == len(signal)
        assert not np.any(np.isnan(result))


class TestBreathRate:
    def test_breath_rate_from_synthetic(self):
        """合成 0.3Hz (=18 bpm) 正弦波 → 应检出 18 bpm"""
        fs = 20.0
        signal = make_test_signal(0.3, 20.0, fs, noise_std=0.005)
        rate, quality = extract_breath_rate(signal, fs)
        assert 15 <= rate <= 21  # 允许 ±3 bpm 误差
        assert quality > 0.3      # 清晰信号 quality 应较高

    def test_breath_rate_quality_decreases_with_noise(self):
        """噪声越高 → quality 降低（相对比较）"""
        fs = 20.0
        signal_clean = make_test_signal(0.3, 20.0, fs, noise_std=0.001)
        signal_noisy = make_test_signal(0.3, 20.0, fs, noise_std=5.0)
        _, q_clean = extract_breath_rate(signal_clean, fs)
        _, q_noisy = extract_breath_rate(signal_noisy, fs)
        assert q_noisy < q_clean, f"noisy quality {q_noisy} should be < clean quality {q_clean}"

    def test_breath_rate_out_of_band_zero_quality(self):
        """带外信号（2Hz）→ breath band (0.1-0.5Hz) 内能量 <1% → quality=0"""
        fs = 20.0
        t = np.linspace(0, 5, 100, endpoint=False)
        signal = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz, way outside 0.1-0.5 Hz band
        rate, quality = extract_breath_rate(signal, fs)
        assert quality == 0.0, f"out-of-band signal should have zero quality, got {quality}"


class TestHeartRate:
    def test_heart_rate_from_synthetic(self):
        """合成 1.2Hz (=72 bpm) 正弦波 → 应检出 72 bpm"""
        fs = 20.0
        signal = make_test_signal(1.2, 10.0, fs, noise_std=0.005)
        rate, quality = extract_heart_rate(signal, fs)
        assert 65 <= rate <= 79
        assert quality > 0.5


class TestQuietDetector:
    def make_fv(self, posture="sit", moving=False, vel_var=0.01):
        return FeatureVector(
            ts=1000, device_id="r1", room="bedroom",
            posture=posture, posture_confidence=0.9, presence=True,
            moving=moving, centroid=(1, 1, 0.5), height=0.3, n_points=10,
            occupancy_estimate=1, resp_rate=16.0, heart_rate=72.0,
            vital_quality=0.85, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=vel_var,
        )

    def test_sitting_still_is_quiet(self):
        # 递增时间戳：保证当前帧 ts > 所有历史帧 ts
        fvs = [self.make_fv() for _ in range(25)]
        history = []
        for i, fv in enumerate(fvs):
            history.append(FeatureVector(
                ts=500 + i * 200, device_id=fv.device_id, room=fv.room,
                posture=fv.posture, posture_confidence=fv.posture_confidence,
                presence=fv.presence, moving=fv.moving,
                centroid=fv.centroid, height=fv.height, n_points=fv.n_points,
                occupancy_estimate=fv.occupancy_estimate,
                resp_rate=fv.resp_rate, heart_rate=fv.heart_rate,
                vital_quality=fv.vital_quality,
                hour_sin=fv.hour_sin, hour_cos=fv.hour_cos,
                weekday=fv.weekday, velocity_variance=fv.velocity_variance,
            ))
        current = FeatureVector(
            ts=6000, device_id="r1", room="bedroom",
            posture="sit", posture_confidence=0.9, presence=True,
            moving=False, centroid=(1, 1, 0.5), height=0.3, n_points=10,
            occupancy_estimate=1, resp_rate=16.0, heart_rate=72.0,
            vital_quality=0.85, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.01,
        )
        assert is_quiet(current, history) is True

    def test_standing_not_quiet(self):
        fv = self.make_fv(posture="stand")
        history = [fv] * 25
        assert is_quiet(fv, history) is False

    def test_moving_not_quiet(self):
        fv = self.make_fv(moving=True)
        history = [fv] * 25
        assert is_quiet(fv, history) is False

    def test_short_duration_not_quiet(self):
        """静止时长不足 → 非安静态"""
        fv = self.make_fv()
        history = [fv] * 3  # 不够 5s
        assert is_quiet(fv, history) is False
