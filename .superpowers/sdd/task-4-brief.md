### Task 4: 生命体征提取

**Files:**
- Create: `ai/vital_signs/__init__.py`
- Create: `ai/vital_signs/filters.py`
- Create: `ai/vital_signs/quiet_detector.py`
- Create: `ai/vital_signs/extractor.py`
- Create: `ai/tests/test_vital_signs.py`

**Interfaces:**
- Consumes: `FeatureVector` from Task 1, `VitalFrame` from Task 1
- Produces: `extract_breath_rate(phase_signal, fs) -> tuple[float, float]`, `extract_heart_rate(...)`, `is_quiet(fv, history) -> bool`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/vital_signs
touch ai/vital_signs/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_vital_signs.py`:

```python
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
        assert len(b) == 5  # 4th order → 5 coefs

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
        assert quality > 0.5      # SNR 应较高

    def test_breath_rate_quality_low_on_noise(self):
        """高噪声 → quality 降低"""
        fs = 20.0
        signal = make_test_signal(0.3, 20.0, fs, noise_std=1.0)
        _, quality = extract_breath_rate(signal, fs)
        assert quality < 0.5

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
        history = [self.make_fv() for _ in range(25)]  # ~5s @ 5fps
        assert is_quiet(self.make_fv(), history) is True

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
```

- [ ] **Step 3: Run test (verify failure)**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_vital_signs.py -v
```
Expected: ModuleNotFoundError

- [ ] **Step 4: Implement `ai/vital_signs/filters.py`**

```python
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
```

- [ ] **Step 5: Implement `ai/vital_signs/extractor.py`**

```python
"""从雷达相位信号提取呼吸率和心率 — 经典 FFT 频谱分析"""
import numpy as np


def _extract_peak_frequency(signal: np.ndarray, fs: float,
                             lowcut: float, highcut: float) -> tuple[float, float]:
    """
    对信号加窗 → FFT → 在 [lowcut, highcut] 频带内找最高峰。
    Returns: (peak_freq_hz, quality=peak/noise_floor_mean)
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
    quality = band_fft[peak_idx] / (np.mean(fft[band_mask]) + 1e-10)

    return peak_freq, min(float(quality), 1.0)


def extract_breath_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取呼吸率。
    Args:
        phase_signal: ≥20s 的相位序列
        fs: 采样率 (Hz)
    Returns:
        (breath_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    filtered = apply_bandpass_filter(phase_signal, 0.1, 0.5, fs)  # 6-30 bpm
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.1, 0.5)
    return round(freq_hz * 60, 1), round(quality, 3)


def extract_heart_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取心率。
    Args:
        phase_signal: ≥10s 的相位序列
        fs: 采样率 (Hz)
    Returns:
        (heart_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    filtered = apply_bandpass_filter(phase_signal, 0.8, 2.5, fs)  # 48-150 bpm
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.8, 2.5)
    return round(freq_hz * 60, 1), round(quality, 3)
```

- [ ] **Step 6: Implement `ai/vital_signs/quiet_detector.py`**

```python
"""安静态判定 — 只有在安静态才适合提取生命体征"""
from ai.shared.types import FeatureVector

QUIET_DURATION_S = 5.0
VELOCITY_VAR_THRESHOLD = 0.05


def is_quiet(fv: FeatureVector, history: list[FeatureVector]) -> bool:
    """
    判断当前是否处于安静态。
    条件: posture in (sit, lie) + not moving + 持续 ≥5s + 速度方差低
    """
    if fv.posture not in ("sit", "lie"):
        return False
    if fv.moving:
        return False
    if fv.velocity_variance > VELOCITY_VAR_THRESHOLD:
        return False

    # 检查持续时间
    device_hist = [h for h in history if h.device_id == fv.device_id]
    device_hist.sort(key=lambda h: h.ts)

    # 从当前帧往回找连续静止的时长
    quiet_start = fv.ts
    for h in reversed(device_hist):
        if h.ts >= fv.ts:
            continue
        if h.moving or h.posture not in ("sit", "lie"):
            quiet_start = h.ts
            break
        quiet_start = h.ts

    duration_s = (fv.ts - quiet_start) / 1000.0
    return duration_s >= QUIET_DURATION_S
```

- [ ] **Step 7: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_vital_signs.py -v
```
Expected: all passed

- [ ] **Step 8: Commit**

```bash
git add ai/vital_signs/ ai/tests/test_vital_signs.py
git commit -m "feat(ai): add vital signs extraction (FFT-based) and quiet state detector

Classic signal processing: Butterworth bandpass → FFT peak detection.
Tested with synthetic sine waves. Requires real radar phase signal for
production use; simulator provides vital signs directly via VitalFrame.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

