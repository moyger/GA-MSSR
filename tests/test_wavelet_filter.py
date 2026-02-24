"""Tests for WaveletFilter."""
import numpy as np
import pandas as pd
import pytest
from data.wavelet_filter import WaveletFilter


@pytest.fixture
def wf():
    return WaveletFilter()


class TestDenoise:
    def test_preserves_length(self, wf):
        signal = np.random.randn(200)
        result = wf.denoise(signal)
        assert len(result) == len(signal)

    def test_reduces_noise(self, wf):
        """Denoised signal should be closer to clean signal than noisy input."""
        t = np.linspace(0, 4 * np.pi, 500)
        clean = np.sin(t) * 100 + 16800
        noisy = clean + np.random.randn(500) * 5
        denoised = wf.denoise(noisy)

        rmse_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
        rmse_denoised = np.sqrt(np.mean((denoised - clean) ** 2))
        assert rmse_denoised < rmse_noisy

    def test_preserves_jumps(self, wf):
        """A large price jump should survive denoising."""
        signal = np.ones(200) * 16800.0
        signal[100:] = 16900.0  # 100-point jump
        denoised = wf.denoise(signal)

        # The jump should still be visible (at least 80% preserved)
        jump_size = denoised[150] - denoised[50]
        assert jump_size > 80.0

    def test_preserves_trend(self, wf):
        """A monotonic trend should remain approximately monotonic."""
        signal = np.linspace(16800, 16900, 300)
        denoised = wf.denoise(signal)
        # Check that denoised is mostly increasing
        diffs = np.diff(denoised)
        pct_increasing = np.mean(diffs > -0.5)
        assert pct_increasing > 0.9

    def test_short_signal_raises(self, wf):
        with pytest.raises(ValueError, match="too short"):
            wf.denoise(np.array([1.0]))

    def test_two_element_signal(self, wf):
        result = wf.denoise(np.array([16800.0, 16805.0]))
        assert len(result) == 2

    def test_custom_wavelet(self):
        wf = WaveletFilter(wavelet="haar")
        signal = np.random.randn(100) + 16800
        result = wf.denoise(signal)
        assert len(result) == 100

    def test_sigma_estimation(self, wf):
        """Known noise level should produce approximately correct sigma."""
        np.random.seed(42)
        noise = np.random.randn(10000) * 3.0
        wf.denoise(noise + 16800)
        # MAD estimator should recover sigma ~ 3.0 (within 50%)
        assert 1.5 < wf.last_sigma < 6.0

    def test_idempotence_approximation(self, wf):
        """Denoising a smooth signal should change it minimally."""
        smooth = np.linspace(16800, 16850, 200)
        denoised = wf.denoise(smooth)
        max_diff = np.max(np.abs(denoised - smooth))
        assert max_diff < 5.0  # less than 5 points difference


class TestDenoiseSeries:
    def test_preserves_index(self, wf):
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        series = pd.Series(np.random.randn(100) + 16800, index=idx, name="close")
        result = wf.denoise_series(series)
        assert result.index.equals(series.index)
        assert result.name == "close"


class TestDenoiseOHLC:
    def test_ohlc_constraints(self, wf):
        """After denoise_ohlc, high >= max(open,close) and low <= min(open,close)."""
        np.random.seed(0)
        n = 200
        base = np.cumsum(np.random.randn(n) * 2) + 16800
        df = pd.DataFrame({
            "open": base + np.random.randn(n) * 0.5,
            "high": base + np.abs(np.random.randn(n) * 3),
            "low": base - np.abs(np.random.randn(n) * 3),
            "close": base + np.random.randn(n) * 0.5,
        })
        result = wf.denoise_ohlc(df)

        oc_max = result[["open", "close"]].max(axis=1)
        oc_min = result[["open", "close"]].min(axis=1)
        assert (result["high"] >= oc_max - 1e-10).all()
        assert (result["low"] <= oc_min + 1e-10).all()

    def test_preserves_shape(self, wf):
        df = pd.DataFrame({
            "open": np.random.randn(100) + 16800,
            "high": np.random.randn(100) + 16810,
            "low": np.random.randn(100) + 16790,
            "close": np.random.randn(100) + 16800,
        })
        result = wf.denoise_ohlc(df)
        assert result.shape == df.shape
