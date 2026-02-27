"""Tests for the end-to-end data pipeline."""
import numpy as np
import pytest
from pathlib import Path
from data.pipeline import build_denoised_dataset

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_nq_1m.csv"


class TestBuildDenoisedDataset:
    def test_close_only_mode(self):
        df = build_denoised_dataset(FIXTURE_PATH, denoise_columns="close")
        assert "close_raw" in df.columns
        assert "close_denoised" in df.columns
        # Denoised close should differ from raw (unless signal is perfectly smooth)
        assert not np.allclose(df["close"].values, df["close_raw"].values, atol=0.01)
        # close and close_denoised should be identical
        np.testing.assert_array_equal(df["close"].values, df["close_denoised"].values)

    def test_ohlc_mode(self):
        df = build_denoised_dataset(FIXTURE_PATH, denoise_columns="ohlc")
        for col in ["open_raw", "high_raw", "low_raw", "close_raw"]:
            assert col in df.columns
        assert "close_denoised" in df.columns

    def test_none_mode(self):
        df = build_denoised_dataset(FIXTURE_PATH, denoise_columns="none")
        np.testing.assert_array_equal(df["close"].values, df["close_raw"].values)
        np.testing.assert_array_equal(df["close"].values, df["close_denoised"].values)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid denoise_columns"):
            build_denoised_dataset(FIXTURE_PATH, denoise_columns="invalid")

    def test_output_has_core_columns(self):
        df = build_denoised_dataset(FIXTURE_PATH, denoise_columns="close")
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_custom_wavelet(self):
        df = build_denoised_dataset(FIXTURE_PATH, wavelet="haar")
        assert len(df) == 100
