"""
Wavelet denoising filter for financial time series.

Implements Discrete Wavelet Transform (DWT) denoising using the Daubechies
db4 wavelet with universal soft thresholding, as specified in the GA-MSSR
framework for filtering high-frequency noise while preserving price jumps.
"""
import numpy as np
import pandas as pd
import pywt
from typing import Optional


class WaveletFilter:
    """
    Applies DWT denoising to financial price series.

    Uses the Daubechies 4 (db4) wavelet with soft thresholding.
    The universal threshold is: sigma * sqrt(2 * ln(n))
    where sigma is estimated from the finest-level detail coefficients
    using the Median Absolute Deviation (MAD) estimator:
        sigma = median(|detail_coeffs|) / 0.6745

    Parameters
    ----------
    wavelet : str
        Wavelet name (default: 'db4').
    level : int or None
        Decomposition level. If None, uses pywt.dwt_max_level().
    mode : str
        Signal extension mode for DWT (default: 'symmetric').
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: Optional[int] = None,
        mode: str = "symmetric",
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.last_sigma: float = 0.0
        self.last_threshold: float = 0.0

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Denoise a 1-D signal using DWT soft thresholding.

        Parameters
        ----------
        signal : np.ndarray
            Input 1-D array (e.g., close prices).

        Returns
        -------
        np.ndarray
            Denoised signal, same length as input.
        """
        signal = np.array(signal, dtype=np.float64, copy=True)
        n = len(signal)

        if n < 2:
            raise ValueError(f"Signal too short for wavelet denoising: {n}")

        # Determine decomposition level
        level = self.level
        if level is None:
            level = pywt.dwt_max_level(n, pywt.Wavelet(self.wavelet).dec_len)
            level = max(1, level)

        # Multilevel DWT decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=level)

        # Estimate noise sigma from finest detail coefficients
        detail_finest = coeffs[-1]
        self.last_sigma = self._estimate_sigma(detail_finest)

        # Universal threshold: sigma * sqrt(2 * ln(n))
        self.last_threshold = self.last_sigma * np.sqrt(2.0 * np.log(n))

        # Soft-threshold all detail coefficients, keep approximation intact
        denoised_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            denoised_coeffs.append(
                pywt.threshold(detail, value=self.last_threshold, mode="soft")
            )

        # Reconstruct
        reconstructed = pywt.waverec(denoised_coeffs, self.wavelet, mode=self.mode)

        # waverec can return 1 element longer due to padding; trim
        return reconstructed[:n]

    def denoise_series(self, series: pd.Series) -> pd.Series:
        """Denoise a pandas Series, preserving its index."""
        denoised_values = self.denoise(series.values)
        return pd.Series(denoised_values, index=series.index, name=series.name)

    def denoise_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denoise all four OHLC columns independently.

        A post-processing clamp enforces OHLC consistency after
        denoising (high >= max(open,close), low <= min(open,close)).
        """
        result = df.copy()
        for col in ["open", "high", "low", "close"]:
            if col in result.columns:
                result[col] = self.denoise(result[col].values)

        result = self._clamp_ohlc(result)
        return result

    @staticmethod
    def _clamp_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce high >= max(open, close) and low <= min(open, close)."""
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            oc_max = df[["open", "close"]].max(axis=1)
            oc_min = df[["open", "close"]].min(axis=1)
            df["high"] = df["high"].clip(lower=oc_max)
            df["low"] = df["low"].clip(upper=oc_min)
        return df

    @staticmethod
    def _estimate_sigma(detail_coeffs: np.ndarray) -> float:
        """
        Estimate noise std via MAD: sigma = median(|d|) / 0.6745

        The 0.6745 constant is the 75th percentile of the standard
        normal distribution, making this a consistent estimator of
        sigma for Gaussian noise.
        """
        mad = np.median(np.abs(detail_coeffs))
        return mad / 0.6745 if mad > 0 else 1e-10
