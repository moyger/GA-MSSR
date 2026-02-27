"""End-to-end data pipeline: load -> resample -> denoise -> output DataFrame ready for indicators."""
import pandas as pd
from pathlib import Path
from typing import Optional

from data.loader import load_nq_data
from data.wavelet_filter import WaveletFilter


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """
    Resample OHLCV data to a larger timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    timeframe : str
        Pandas resample frequency string, e.g. '5min', '15min', '1h'.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame.
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


def build_denoised_dataset(
    filepath: str | Path,
    denoise_columns: str = "close",
    wavelet: str = "db4",
    level: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load NQ data, apply wavelet denoising, return enriched DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to NQ OHLCV data file.
    denoise_columns : str
        Which columns to denoise: 'close' (default), 'ohlc', or 'none'.
    wavelet : str
        Wavelet name (default 'db4').
    level : int or None
        Decomposition level (None = auto).
    start, end : str or None
        Optional date filters.
    timeframe : str or None
        Resample to this timeframe before denoising, e.g. '5min', '15min'.
        None means no resampling (use original bars).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, close_raw, close_denoised.
        If denoise_columns='ohlc', also includes open_raw, high_raw, low_raw.
    """
    df = load_nq_data(filepath, start=start, end=end)

    if timeframe is not None:
        df = resample_ohlcv(df, timeframe)
    wf = WaveletFilter(wavelet=wavelet, level=level)

    if denoise_columns == "none":
        df["close_raw"] = df["close"].copy()
        df["close_denoised"] = df["close"].copy()
        return df

    if denoise_columns == "close":
        df["close_raw"] = df["close"].copy()
        df["close"] = wf.denoise(df["close"].values)
        df["close_denoised"] = df["close"].copy()

    elif denoise_columns == "ohlc":
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_raw"] = df[col].copy()
        denoised = wf.denoise_ohlc(df[["open", "high", "low", "close"]])
        df[["open", "high", "low", "close"]] = denoised
        df["close_denoised"] = df["close"].copy()

    else:
        raise ValueError(f"Invalid denoise_columns: {denoise_columns!r}")

    return df
