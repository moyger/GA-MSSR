"""Data loading utilities for NQ 1-minute OHLCV data."""
import pandas as pd
from pathlib import Path
from typing import Optional

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def load_nq_data(
    filepath: str | Path,
    timestamp_col: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load NQ OHLCV data from CSV or Parquet.

    Parameters
    ----------
    filepath : str or Path
        Path to data file (.csv or .parquet).
    timestamp_col : str, optional
        Name of the timestamp column. If None, auto-detects.
    start : str, optional
        Start date filter (inclusive), e.g. '2024-01-01'.
    end : str, optional
        End date filter (inclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Handle timestamp -> DatetimeIndex
    df = _set_datetime_index(df, timestamp_col)

    # Rename common column variations
    rename_map = {"vol": "volume", "vol.": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add volume if missing
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Select and type-cast
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols].astype("float64")
    df = df.sort_index()

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    return df


def _set_datetime_index(
    df: pd.DataFrame, timestamp_col: Optional[str]
) -> pd.DataFrame:
    """Detect or use specified timestamp column, convert to DatetimeIndex."""
    candidates = ["timestamp", "datetime", "date", "local time", "time"]

    if timestamp_col:
        col = timestamp_col.strip().lower()
    else:
        col = None
        for c in candidates:
            if c in df.columns:
                col = c
                break

    if col and col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)
        df = df.set_index(col)
        df.index.name = "timestamp"
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"

    return df
