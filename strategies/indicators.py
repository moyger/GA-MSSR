"""
Technical indicators for the Khushi 16-rule trading system.

Ported from reference/Rule-based-forex-trading-system/ta.py with bug fixes:
- RSI: fixed in-place mutation of input Series
- Consistent snake_case naming
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=n, min_periods=n).mean()


def ma(close: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(n).mean()


def dema(close: pd.Series, n: int) -> pd.Series:
    """Double Exponential Moving Average: 2*EMA - EMA(EMA)."""
    e = ema(close, n)
    return 2 * e - ema(e, n)


def tema(close: pd.Series, n: int) -> pd.Series:
    """Triple Exponential Moving Average: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e = ema(close, n)
    ee = ema(e, n)
    return 3 * e - 3 * ee + ema(ee, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Fixed from reference: uses .copy() to avoid mutating the input Series.
    """
    diff = close.diff(1)
    up = diff.copy()
    dn = diff.copy() * 0

    which_dn = diff < 0
    up[which_dn] = 0
    dn[which_dn] = -diff[which_dn]

    ema_up = ema(up, n)
    ema_dn = ema(dn, n)
    result = 100 * ema_up / (ema_up + ema_dn)
    return pd.Series(result, name="rsi")


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Stochastic %K."""
    smin = low.rolling(n, min_periods=0).min()
    smax = high.rolling(n, min_periods=0).max()
    stoch_k = 100 * (close - smin) / (smax - smin)
    return pd.Series(stoch_k, name="stoch_k")


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, c: float = 0.015) -> pd.Series:
    """Commodity Channel Index."""
    pp = (high + low + close) / 3.0
    result = (pp - pp.rolling(n, min_periods=0).mean()) / (c * pp.rolling(n, min_periods=0).std())
    return pd.Series(result, name="cci")


def vortex_indicator_pos(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Vortex Indicator Positive (VI+)."""
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()
    vmp = np.abs(high - low.shift(1))
    vip = vmp.rolling(n, min_periods=0).sum() / trn
    return pd.Series(vip, name="vip")


def vortex_indicator_neg(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Vortex Indicator Negative (VI-)."""
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()
    vmm = np.abs(low - high.shift(1))
    vin = vmm.rolling(n).sum() / trn
    return pd.Series(vin, name="vin")


def ichimoku_a(high: pd.Series, low: pd.Series, n1: int = 9, n2: int = 26) -> pd.Series:
    """Ichimoku Span A (Senkou Span A)."""
    conv = 0.5 * (high.rolling(n1, min_periods=0).max() + low.rolling(n1, min_periods=0).min())
    base = 0.5 * (high.rolling(n2, min_periods=0).max() + low.rolling(n2, min_periods=0).min())
    spana = 0.5 * (conv + base)
    return pd.Series(spana, name=f"ichimoku_a_{n2}")


def ichimoku_b(high: pd.Series, low: pd.Series, n2: int = 26, n3: int = 52) -> pd.Series:
    """Ichimoku Span B (Senkou Span B)."""
    spanb = 0.5 * (high.rolling(n3, min_periods=0).max() + low.rolling(n3, min_periods=0).min())
    return pd.Series(spanb, name=f"ichimoku_b_{n2}")


def bollinger_hband(close: pd.Series, n: int = 20, ndev: int = 2) -> pd.Series:
    """Bollinger Band Upper."""
    mavg = close.rolling(n, min_periods=0).mean()
    mstd = close.rolling(n, min_periods=0).std()
    return pd.Series(mavg + ndev * mstd, name="hband")


def bollinger_lband(close: pd.Series, n: int = 20, ndev: int = 2) -> pd.Series:
    """Bollinger Band Lower."""
    mavg = close.rolling(n, min_periods=0).mean()
    mstd = close.rolling(n, min_periods=0).std()
    return pd.Series(mavg - ndev * mstd, name="lband")


def keltner_channel_hband(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10) -> pd.Series:
    """Keltner Channel Upper Band."""
    tp = ((4 * high) - (2 * low) + close) / 3.0
    return pd.Series(tp.rolling(n, min_periods=0).mean(), name="kc_hband")


def keltner_channel_lband(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10) -> pd.Series:
    """Keltner Channel Lower Band."""
    tp = ((-2 * high) + (4 * low) + close) / 3.0
    return pd.Series(tp.rolling(n, min_periods=0).mean(), name="kc_lband")


def donchian_channel_hband(close: pd.Series, n: int = 20) -> pd.Series:
    """Donchian Channel Upper Band (rolling max)."""
    return pd.Series(close.rolling(n, min_periods=0).max(), name="dchband")


def donchian_channel_lband(close: pd.Series, n: int = 20) -> pd.Series:
    """Donchian Channel Lower Band (rolling min)."""
    return pd.Series(close.rolling(n, min_periods=0).min(), name="dclband")
