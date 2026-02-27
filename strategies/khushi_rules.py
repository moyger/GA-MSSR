"""
Khushi 16-rule signal generation engine.

Ported from reference/Rule-based-forex-trading-system/tradingrule.py with:
- Lowercase column convention (open, high, low, close)
- Bug fix: Rule15 now uses donchian_channel_lband for lower band
- Snake_case naming for public API

Each rule takes (params, OHLC) and returns (score, signal_series).
- score: abs(sum(signal * log_returns)) — magnitude of rule profitability
- signal: pd.Series of {-1, 0, 1} shifted by 1 bar (no look-ahead bias)
"""
import numpy as np
import pandas as pd
from typing import Optional

from strategies.indicators import (
    ema, dema, tema, rsi, stoch, cci,
    vortex_indicator_pos, vortex_indicator_neg,
    ichimoku_a, ichimoku_b,
    bollinger_hband, bollinger_lband,
    keltner_channel_hband, keltner_channel_lband,
    donchian_channel_hband, donchian_channel_lband,
)


# ---------------------------------------------------------------------------
# Type 1 Rules: Crossover (two-parameter)
# ---------------------------------------------------------------------------

def rule_1(param, ohlc):
    """MA x MA crossover."""
    ma1, ma2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = close.rolling(ma1).mean()
    s2 = close.rolling(ma2).mean()
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_2(param, ohlc):
    """EMA x MA crossover."""
    ema1, ma2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = ema(close, ema1)
    s2 = close.rolling(ma2).mean()
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_3(param, ohlc):
    """EMA x EMA crossover."""
    ema1, ema2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = ema(close, ema1)
    s2 = ema(close, ema2)
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_4(param, ohlc):
    """DEMA x MA crossover."""
    dema1, ma2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = dema(close, dema1)
    s2 = close.rolling(ma2).mean()
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_5(param, ohlc):
    """DEMA x DEMA crossover."""
    dema1, dema2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = dema(close, dema1)
    s2 = dema(close, dema2)
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_6(param, ohlc):
    """TEMA x MA crossover."""
    tema1, ma2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = tema(close, tema1)
    s2 = close.rolling(ma2).mean()
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_7(param, ohlc):
    """Stochastic x Stochastic MA."""
    stoch1, stochma2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = stoch(high, low, close, stoch1)
    s2 = s1.rolling(stochma2, min_periods=0).mean()
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_8(param, ohlc):
    """Vortex VI+ x VI- crossover."""
    vortex1, vortex2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = vortex_indicator_pos(high, low, close, vortex1)
    s2 = vortex_indicator_neg(high, low, close, vortex2)
    signal = (2 * (s1 < s2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_9(param, ohlc):
    """Ichimoku: price vs Span A and Span B."""
    p1, p2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = ichimoku_a(high, low, n1=p1, n2=round((p1 + p2) / 2))
    s2 = ichimoku_b(high, low, n2=round((p1 + p2) / 2), n3=p2)
    s3 = close
    signal = (
        -1 * ((s3 > s1) & (s3 > s2)).astype(int)
        + 1 * ((s3 < s2) & (s3 < s1)).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


# ---------------------------------------------------------------------------
# Type 2 Rules: Indicator vs constant threshold
# ---------------------------------------------------------------------------

def rule_10(param, ohlc):
    """RSI vs threshold."""
    rsi1, c2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = rsi(close, rsi1)
    signal = (2 * (s1 < c2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_11(param, ohlc):
    """CCI vs threshold."""
    cci1, c2 = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = cci(high, low, close, cci1)
    signal = (2 * (s1 < c2).astype(int) - 1).shift(1)
    return (abs((signal * logr).sum()), signal)


# ---------------------------------------------------------------------------
# Type 3 Rules: Oscillator with upper/lower bands
# ---------------------------------------------------------------------------

def rule_12(param, ohlc):
    """RSI with upper and lower thresholds."""
    rsi1, hl, ll = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = rsi(close, rsi1)
    signal = (
        -1 * (s1 > hl).astype(int)
        + 1 * (s1 < ll).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_13(param, ohlc):
    """CCI with upper and lower thresholds."""
    cci1, hl, ll = param
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = cci(high, low, close, cci1)
    signal = (
        -1 * (s1 > hl).astype(int)
        + 1 * (s1 < ll).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


# ---------------------------------------------------------------------------
# Type 4 Rules: Channel mean-reversion (single period)
# ---------------------------------------------------------------------------

def rule_14(period, ohlc):
    """Keltner Channel mean-reversion."""
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = keltner_channel_hband(high, low, close, n=period)
    s2 = keltner_channel_lband(high, low, close, n=period)
    signal = (
        -1 * (close > s1).astype(int)
        + 1 * (close < s2).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_15(period, ohlc):
    """Donchian Channel mean-reversion. (Fixed: uses lband for lower.)"""
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = donchian_channel_hband(close, n=period)
    s2 = donchian_channel_lband(close, n=period)
    signal = (
        -1 * (close > s1).astype(int)
        + 1 * (close < s2).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


def rule_16(period, ohlc):
    """Bollinger Band mean-reversion."""
    _open, high, low, close = ohlc
    logr = np.log(close / close.shift(1))
    s1 = bollinger_hband(close, n=period)
    s2 = bollinger_lband(close, n=period)
    signal = (
        -1 * (close > s1).astype(int)
        + 1 * (close < s2).astype(int)
    ).shift(1)
    return (abs((signal * logr).sum()), signal)


# ---------------------------------------------------------------------------
# All rules registry
# ---------------------------------------------------------------------------

ALL_RULES = [
    rule_1, rule_2, rule_3, rule_4, rule_5, rule_6,
    rule_7, rule_8, rule_9,
    rule_10, rule_11,
    rule_12, rule_13,
    rule_14, rule_15, rule_16,
]

# Default parameter search grid (from reference)
PERIOD_GRID = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
RSI_LIMITS = list(range(0, 101, 5))
CCI_LIMITS = list(range(-120, 121, 20))


# ---------------------------------------------------------------------------
# Training: find best parameters for each rule via grid search
# ---------------------------------------------------------------------------

def train_rule_params(
    df: pd.DataFrame,
    periods: Optional[list[int]] = None,
) -> list[tuple]:
    """
    Find optimal parameters for each of the 16 rules via grid search.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with lowercase columns (open, high, low, close).
    periods : list[int], optional
        Custom period grid. Defaults to PERIOD_GRID.

    Returns
    -------
    list[tuple]
        16 parameter tuples, one per rule.
    """
    if periods is None:
        periods = PERIOD_GRID

    ohlc = (df["open"], df["high"], df["low"], df["close"])

    # Type 1: crossover rules (Rules 1-9), two-period params
    type1_rules = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9]
    type1_params = []
    for rule in type1_rules:
        best_score = -1
        best_param = (periods[0], periods[0])
        for i in range(len(periods)):
            for j in range(i, len(periods)):
                param = (periods[i], periods[j])
                score = rule(param, ohlc)[0]
                if score > best_score:
                    best_score = score
                    best_param = param
        type1_params.append(best_param)

    # Type 2: indicator vs threshold (Rules 10-11)
    type2_rules = [rule_10, rule_11]
    type2_limits = [RSI_LIMITS, CCI_LIMITS]
    type2_params = []
    for rule, limits in zip(type2_rules, type2_limits):
        best_score = -1
        best_param = (periods[0], limits[0])
        for period in periods:
            for threshold in limits:
                param = (period, threshold)
                score = rule(param, ohlc)[0]
                if score > best_score:
                    best_score = score
                    best_param = param
        type2_params.append(best_param)

    # Type 3: oscillator with upper/lower bands (Rules 12-13)
    type3_rules = [rule_12, rule_13]
    type3_limits = [RSI_LIMITS, CCI_LIMITS]
    type3_params = []
    for rule, limits in zip(type3_rules, type3_limits):
        n = len(limits)
        best_score = -1
        best_param = (periods[0], limits[-1], limits[0])
        for period in periods:
            for lb_idx in range(n - 1):
                for ub_idx in range(lb_idx + 1, n):
                    param = (period, limits[ub_idx], limits[lb_idx])
                    score = rule(param, ohlc)[0]
                    if score > best_score:
                        best_score = score
                        best_param = param
        type3_params.append(best_param)

    # Type 4: channel mean-reversion (Rules 14-16), single period
    type4_rules = [rule_14, rule_15, rule_16]
    type4_params = []
    for rule in type4_rules:
        best_score = -1
        best_param = periods[0]
        for period in periods:
            score = rule(period, ohlc)[0]
            if score > best_score:
                best_score = score
                best_param = period
        type4_params.append(best_param)

    return type1_params + type2_params + type3_params + type4_params


def get_rule_features(
    df: pd.DataFrame,
    rule_params: list[tuple],
) -> pd.DataFrame:
    """
    Generate the trading rule feature matrix for GA consumption.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with lowercase columns.
    rule_params : list[tuple]
        16 parameter tuples from train_rule_params().

    Returns
    -------
    pd.DataFrame
        Columns: logr, rule_1, rule_2, ..., rule_16.
        NaN rows dropped.
    """
    ohlc = (df["open"], df["high"], df["low"], df["close"])
    logr = np.log(df["close"] / df["close"].shift(1))

    features = pd.DataFrame({"logr": logr}, index=df.index)
    for i, rule in enumerate(ALL_RULES):
        _, signal = rule(rule_params[i], ohlc)
        features[f"rule_{i + 1}"] = signal

    features.dropna(inplace=True)
    return features
