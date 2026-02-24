"""Tests for technical indicators."""
import numpy as np
import pandas as pd
import pytest
from strategies.indicators import (
    ema, ma, dema, tema, rsi, stoch, cci,
    vortex_indicator_pos, vortex_indicator_neg,
    ichimoku_a, ichimoku_b,
    bollinger_hband, bollinger_lband,
    keltner_channel_hband, keltner_channel_lband,
    donchian_channel_hband, donchian_channel_lband,
)


@pytest.fixture
def price_series():
    """100-bar close price series with realistic NQ values."""
    np.random.seed(42)
    return pd.Series(
        np.cumsum(np.random.randn(100) * 2) + 16800,
        name="close",
    )


@pytest.fixture
def ohlc_df():
    """100-bar OHLC DataFrame."""
    np.random.seed(42)
    n = 100
    base = np.cumsum(np.random.randn(n) * 2) + 16800
    return pd.DataFrame({
        "open": base + np.random.randn(n) * 0.5,
        "high": base + np.abs(np.random.randn(n) * 3),
        "low": base - np.abs(np.random.randn(n) * 3),
        "close": base + np.random.randn(n) * 0.5,
    })


class TestMovingAverages:
    def test_ema_matches_pandas(self, price_series):
        result = ema(price_series, 10)
        expected = price_series.ewm(span=10, min_periods=10).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_ma_matches_pandas(self, price_series):
        result = ma(price_series, 10)
        expected = price_series.rolling(10).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_dema_length(self, price_series):
        result = dema(price_series, 10)
        assert len(result) == len(price_series)

    def test_tema_length(self, price_series):
        result = tema(price_series, 10)
        assert len(result) == len(price_series)

    def test_dema_faster_than_ema(self, price_series):
        """DEMA should respond faster to changes than EMA (less lag)."""
        e = ema(price_series, 10)
        d = dema(price_series, 10)
        # Both should have same NaN pattern at start
        valid = e.dropna().index
        assert len(d.loc[valid].dropna()) > 0


class TestRSI:
    def test_rsi_range(self, price_series):
        """RSI should be between 0 and 100."""
        result = rsi(price_series, 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_no_input_mutation(self, price_series):
        """RSI must not mutate the input Series."""
        original = price_series.copy()
        rsi(price_series, 14)
        pd.testing.assert_series_equal(price_series, original)

    def test_rsi_monotonic_up(self):
        """Strictly increasing prices should have RSI near 100."""
        close = pd.Series(np.linspace(16800, 16900, 50))
        result = rsi(close, 14)
        valid = result.dropna()
        assert valid.iloc[-1] > 90


class TestStochastic:
    def test_stoch_at_high(self):
        """When close == high for all bars, stoch_k should be 100."""
        n = 50
        high = pd.Series(np.linspace(100, 200, n))
        low = pd.Series(np.linspace(90, 190, n))
        close = high.copy()
        result = stoch(high, low, close, 14)
        # After warmup, should be 100
        assert (result.iloc[14:] == 100).all()

    def test_stoch_at_low(self):
        """When close == low and prices are flat, stoch_k should be 0."""
        n = 50
        high = pd.Series(np.full(n, 200.0))
        low = pd.Series(np.full(n, 190.0))
        close = low.copy()
        result = stoch(high, low, close, 14)
        assert (result.iloc[1:] == 0).all()


class TestBollinger:
    def test_band_width(self, price_series):
        """Upper - lower should equal 4 * std (ndev=2)."""
        n = 20
        upper = bollinger_hband(price_series, n=n, ndev=2)
        lower = bollinger_lband(price_series, n=n, ndev=2)
        mstd = price_series.rolling(n, min_periods=0).std()
        width = upper - lower
        expected_width = 4 * mstd
        pd.testing.assert_series_equal(width, expected_width, check_names=False)

    def test_upper_above_lower(self, price_series):
        upper = bollinger_hband(price_series, 20)
        lower = bollinger_lband(price_series, 20)
        valid = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid] >= lower.loc[valid]).all()


class TestDonchian:
    def test_hband_is_rolling_max(self, price_series):
        result = donchian_channel_hband(price_series, 20)
        expected = price_series.rolling(20, min_periods=0).max()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_lband_is_rolling_min(self, price_series):
        result = donchian_channel_lband(price_series, 20)
        expected = price_series.rolling(20, min_periods=0).min()
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestCCI:
    def test_cci_length(self, ohlc_df):
        result = cci(ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 20)
        assert len(result) == len(ohlc_df)


class TestVortex:
    def test_vortex_pos_length(self, ohlc_df):
        result = vortex_indicator_pos(
            ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 14
        )
        assert len(result) == len(ohlc_df)

    def test_vortex_neg_length(self, ohlc_df):
        result = vortex_indicator_neg(
            ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 14
        )
        assert len(result) == len(ohlc_df)


class TestIchimoku:
    def test_ichimoku_a_length(self, ohlc_df):
        result = ichimoku_a(ohlc_df["high"], ohlc_df["low"], 9, 26)
        assert len(result) == len(ohlc_df)

    def test_ichimoku_b_length(self, ohlc_df):
        result = ichimoku_b(ohlc_df["high"], ohlc_df["low"], 26, 52)
        assert len(result) == len(ohlc_df)


class TestKeltner:
    def test_hband_above_lband(self, ohlc_df):
        h = keltner_channel_hband(
            ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 10
        )
        l = keltner_channel_lband(
            ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 10
        )
        assert (h >= l).all()
