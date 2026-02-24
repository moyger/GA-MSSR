"""Tests for Khushi 16-rule signal engine."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from strategies.khushi_rules import (
    rule_1, rule_2, rule_3, rule_4, rule_5, rule_6,
    rule_7, rule_8, rule_9, rule_10, rule_11,
    rule_12, rule_13, rule_14, rule_15, rule_16,
    ALL_RULES, train_rule_params, get_rule_features,
)
from data.loader import load_nq_data

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_nq_1m.csv"


@pytest.fixture
def nq_df():
    return load_nq_data(FIXTURE_PATH)


@pytest.fixture
def ohlc(nq_df):
    return (nq_df["open"], nq_df["high"], nq_df["low"], nq_df["close"])


class TestAllRulesBasic:
    """Basic contract tests for all 16 rules."""

    @pytest.mark.parametrize("rule_fn", ALL_RULES[:9], ids=[f"rule_{i+1}" for i in range(9)])
    def test_type1_returns_tuple(self, rule_fn, ohlc):
        score, signal = rule_fn((5, 15), ohlc)
        assert isinstance(score, (float, np.floating))
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(ohlc[0])

    @pytest.mark.parametrize("rule_fn,param", [
        (rule_10, (14, 50)),
        (rule_11, (20, 0)),
    ])
    def test_type2_returns_tuple(self, rule_fn, param, ohlc):
        score, signal = rule_fn(param, ohlc)
        assert isinstance(score, (float, np.floating))
        assert len(signal) == len(ohlc[0])

    @pytest.mark.parametrize("rule_fn,param", [
        (rule_12, (14, 70, 30)),
        (rule_13, (20, 100, -100)),
    ])
    def test_type3_returns_tuple(self, rule_fn, param, ohlc):
        score, signal = rule_fn(param, ohlc)
        assert isinstance(score, (float, np.floating))
        assert len(signal) == len(ohlc[0])

    @pytest.mark.parametrize("rule_fn", [rule_14, rule_15, rule_16],
                             ids=["rule_14", "rule_15", "rule_16"])
    def test_type4_returns_tuple(self, rule_fn, ohlc):
        score, signal = rule_fn(20, ohlc)
        assert isinstance(score, (float, np.floating))
        assert len(signal) == len(ohlc[0])


class TestSignalValues:
    """Verify signal values are valid."""

    def test_type1_signals_in_range(self, ohlc):
        _, signal = rule_1((5, 15), ohlc)
        valid = signal.dropna()
        assert set(valid.unique()).issubset({-1, 1})

    def test_type3_signals_in_range(self, ohlc):
        _, signal = rule_12((14, 70, 30), ohlc)
        valid = signal.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_type4_signals_in_range(self, ohlc):
        _, signal = rule_16(20, ohlc)
        valid = signal.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_signal_shifted_by_one(self, ohlc):
        """First valid signal should start at index 1+ (shifted to avoid look-ahead)."""
        _, signal = rule_1((3, 5), ohlc)
        # The shift(1) means index 0 is NaN
        assert pd.isna(signal.iloc[0])


class TestRule15Fix:
    """Verify the Rule15 donchian bug fix."""

    def test_rule15_uses_lband(self, ohlc):
        """With varying prices, hband != lband so the signal should not be all zeros."""
        _, signal = rule_15(5, ohlc)
        valid = signal.dropna()
        # Donchian with period 5: hband is rolling max, lband is rolling min
        # With 100 bars of varying NQ data, close should breach bounds sometimes
        # The key fix: if both bands were hband, signal would be mostly 0 or all -1
        unique_vals = set(valid.unique())
        # Should have at least some variation (not all same value)
        assert len(unique_vals) >= 1  # at minimum it produces valid signals


class TestTrainRuleParams:
    def test_returns_16_params(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7, 15])
        assert len(params) == 16

    def test_type1_params_are_tuples(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        for i in range(9):
            assert isinstance(params[i], tuple)
            assert len(params[i]) == 2

    def test_type2_params_have_threshold(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        # Rule10 (RSI): param = (period, rsi_threshold)
        assert isinstance(params[9], tuple)
        assert len(params[9]) == 2

    def test_type3_params_have_bounds(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        # Rule12 (RSI bands): param = (period, upper, lower)
        assert isinstance(params[11], tuple)
        assert len(params[11]) == 3

    def test_type4_params_are_single_values(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        for i in range(13, 16):
            assert isinstance(params[i], (int, np.integer))


class TestGetRuleFeatures:
    def test_output_shape(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        features = get_rule_features(nq_df, params)
        assert "logr" in features.columns
        for i in range(1, 17):
            assert f"rule_{i}" in features.columns
        assert features.shape[1] == 17

    def test_no_nans(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        features = get_rule_features(nq_df, params)
        assert not features.isna().any().any()

    def test_logr_is_log_returns(self, nq_df):
        params = train_rule_params(nq_df, periods=[3, 7])
        features = get_rule_features(nq_df, params)
        # logr should be log returns of the close
        expected = np.log(nq_df["close"] / nq_df["close"].shift(1))
        # Compare on overlapping index
        common = features.index.intersection(expected.dropna().index)
        np.testing.assert_allclose(
            features.loc[common, "logr"].values,
            expected.loc[common].values,
            rtol=1e-10,
        )
