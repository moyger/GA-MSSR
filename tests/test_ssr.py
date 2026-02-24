"""Tests for Sharpe-Sterling Ratio calculator."""
import numpy as np
import pytest
from optimizers.ssr import SSR, SSRResult


@pytest.fixture
def ssr():
    return SSR()


class TestSSRCalculate:
    def test_known_values(self, ssr):
        """Hand-computed SSR from a small known array."""
        r = np.array([0.01, -0.005, 0.02, -0.003, 0.015])
        result = ssr.calculate(r)

        mean_r = np.mean(r)
        std_r = np.std(r, ddof=0)
        neg_sum = np.sum(r[r < 0])  # -0.008

        expected_ssr = mean_r / std_r / (-neg_sum)
        assert abs(result.ssr - expected_ssr) < 1e-10
        assert abs(result.mean_return - mean_r) < 1e-10
        assert abs(result.std_return - std_r) < 1e-10
        assert abs(result.negative_return_sum - neg_sum) < 1e-10

    def test_all_positive_returns(self, ssr):
        """All positive returns => neg_sum=0 => SSR=0 (degenerate)."""
        r = np.array([0.01, 0.02, 0.03])
        result = ssr.calculate(r)
        assert result.ssr == 0.0
        assert result.negative_return_sum == 0.0

    def test_all_negative_returns(self, ssr):
        """All negative returns => SSR should be negative."""
        r = np.array([-0.01, -0.02, -0.03])
        result = ssr.calculate(r)
        assert result.ssr < 0.0

    def test_empty_returns(self, ssr):
        result = ssr.calculate(np.array([]))
        assert result.ssr == 0.0

    def test_nan_handling(self, ssr):
        r = np.array([0.01, np.nan, -0.005, 0.02, np.nan])
        result = ssr.calculate(r)
        # Should compute on non-NaN values only
        clean = np.array([0.01, -0.005, 0.02])
        expected = ssr.calculate(clean)
        assert abs(result.ssr - expected.ssr) < 1e-10

    def test_max_drawdown(self, ssr):
        """Verify max drawdown for a known sequence."""
        r = np.array([0.10, 0.05, -0.15, -0.10, 0.20])
        result = ssr.calculate(r)
        # Cumulative: [0.10, 0.15, 0.00, -0.10, 0.10]
        # Running max: [0.10, 0.15, 0.15, 0.15, 0.15]
        # Drawdown:    [0.00, 0.00, -0.15, -0.25, -0.05]
        assert abs(result.max_drawdown - (-0.25)) < 1e-10

    def test_commission_reduces_ssr(self):
        ssr_no_comm = SSR(commission_per_trade=0.0)
        ssr_with_comm = SSR(commission_per_trade=2.40)
        r = np.array([0.01, -0.005, 0.02, -0.003, 0.015])

        result_no = ssr_no_comm.calculate(r, num_trades=5)
        result_with = ssr_with_comm.calculate(r, num_trades=5)
        assert result_with.total_return < result_no.total_return

    def test_total_return(self, ssr):
        r = np.array([0.01, -0.005, 0.02])
        result = ssr.calculate(r)
        assert abs(result.total_return - 0.025) < 1e-10


class TestSSRVectorized:
    def test_shape(self, ssr):
        matrix = np.random.randn(8, 100) * 0.01
        result = ssr.calculate_vectorized(matrix)
        assert result.shape == (8,)

    def test_matches_reference_formula(self, ssr):
        """Verify vectorized matches reference ga.py cal_pop_fitness."""
        np.random.seed(42)
        matrix = np.random.randn(4, 50) * 0.01

        # Reference formula from ga.py line 60
        mean_r = np.mean(matrix, axis=1)
        std_r = np.std(matrix, axis=1)
        neg_sum = np.sum(matrix[matrix < 0])
        expected = mean_r / std_r / (-neg_sum)

        result = ssr.calculate_vectorized(matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_all_zero_returns(self, ssr):
        matrix = np.zeros((4, 50))
        result = ssr.calculate_vectorized(matrix)
        assert (result == 0.0).all()
