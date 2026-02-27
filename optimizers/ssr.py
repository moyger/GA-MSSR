"""
Sharpe-Sterling Ratio (SSR) calculator.

Implements the custom SSR fitness function from Dr. Khushi's GA-MSSR framework.
This combines Sharpe-style risk adjustment with a penalty based on the sum of
all negative returns (a drawdown proxy).

Reference: ga.py line 60 from zzzac/Rule-based-forex-trading-system:
    SSR = mean(port_r) / std(port_r) / (-sum(port_r[port_r < 0]))
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SSRResult:
    """Container for SSR calculation results."""
    ssr: float
    mean_return: float
    std_return: float
    negative_return_sum: float
    sharpe_component: float
    sterling_penalty: float
    num_trades: int
    max_drawdown: float
    total_return: float


class SSR:
    """
    Compute the Sharpe-Sterling Ratio from a returns series.

    SSR = mean(R) / std(R) / (-sum(R[R < 0]))

    Parameters
    ----------
    commission_per_trade : float
        Round-turn commission in the same units as PnL (default: 0.0).
    annualization_factor : float
        Factor to annualize the Sharpe component (default: 1.0).
    """

    def __init__(
        self,
        commission_per_trade: float = 0.0,
        annualization_factor: float = 1.0,
    ):
        self.commission_per_trade = commission_per_trade
        self.annualization_factor = annualization_factor

    def calculate(
        self,
        returns: np.ndarray | pd.Series,
        num_trades: Optional[int] = None,
    ) -> SSRResult:
        """
        Calculate SSR from a returns or PnL series.

        Parameters
        ----------
        returns : array-like
            Per-bar returns or PnL values. NaN values are dropped.
        num_trades : int, optional
            Number of round-turn trades (for commission deduction).

        Returns
        -------
        SSRResult
        """
        r = np.asarray(returns, dtype=np.float64)
        r = r[~np.isnan(r)]

        if len(r) == 0:
            return self._empty_result()

        # Apply commissions
        if num_trades is not None and num_trades > 0 and self.commission_per_trade > 0:
            total_commission = num_trades * self.commission_per_trade
            r = r.copy()
            r -= total_commission / len(r)

        mean_r = np.mean(r)
        std_r = np.std(r, ddof=0)  # population std, matching reference
        neg_sum = np.sum(r[r < 0])

        if std_r == 0.0 or neg_sum == 0.0:
            ssr_value = 0.0
            sharpe_component = 0.0
            sterling_penalty = 0.0
        else:
            sharpe_component = mean_r / std_r
            sterling_penalty = -neg_sum
            ssr_value = sharpe_component / sterling_penalty

        # Max drawdown
        cumulative = np.cumsum(r)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return SSRResult(
            ssr=ssr_value,
            mean_return=mean_r,
            std_return=std_r,
            negative_return_sum=neg_sum,
            sharpe_component=sharpe_component * self.annualization_factor,
            sterling_penalty=sterling_penalty,
            num_trades=num_trades or 0,
            max_drawdown=max_dd,
            total_return=float(np.sum(r)),
        )

    def calculate_vectorized(
        self,
        returns_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized SSR for a population of chromosomes (GA fitness).

        Mirrors reference ga.py cal_pop_fitness exactly:
            SSR = mean(port_r, axis=1) / std(port_r, axis=1) / (-sum(port_r[port_r<0]))

        Note: the negative return sum is across ALL chromosomes (single scalar),
        making SSR population-relative. This matches the reference behavior.

        Parameters
        ----------
        returns_matrix : np.ndarray
            Shape (population_size, num_bars).

        Returns
        -------
        np.ndarray
            Shape (population_size,) SSR values.
        """
        r = returns_matrix.astype(np.float64)
        mean_r = np.mean(r, axis=1)
        std_r = np.std(r, axis=1)
        neg_sum = np.sum(r[r < 0])

        with np.errstate(divide="ignore", invalid="ignore"):
            ssr = mean_r / std_r / (-neg_sum)

        ssr = np.where(np.isfinite(ssr), ssr, 0.0)
        return ssr

    def _empty_result(self) -> SSRResult:
        return SSRResult(
            ssr=0.0, mean_return=0.0, std_return=0.0,
            negative_return_sum=0.0, sharpe_component=0.0,
            sterling_penalty=0.0, num_trades=0,
            max_drawdown=0.0, total_return=0.0,
        )
