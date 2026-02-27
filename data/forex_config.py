"""Forex spot pair configuration for mini lot (10K units) backtesting."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ForexConfig:
    """Immutable configuration for a forex pair (mini lot = 10K units)."""
    symbol: str = ""
    pip_size: float = 0.0001        # 1 pip = 0.0001 for USD-quoted pairs
    tick_size: float = 0.00001      # pipette (5-decimal quoting)
    lot_size: float = 10_000        # mini lot = 10K units
    pip_value: float = 1.00         # $1.00/pip for USD-quoted pairs at mini lot
    point_value: float = 10_000.0   # lot_size — PnL = price_change * point_value
    commission_rt: float = 0.50     # round-turn commission estimate (ECN)
    spread_cost_pips: float = 1.0   # average spread in pips (informational)
    hard_stop_daily: float = 50.0   # max daily loss per mini lot
    max_drawdown_pct: float = 0.15  # 15% max drawdown constraint for GA


EURUSD = ForexConfig(
    symbol="EURUSD",
    spread_cost_pips=0.8,
)

AUDUSD = ForexConfig(
    symbol="AUDUSD",
    spread_cost_pips=1.5,
)
