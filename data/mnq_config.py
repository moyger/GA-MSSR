"""MNQ Micro E-mini Nasdaq 100 futures contract configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MNQConfig:
    """Immutable configuration for MNQ Micro futures."""
    symbol: str = "MNQ"
    tick_size: float = 0.25
    tick_value: float = 0.50         # dollars per tick (1/10th of NQ)
    point_value: float = 2.00        # tick_value / tick_size = $2 per point
    commission_rt: float = 1.24      # round-turn commission in dollars
    hard_stop_daily: float = 1200.0  # FundedNext $50K daily loss limit
    max_drawdown_pct: float = 0.05   # 5% max drawdown ($2,500 / $50,000)


MNQ = MNQConfig()
