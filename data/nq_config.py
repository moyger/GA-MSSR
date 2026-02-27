"""NQ E-mini Nasdaq 100 futures contract configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NQConfig:
    """Immutable configuration for NQ E-mini futures."""
    symbol: str = "NQ"
    tick_size: float = 0.25
    tick_value: float = 5.00         # dollars per tick
    point_value: float = 20.00       # tick_value / tick_size = $20 per point
    commission_rt: float = 2.40      # round-turn commission in dollars
    hard_stop_daily: float = 400.0   # max daily loss per contract
    max_drawdown_pct: float = 0.15   # 15% max drawdown constraint for GA


NQ = NQConfig()
