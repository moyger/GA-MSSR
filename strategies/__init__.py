"""Signal engine: Khushi 16-rule trading system + NautilusTrader strategy."""
from strategies.khushi_rules import (
    ALL_RULES,
    train_rule_params,
    get_rule_features,
)
from strategies.khushi_strategy import KhushiSignalEngine

__all__ = [
    "ALL_RULES",
    "train_rule_params",
    "get_rule_features",
    "KhushiSignalEngine",
]

# Conditionally export nautilus-dependent classes
try:
    from strategies.khushi_strategy import KhushiStrategy, KhushiStrategyConfig
    from strategies.backtest import create_nq_instrument, run_backtest

    __all__ += [
        "KhushiStrategy",
        "KhushiStrategyConfig",
        "create_nq_instrument",
        "run_backtest",
    ]
except ImportError:
    pass
