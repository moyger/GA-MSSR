"""Live forex trading configuration.

Loads from .env.forex. Uses ib_insync for IBKR paper/live trading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LiveForexConfig:
    """Immutable configuration for the forex bot."""

    # --- IBKR ---
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497   # TWS paper=7497, live=7496; Gateway paper=4002, live=4001
    ibkr_client_id: int = 2  # Use 2 to avoid conflict with futures bot (client 1)
    ibkr_account: str = ""

    # --- Pair ---
    forex_pair: str = "AUDUSD"   # e.g. "EURUSD", "AUDUSD"

    # --- Position sizing ---
    order_units: int = 500_000   # 500K = 50 mini lots = 5 standard lots ($50/pip)

    # --- Strategy ---
    timeframe: str = "15m"
    timeframe_minutes: int = 15
    warmup_bars: int = 200
    denoise: bool = True
    wavelet: str = "db4"
    position_threshold: float = 0.50  # best from backtest threshold sweep

    # --- ER Filter ---
    er_period: int = 20
    er_threshold: float = 0.0             # Legacy simple threshold (0 = disabled)
    er_enter_threshold: float = 0.45      # Hysteresis: ER must exceed to enter
    er_exit_threshold: float = 0.25       # Hysteresis: ER must drop below to exit
    min_hold_bars: int = 3                # Minimum bars before allowing position flip

    # --- Training ---
    train_days: int = 20
    bars_per_day: int = 96     # forex 24h: 96 fifteen-min bars per day
    rule_param_periods: tuple[int, ...] = (3, 7, 15, 27)
    ga_sol_per_pop: int = 20
    ga_num_generations: int = 200
    ga_random_seed: Optional[int] = 42
    retrain_hour_utc: int = 0

    # --- FTMO Rules ($100K 2-Step Challenge) ---
    ftmo_account_size: float = 100_000.0
    ftmo_daily_loss_limit: float = 500.0     # custom tight daily stop
    ftmo_max_total_loss: float = 10_000.0    # 10% of account
    ftmo_profit_target: float = 10_000.0     # 10% challenge target

    # --- Paths ---
    log_dir: str = "logs"
    state_dir: str = "live_forex/state"

    # --- Notifications ---
    slack_webhook_url: str = ""

    # --- Order Execution (limit order re-pricing) ---
    reprice_max_attempts: int = 6          # 6 × 10s = 60s max wait
    reprice_timeout_sec: float = 10.0      # seconds per attempt before re-pricing
    reprice_poll_interval_sec: float = 0.5 # poll frequency within each attempt

    # --- Weekend ---
    flatten_before_weekend_min: int = 30


def load_forex_config(env_path: Optional[str] = None) -> LiveForexConfig:
    """Load config from environment variables / .env.forex file."""
    from dotenv import load_dotenv

    if env_path:
        load_dotenv(env_path)
    else:
        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env.forex")

    return LiveForexConfig(
        # IBKR
        ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.environ.get("IBKR_PORT", "7497")),
        ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "2")),
        ibkr_account=os.environ.get("IBKR_ACCOUNT", ""),
        # Pair
        forex_pair=os.environ.get("FOREX_PAIR", "AUDUSD"),
        # Position sizing
        order_units=int(os.environ.get("ORDER_UNITS", "500000")),
        # Strategy
        position_threshold=float(os.environ.get("POSITION_THRESHOLD", "0.50")),
        # FTMO
        ftmo_account_size=float(os.environ.get("FTMO_ACCOUNT_SIZE", "100000")),
        ftmo_daily_loss_limit=float(os.environ.get("FTMO_DAILY_LOSS_LIMIT", "500")),
        ftmo_max_total_loss=float(os.environ.get("FTMO_MAX_TOTAL_LOSS", "10000")),
        ftmo_profit_target=float(os.environ.get("FTMO_PROFIT_TARGET", "10000")),
        # Order execution
        reprice_max_attempts=int(os.environ.get("REPRICE_MAX_ATTEMPTS", "6")),
        reprice_timeout_sec=float(os.environ.get("REPRICE_TIMEOUT_SEC", "10.0")),
        reprice_poll_interval_sec=float(os.environ.get("REPRICE_POLL_INTERVAL_SEC", "0.5")),
        # State (unique per pair for multi-pair support)
        state_dir=os.environ.get("STATE_DIR", "live_forex/state"),
        # Notifications
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
    )
