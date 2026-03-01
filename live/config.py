"""
Live trading configuration.

Loads API keys from environment variables (via .env file).
All trading parameters are centralized here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LiveConfig:
    """Immutable configuration for the live trading bot."""

    # --- Exchange ---
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    symbol: str = "ETH/USDT:USDT"

    # --- Strategy ---
    timeframe: str = "15m"
    timeframe_minutes: int = 15
    warmup_bars: int = 200
    denoise: bool = True
    wavelet: str = "db4"
    position_threshold: float = 0.10

    # --- Training ---
    train_days: int = 20
    rule_param_periods: tuple[int, ...] = (3, 7, 15, 27)
    ga_sol_per_pop: int = 20
    ga_num_generations: int = 200
    ga_random_seed: Optional[int] = 42
    retrain_hour_utc: int = 0

    # --- ER Filter ---
    er_period: int = 14
    er_threshold: float = 0.0             # Legacy simple threshold (0 = disabled)
    er_enter_threshold: float = 0.50      # Hysteresis: ER must exceed to enter
    er_exit_threshold: float = 0.30       # Hysteresis: ER must drop below to exit
    min_hold_bars: int = 3                # Minimum bars before allowing position flip

    # --- Position Sizing ---
    trade_size: float = 0.01
    leverage: int = 5

    # --- Risk Management ---
    max_daily_loss_usd: float = 50.0
    max_position_notional_usd: float = 5000.0

    # --- Order Execution (limit order re-pricing) ---
    reprice_max_attempts: int = 6          # 6 × 10s = 60s max wait
    reprice_timeout_sec: float = 10.0      # seconds per attempt before re-pricing
    reprice_poll_interval_sec: float = 0.5 # poll frequency within each attempt

    # --- Paths ---
    log_dir: str = "logs"
    state_dir: str = "live/state"

    # --- Notifications ---
    slack_webhook_url: str = ""

    # --- Misc ---
    heartbeat_interval_sec: int = 60


def load_config(env_path: Optional[str] = None) -> LiveConfig:
    """Load config from environment variables / .env file."""
    from dotenv import load_dotenv

    if env_path:
        load_dotenv(env_path)
    else:
        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")

    return LiveConfig(
        api_key=os.environ.get("BYBIT_API_KEY", ""),
        api_secret=os.environ.get("BYBIT_API_SECRET", ""),
        testnet=os.environ.get("BYBIT_TESTNET", "true").lower() == "true",
        symbol=os.environ.get("TRADE_SYMBOL", "ETH/USDT:USDT"),
        trade_size=float(os.environ.get("TRADE_SIZE", "0.01")),
        leverage=int(os.environ.get("LEVERAGE", "5")),
        max_daily_loss_usd=float(os.environ.get("MAX_DAILY_LOSS_USD", "50.0")),
        position_threshold=float(os.environ.get("POSITION_THRESHOLD", "0.10")),
        reprice_max_attempts=int(os.environ.get("REPRICE_MAX_ATTEMPTS", "6")),
        reprice_timeout_sec=float(os.environ.get("REPRICE_TIMEOUT_SEC", "10.0")),
        reprice_poll_interval_sec=float(os.environ.get("REPRICE_POLL_INTERVAL_SEC", "0.5")),
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
    )
