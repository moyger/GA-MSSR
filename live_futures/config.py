"""
Live futures trading configuration.

Loads credentials and parameters from .env.futures file.
Completely independent from live/config.py (Bybit).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LiveFuturesConfig:
    """Immutable configuration for the futures bot."""

    # --- IBKR ---
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497  # TWS paper=7497, live=7496; Gateway paper=4002, live=4001
    ibkr_client_id: int = 1
    ibkr_account: str = ""  # e.g. "DU123456" for paper, "U123456" for live

    # --- TradersPost (prop firm execution via run_live_futures.py) ---
    traderspost_webhook_url: str = ""

    # --- Contract ---
    contract_root: str = "MNQ"
    trade_size: int = 1  # number of contracts

    # --- Strategy ---
    timeframe: str = "15m"
    timeframe_minutes: int = 15
    warmup_bars: int = 200
    denoise: bool = True
    wavelet: str = "db4"
    position_threshold: float = 0.10

    # --- Training ---
    train_days: int = 20
    bars_per_day: int = 92  # 23 CME trading hours * 4 bars/hr
    rule_param_periods: tuple[int, ...] = (3, 7, 15, 27)
    ga_sol_per_pop: int = 20
    ga_num_generations: int = 200
    ga_random_seed: Optional[int] = 42
    retrain_hour_utc: int = 0

    # --- FundedNext $50K Rapid Challenge ---
    fn_account_size: float = 50000.0
    fn_daily_loss_limit: float = 500.0  # Our hard stop (actual FN limit: $1,000)
    fn_trailing_max_loss: float = 2000.0  # FN MLL: trailing EOD, locks at starting balance
    fn_profit_target: float = 3000.0
    fn_consistency_pct: float = 0.0  # No consistency rule in Rapid Challenge

    # --- ER Filter ---
    er_period: int = 14
    er_threshold: float = 0.40

    # --- Risk ---
    max_daily_loss_usd: float = 250.0
    max_position_contracts: int = 5
    stop_loss_points: float = 100.0  # Per-trade stop-loss in index points

    # --- Paths ---
    log_dir: str = "logs"
    state_dir: str = "live_futures/state"

    # --- Notifications ---
    slack_webhook_url: str = ""

    # --- Weekend ---
    flatten_before_weekend_min: int = 30


def load_futures_config(env_path: Optional[str] = None) -> LiveFuturesConfig:
    """Load config from environment variables / .env.futures file."""
    from dotenv import load_dotenv

    if env_path:
        load_dotenv(env_path)
    else:
        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env.futures")

    return LiveFuturesConfig(
        # IBKR
        ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.environ.get("IBKR_PORT", "7497")),
        ibkr_client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
        ibkr_account=os.environ.get("IBKR_ACCOUNT", ""),
        # TradersPost (prop firm execution)
        traderspost_webhook_url=os.environ.get("TRADERSPOST_WEBHOOK_URL", ""),
        # Contract
        contract_root=os.environ.get("CONTRACT_ROOT", "MNQ"),
        trade_size=int(os.environ.get("TRADE_SIZE", "1")),
        # FundedNext
        fn_account_size=float(os.environ.get("FN_ACCOUNT_SIZE", "50000")),
        fn_daily_loss_limit=float(os.environ.get("FN_DAILY_LOSS_LIMIT", "500")),
        fn_trailing_max_loss=float(os.environ.get("FN_TRAILING_MAX_LOSS", "2000")),
        fn_profit_target=float(os.environ.get("FN_PROFIT_TARGET", "3000")),
        fn_consistency_pct=float(os.environ.get("FN_CONSISTENCY_PCT", "0.0")),
        # Risk
        max_daily_loss_usd=float(os.environ.get("FN_DAILY_LOSS_LIMIT", "250")),
        max_position_contracts=int(os.environ.get("MAX_POSITION_CONTRACTS", "5")),
        stop_loss_points=float(os.environ.get("STOP_LOSS_POINTS", "100")),
        # Strategy
        position_threshold=float(os.environ.get("POSITION_THRESHOLD", "0.10")),
        # State (unique per account for multi-account support)
        state_dir=os.environ.get("STATE_DIR", "live_futures/state"),
        # Notifications
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
    )
