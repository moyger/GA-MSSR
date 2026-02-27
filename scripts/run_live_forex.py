#!/usr/bin/env python3
"""
GA-MSSR live forex trading bot (IBKR paper/live via ib_insync).

Runs the GA-MSSR signal engine on 15-min forex bars with FTMO-style
risk rules ($500 daily limit, 10% max total loss on $100K account).

Requires TWS or IB Gateway running with API enabled.

Usage:
    # Default (AUDUSD, loads .env.forex)
    .venv/bin/python scripts/run_live_forex.py

    # Specific pair
    .venv/bin/python scripts/run_live_forex.py --pair EURUSD

    # Custom env file
    .venv/bin/python scripts/run_live_forex.py --env .env.forex.eurusd
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from live_forex.config import load_forex_config, LiveForexConfig
from live_forex.bot import LiveForexBot


def main():
    parser = argparse.ArgumentParser(
        description="GA-MSSR Forex Trading Bot (IBKR via ib_insync)"
    )
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")
    parser.add_argument("--pair", type=str, default=None, help="Forex pair (e.g. EURUSD, AUDUSD)")
    args = parser.parse_args()

    config = load_forex_config(env_path=args.env)

    # Override pair from CLI if provided
    if args.pair:
        config = LiveForexConfig(
            ibkr_host=config.ibkr_host,
            ibkr_port=config.ibkr_port,
            ibkr_client_id=config.ibkr_client_id,
            ibkr_account=config.ibkr_account,
            forex_pair=args.pair.upper(),
            order_units=config.order_units,
            timeframe=config.timeframe,
            timeframe_minutes=config.timeframe_minutes,
            warmup_bars=config.warmup_bars,
            denoise=config.denoise,
            wavelet=config.wavelet,
            position_threshold=config.position_threshold,
            train_days=config.train_days,
            bars_per_day=config.bars_per_day,
            rule_param_periods=config.rule_param_periods,
            ga_sol_per_pop=config.ga_sol_per_pop,
            ga_num_generations=config.ga_num_generations,
            ga_random_seed=config.ga_random_seed,
            retrain_hour_utc=config.retrain_hour_utc,
            ftmo_account_size=config.ftmo_account_size,
            ftmo_daily_loss_limit=config.ftmo_daily_loss_limit,
            ftmo_max_total_loss=config.ftmo_max_total_loss,
            ftmo_profit_target=config.ftmo_profit_target,
            log_dir=config.log_dir,
            state_dir=config.state_dir,
            slack_webhook_url=config.slack_webhook_url,
            flatten_before_weekend_min=config.flatten_before_weekend_min,
        )

    units = config.order_units
    mini_lots = units // 10_000
    std_lots = units // 100_000

    print(f"Pair:        {config.forex_pair}")
    print(f"Trade size:  {units:,} units ({mini_lots} mini / {std_lots} std lots)")
    print(f"IBKR:        {config.ibkr_host}:{config.ibkr_port} (client {config.ibkr_client_id})")
    print(f"Account:     ${config.ftmo_account_size:,.0f}")
    print(f"Daily stop:  ${config.ftmo_daily_loss_limit:,.0f}")
    print(f"Max loss:    ${config.ftmo_max_total_loss:,.0f}")
    print(f"Threshold:   {config.position_threshold:.2f}")
    print()

    bot = LiveForexBot(config)
    bot.start()


if __name__ == "__main__":
    main()
