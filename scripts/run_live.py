#!/usr/bin/env python3
"""
Entry point for the GA-MSSR Bybit live trading bot.

Usage:
    .venv/bin/python scripts/run_live.py
    .venv/bin/python scripts/run_live.py --testnet
    .venv/bin/python scripts/run_live.py --mainnet
    .venv/bin/python scripts/run_live.py --env /path/to/.env

Environment variables (or .env file):
    BYBIT_API_KEY       - Bybit API key
    BYBIT_API_SECRET    - Bybit API secret
    BYBIT_TESTNET       - "true" (default) or "false"
    TRADE_SYMBOL        - ccxt symbol (default: "ETH/USDT:USDT")
    TRADE_SIZE          - position size in ETH (default: "0.01")
    LEVERAGE            - leverage multiplier (default: "5")
    MAX_DAILY_LOSS_USD  - daily loss limit (default: "50.0")
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.config import load_config
from live.bot import LiveBot


def main():
    parser = argparse.ArgumentParser(description="GA-MSSR Bybit Live Trading Bot")
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")
    parser.add_argument("--testnet", action="store_true", help="Force testnet mode")
    parser.add_argument(
        "--mainnet", action="store_true",
        help="Force mainnet mode (real money, use with caution)",
    )
    args = parser.parse_args()

    config = load_config(env_path=args.env)

    # Override testnet flag from CLI
    if args.mainnet:
        from dataclasses import replace
        config = replace(config, testnet=False)
    elif args.testnet:
        from dataclasses import replace
        config = replace(config, testnet=True)

    # Require API keys
    if not config.api_key or not config.api_secret:
        print("ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set")
        print("  Set them in .env file or as environment variables")
        print("  See .env.example for the template")
        sys.exit(1)

    # Mainnet safety confirmation
    # Skip interactive prompt when running under launchd (no TTY)
    if not config.testnet and sys.stdin.isatty():
        print("=" * 50)
        print("  WARNING: MAINNET MODE - REAL MONEY")
        print(f"  Symbol:     {config.symbol}")
        print(f"  Trade size: {config.trade_size}")
        print(f"  Leverage:   {config.leverage}x")
        print("=" * 50)
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    bot = LiveBot(config)
    bot.start()


if __name__ == "__main__":
    main()
