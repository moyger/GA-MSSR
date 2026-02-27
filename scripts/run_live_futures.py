#!/usr/bin/env python3
"""
GA-MSSR prop firm trading bot (IBKR data + TradersPost execution).

Runs the same GA-MSSR signal engine as the personal IBKR bot,
but executes via TradersPost webhooks for prop firm accounts
(FundedNext, MyFundedFutures, etc.). Market data comes from IBKR
(read-only connection).

Each prop firm account uses its own .env file with separate webhook,
risk limits, and state directory. See .env.propfirm.example for template.

Usage:
    # FundedNext account
    .venv/bin/python scripts/run_live_futures.py --env .env.propfirm.fundednext

    # MyFundedFutures account
    .venv/bin/python scripts/run_live_futures.py --env .env.propfirm.myfundedfutures

    # Default (uses .env.futures)
    .venv/bin/python scripts/run_live_futures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from live_futures.config import load_futures_config
from live_futures.bot import LiveFuturesBot


def main():
    parser = argparse.ArgumentParser(
        description="GA-MSSR Prop Firm Trading Bot (IBKR data + TradersPost execution)"
    )
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")
    args = parser.parse_args()

    config = load_futures_config(env_path=args.env)

    # Warn (but don't block) if no TradersPost webhook
    if not config.traderspost_webhook_url:
        print("WARNING: TRADERSPOST_WEBHOOK_URL is not set")
        print("  Orders will be logged but NOT sent to TradersPost")
        print("  Set up a webhook at https://traderspost.io")
        print()

    print(f"IBKR:       {config.ibkr_host}:{config.ibkr_port} (clientId={config.ibkr_client_id})")
    print(f"Contract:   {config.contract_root}")
    print(f"Trade size: {config.trade_size} contracts")
    print(f"FN account: ${config.fn_account_size:,.0f}")
    print(f"FN daily:   ${config.fn_daily_loss_limit:,.0f}")
    print(f"Webhook:    {'configured' if config.traderspost_webhook_url else 'NOT SET'}")
    print()

    bot = LiveFuturesBot(config)
    bot.start()


if __name__ == "__main__":
    main()
