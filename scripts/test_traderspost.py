#!/usr/bin/env python3
"""Test TradersPost webhook connectivity.

Sends test signals to TradersPost and prints the response.
Use this to verify your strategy config before running the full bot.

Usage:
    .venv/bin/python scripts/test_traderspost.py --env .env.propfirm.fundednext
    .venv/bin/python scripts/test_traderspost.py --env .env.propfirm.fundednext --action buy
    .venv/bin/python scripts/test_traderspost.py --env .env.propfirm.fundednext --action exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def send_webhook(url: str, payload: dict) -> None:
    """Send a webhook and print the full response."""
    print(f"\n→ POST {url[:60]}...")
    print(f"  Payload: {json.dumps(payload, indent=2)}")

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        print(f"  Status:  {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")

        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                print("  ✓ SUCCESS")
            else:
                print(f"  ✗ REJECTED: {data}")
        else:
            print(f"  ✗ HTTP ERROR {resp.status_code}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test TradersPost webhook")
    parser.add_argument(
        "--env", required=True,
        help="Path to .env file (e.g., .env.propfirm.fundednext)",
    )
    parser.add_argument(
        "--action", default="buy",
        choices=["buy", "sell", "exit"],
        help="Action to send (default: buy)",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Override ticker (default: CONTRACT_ROOT from env)",
    )
    parser.add_argument(
        "--qty", type=int, default=None,
        help="Override quantity (default: TRADE_SIZE from env)",
    )
    args = parser.parse_args()

    # Load env
    env_path = Path(args.env)
    if not env_path.exists():
        print(f"Error: {env_path} not found")
        sys.exit(1)

    load_dotenv(env_path, override=True)

    webhook_url = os.environ.get("TRADERSPOST_WEBHOOK_URL", "")
    ticker = args.ticker or os.environ.get("CONTRACT_ROOT", "MNQ")
    qty = args.qty or int(os.environ.get("TRADE_SIZE", "1"))

    if not webhook_url:
        print("Error: TRADERSPOST_WEBHOOK_URL not set")
        sys.exit(1)

    print("=" * 50)
    print("TradersPost Webhook Test")
    print("=" * 50)
    print(f"  Webhook: {webhook_url[:60]}...")
    print(f"  Ticker:  {ticker}")
    print(f"  Action:  {args.action}")
    print(f"  Qty:     {qty}")

    # Build payload
    payload = {"ticker": ticker, "action": args.action}

    if args.action in ("buy", "sell"):
        payload["sentiment"] = "bullish" if args.action == "buy" else "bearish"
        payload["quantity"] = qty

    send_webhook(webhook_url, payload)

    # Prompt to send exit if we just bought/sold
    if args.action in ("buy", "sell"):
        print("\n" + "-" * 50)
        resp = input("Send EXIT to close this test position? [y/N] ")
        if resp.strip().lower() == "y":
            exit_payload = {"ticker": ticker, "action": "exit"}
            send_webhook(webhook_url, exit_payload)

    print("\nDone.")


if __name__ == "__main__":
    main()
