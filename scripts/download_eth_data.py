#!/usr/bin/env python3
"""
Download ETHUSDT 1-minute historical data from Binance public API.

No authentication required. Downloads ~8 months of 1-min klines.

Usage:
    .venv/bin/python scripts/download_eth_data.py
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

SYMBOL = "ETHUSDT"
INTERVAL = "1m"
LIMIT = 1000  # max per request
OUTPUT = "data/futures/ETHUSD_1min.csv"

# Binance klines endpoint (public, no auth)
BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Fetch klines from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"Downloading {SYMBOL} 1-minute data from Binance...")

    # Go back ~8 months
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=240)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    chunk = 0

    while current_start < end_ms:
        chunk += 1
        try:
            klines = fetch_klines(SYMBOL, INTERVAL, current_start, end_ms, LIMIT)
        except Exception as e:
            print(f"  Error: {e}")
            break

        if not klines:
            break

        all_klines.extend(klines)
        # Next start = last kline close time + 1ms
        current_start = klines[-1][6] + 1

        if chunk % 50 == 0:
            dt = datetime.fromtimestamp(klines[0][0] / 1000)
            print(f"  Chunk {chunk}: {dt} ({len(all_klines):,} bars)")

        # Rate limit: Binance allows 1200 requests/min
        time.sleep(0.1)

    if not all_klines:
        print("ERROR: No data fetched.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    # Format
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Drop duplicates
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} bars to {OUTPUT}")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
