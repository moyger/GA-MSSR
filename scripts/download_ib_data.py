#!/usr/bin/env python3
"""
Download NQ 1-minute historical data from Interactive Brokers.

Downloads from the current front-month NQ contract, looping backwards
in 5-day chunks to get up to ~6 months of 1-minute data.

Prerequisites:
    - TWS or IB Gateway running with API enabled on port 7497
    - CME market data subscription active

Usage:
    .venv/bin/python scripts/download_ib_data.py
"""
import asyncio
import time
import sys

# Python 3.14 removed auto-creation of event loops; ib_insync needs one at import time
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import *
import pandas as pd
from datetime import datetime

# Connect to TWS/Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print("Connected to TWS")

# Use the front-month NQ contract (NQH6 = March 2026)
contract = Future('NQ', '20260320', exchange='CME')
ib.qualifyContracts(contract)
print(f"Qualified contract: {contract}")

# Download 1-min bars in 5-day chunks, looping backwards
bars_all = []
end = ''  # empty string = now
for i in range(60):  # ~60 chunks × 5 days ≈ 300 days
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr='5 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
        )
    except Exception as e:
        print(f"  Request error: {e}")
        break

    if not bars:
        print("No more data available.")
        break

    bars_all = bars + bars_all
    # Format end datetime for next request
    end_dt = bars[0].date
    if hasattr(end_dt, 'strftime'):
        end = end_dt.strftime('%Y%m%d-%H:%M:%S')
    else:
        end = str(end_dt)
    print(f"  Chunk {i+1}: fetched back to {end_dt} ({len(bars_all):,} bars total)")

    # IB pacing: ~2s between requests to stay under rate limits
    time.sleep(2)

if not bars_all:
    print("ERROR: No bars fetched. Check market data subscriptions.")
    ib.disconnect()
    sys.exit(1)

# Convert to DataFrame and save
df = util.df(bars_all)
df = df.rename(columns={'date': 'timestamp'})
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
# Drop duplicates from overlapping chunks
df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

output_path = 'data/futures/NQ_1min_IB.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df):,} bars to {output_path}")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

ib.disconnect()
print("Disconnected.")
