#!/usr/bin/env python3
"""
Live vs Backtest Reconciliation — ETHUSD.

Compares actual live trade log entries against what the model
would have predicted for the same time window.

Checks:
1. Are live trade directions matching backtest signals?
2. Is trade frequency similar?
3. Are we entering/exiting at the right bars?
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from strategies.khushi_rules import train_rule_params, get_rule_features

TRADE_LOG = "live/state/trade_log.jsonl"
DATA_FILE = "data/futures/ETHUSD_1min.csv"
TIMEFRAME = "15min"
BPD = 96
TRAIN_DAYS = 20
POSITION_THRESHOLD = 0.30


def load_trade_log(path):
    """Load JSONL trade log into a DataFrame."""
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main():
    print("=" * 70)
    print("LIVE vs BACKTEST RECONCILIATION — ETHUSD")
    print("=" * 70)

    # ---- Load Trade Log ----
    print(f"\n[1/4] Loading live trade log from {TRADE_LOG}...")
    if not Path(TRADE_LOG).exists():
        print(f"  ERROR: Trade log not found: {TRADE_LOG}")
        sys.exit(1)

    trades_df = load_trade_log(TRADE_LOG)
    print(f"  {len(trades_df)} trades from {trades_df['timestamp'].min()} "
          f"to {trades_df['timestamp'].max()}")

    # Filter to trades with fill data (post-fix)
    filled = trades_df[trades_df["avg_price"].notna() & (trades_df["avg_price"] > 0)]
    print(f"  {len(filled)} filled trades (with price data)")

    if len(filled) == 0:
        print("  No filled trades to reconcile.")
        sys.exit(0)

    # Trade summary
    buys = filled[filled["side"] == "buy"]
    sells = filled[filled["side"] == "sell"]
    print(f"  Buys: {len(buys)}  Sells: {len(sells)}")

    live_start = filled["timestamp"].min()
    live_end = filled["timestamp"].max()
    live_hours = (live_end - live_start).total_seconds() / 3600
    trades_per_day = len(filled) / max(live_hours / 24, 0.01)
    print(f"  Duration: {live_hours:.1f} hours ({live_hours/24:.1f} days)")
    print(f"  Live trades/day: {trades_per_day:.1f}")

    # ---- Load Market Data ----
    print(f"\n[2/4] Loading market data from {DATA_FILE}...")
    df = build_denoised_dataset(DATA_FILE, denoise_columns="close", timeframe=TIMEFRAME)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # Find the data window that covers the live period
    # We need TRAIN_DAYS before the live start for training
    train_size = BPD * TRAIN_DAYS

    # Find index of live_start in data
    idx = pd.DatetimeIndex(df.index)
    mask = idx >= (live_start - pd.Timedelta(hours=1))  # small buffer
    if mask.sum() == 0:
        print(f"  ERROR: No data covering live period starting {live_start}")
        sys.exit(1)

    live_start_loc = int(np.argmax(mask))

    if live_start_loc < train_size:
        print(f"  ERROR: Need {train_size} bars before live start, only have {live_start_loc}")
        sys.exit(1)

    # ---- Train Model on Pre-Live Window ----
    print(f"\n[3/4] Training model on {TRAIN_DAYS}-day window before live start...")
    train_df = df.iloc[live_start_loc - train_size : live_start_loc]
    print(f"  Training: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")

    rule_params = train_rule_params(train_df, periods=[3, 7, 15, 27])
    train_features = get_rule_features(train_df, rule_params)

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )
    ga = GAMSSR(ga_config)
    fit_result = ga.fit(train_features)
    print(f"  Training SSR: {fit_result.best_fitness:.6f}")

    # ---- Generate Backtest Signals for Live Period ----
    print(f"\n[4/4] Generating backtest signals for live period...")
    live_df = df.iloc[live_start_loc:]
    mask_end = live_df.index <= (live_end + pd.Timedelta(hours=1))
    live_df = live_df[mask_end]
    print(f"  Live window: {live_df.index[0]} to {live_df.index[-1]} ({len(live_df)} bars)")

    live_features = get_rule_features(live_df, rule_params)
    predictions = ga.predict(live_features)

    # Discretize positions
    raw_pos = predictions["position"].values
    disc_pos = np.where(raw_pos > POSITION_THRESHOLD, 1,
                        np.where(raw_pos < -POSITION_THRESHOLD, -1, 0))

    # Count backtest trades (position changes)
    bt_trades = np.sum(np.diff(disc_pos) != 0)
    bt_days = len(live_df) / BPD
    bt_trades_per_day = bt_trades / max(bt_days, 0.01)

    # Backtest return for the live window
    bt_return = float(predictions["port_return"].sum())
    bt_cum = np.cumsum(predictions["port_return"].values)
    bt_max_dd = float(np.min(bt_cum - np.maximum.accumulate(bt_cum)))

    flat_pct = (disc_pos == 0).sum() / len(disc_pos) * 100
    long_pct = (disc_pos == 1).sum() / len(disc_pos) * 100
    short_pct = (disc_pos == -1).sum() / len(disc_pos) * 100

    # ---- Bar-by-bar signal vs trade comparison ----
    print(f"\n{'=' * 70}")
    print("RECONCILIATION RESULTS")
    print(f"{'=' * 70}")

    print(f"\n--- Trade Frequency ---")
    print(f"  Live trades/day:     {trades_per_day:.1f}")
    print(f"  Backtest trades/day: {bt_trades_per_day:.1f}")
    freq_ratio = trades_per_day / max(bt_trades_per_day, 0.01)
    print(f"  Ratio (live/bt):     {freq_ratio:.2f}x")
    if freq_ratio > 2.0:
        print(f"  WARNING: Live trading {freq_ratio:.1f}x more than backtest!")
        print(f"  Possible causes: duplicate orders, position desync, or threshold mismatch")
    elif freq_ratio < 0.5:
        print(f"  WARNING: Live trading much less than backtest")
        print(f"  Possible causes: connectivity issues, order rejections")
    else:
        print(f"  OK: Trade frequency is within expected range")

    print(f"\n--- Backtest Performance (live window) ---")
    print(f"  Return:     {bt_return*100:+.2f}%")
    print(f"  Max DD:     {bt_max_dd*100:.2f}%")
    print(f"  Flat:       {flat_pct:.1f}%  Long: {long_pct:.1f}%  Short: {short_pct:.1f}%")
    print(f"  Trades:     {bt_trades} over {bt_days:.1f} days")

    # Match live trades to backtest signals
    print(f"\n--- Signal Alignment ---")
    matches = 0
    mismatches = 0
    unmatched = 0

    for _, trade in filled.iterrows():
        ts = trade["timestamp"]
        side = trade["side"]

        # Find nearest bar
        time_diffs = abs(predictions.index - ts)
        nearest_idx = time_diffs.argmin()
        nearest_bar = predictions.index[nearest_idx]
        time_diff_min = time_diffs[nearest_idx].total_seconds() / 60

        if time_diff_min > 20:  # More than 20 min from any bar
            unmatched += 1
            continue

        bt_pos = disc_pos[nearest_idx]

        if side == "buy" and bt_pos >= 0:
            matches += 1
        elif side == "sell" and bt_pos <= 0:
            matches += 1
        else:
            mismatches += 1

    total = matches + mismatches + unmatched
    match_rate = matches / max(matches + mismatches, 1) * 100

    print(f"  Matches:    {matches}/{total} ({match_rate:.0f}% of matched trades)")
    print(f"  Mismatches: {mismatches}/{total}")
    print(f"  Unmatched:  {unmatched}/{total} (no nearby bar)")

    # Final verdict
    print(f"\n--- Overall Verdict ---")
    issues = []
    if freq_ratio > 2.0:
        issues.append("overtrading vs backtest")
    if match_rate < 60:
        issues.append(f"low signal alignment ({match_rate:.0f}%)")
    if bt_return < 0:
        issues.append("backtest also negative for this period")

    if not issues:
        print(f"  PASS: Live execution aligns well with backtest expectations")
    else:
        print(f"  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()
    print("\nDone.")
