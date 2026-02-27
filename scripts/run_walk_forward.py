#!/usr/bin/env python3
"""
Walk-forward validation for GA-MSSR on NQ futures data.

Trains on a rolling window, tests on unseen data, rolls forward.
This gives realistic out-of-sample performance estimates.

Usage:
    .venv/bin/python scripts/run_walk_forward.py data/futures/NQ_1min_IB.csv
    .venv/bin/python scripts/run_walk_forward.py data/futures/NQ_1min_IB.csv 5min

Window sizes scale automatically with the timeframe:
    - Train: 20 trading days
    - Test:  5 trading days
    - Step:  5 trading days (non-overlapping)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_nq_data
from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

# Bars per day for each timeframe
BARS_PER_DAY_FUTURES = {  # NQ trades ~23 hours/day
    "1min": 1380, "5min": 276, "15min": 92, "30min": 46, "1h": 23,
}
BARS_PER_DAY_CRYPTO = {   # Crypto trades 24/7
    "1min": 1440, "5min": 288, "15min": 96, "30min": 48, "1h": 24,
}


def _detect_bars_per_day(filepath, tf_label):
    """Auto-detect if crypto or futures based on filename."""
    fname = Path(filepath).name.upper()
    if any(x in fname for x in ["ETH", "BTC", "SOL", "BNB", "CRYPTO"]):
        return BARS_PER_DAY_CRYPTO.get(tf_label, 1440)
    return BARS_PER_DAY_FUTURES.get(tf_label, 1380)


def main(filepath: str, timeframe: str = None):
    tf_label = timeframe or "1min"

    print("=" * 60)
    print(f"GA-MSSR Walk-Forward Validation ({tf_label})")
    print("=" * 60)

    # ---- Load & Resample Data ----
    print(f"\nLoading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
    if timeframe:
        print(f"  Resampled to {timeframe}")

    # ---- Window Configuration ----
    bpd = _detect_bars_per_day(filepath, tf_label)
    TRAIN_DAYS = 20       # 4 weeks training
    TEST_DAYS = 5          # 1 week testing
    train_size = bpd * TRAIN_DAYS
    test_size = bpd * TEST_DAYS

    total_bars_needed = train_size + test_size
    if len(df) < total_bars_needed:
        print(f"\nERROR: Need at least {total_bars_needed:,} bars "
              f"(train={train_size:,} + test={test_size:,}), "
              f"but only have {len(df):,}")
        sys.exit(1)

    max_folds = (len(df) - train_size) // test_size
    print(f"\n  Timeframe:      {tf_label} ({bpd} bars/day)")
    print(f"  Train window:   {TRAIN_DAYS} days ({train_size:,} bars)")
    print(f"  Test window:    {TEST_DAYS} days ({test_size:,} bars)")
    print(f"  Expected folds: ~{max_folds}")

    # ---- GA Configuration ----
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    # ---- Run Walk-Forward ----
    print(f"\nRunning walk-forward analysis...")
    print(f"  GA: pop={ga_config.sol_per_pop}, gens={ga_config.num_generations}")
    print("-" * 60)

    t0 = time.time()
    ga = GAMSSR(ga_config)
    result = ga.walk_forward(
        df,
        train_size=train_size,
        test_size=test_size,
        periods=[3, 7, 15, 27],  # smaller grid for speed
    )
    elapsed = time.time() - t0

    # ---- Results ----
    print("\n" + "=" * 60)
    print(f"WALK-FORWARD RESULTS — {tf_label} (OUT-OF-SAMPLE)")
    print("=" * 60)

    print(f"\n  Folds completed:    {result.num_folds}")
    print(f"  Time elapsed:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n  Aggregate OOS SSR:  {result.aggregate_ssr:.6f}")
    print(f"  Total OOS Return:   {result.total_oos_return:.6f} "
          f"({result.total_oos_return * 100:.2f}%)")

    if result.num_folds > 0:
        # Per-fold breakdown
        print(f"\n{'Fold':>6} {'OOS SSR':>12} {'OOS Return':>14} {'Bars':>8}")
        print("-" * 44)
        for i, (ssr_val, oos_ret) in enumerate(
            zip(result.oos_ssr, result.oos_returns), 1
        ):
            fold_return = float(oos_ret.sum())
            print(f"  {i:4d}   {ssr_val:11.6f}   {fold_return:12.6f}   {len(oos_ret):6d}")

        # Summary statistics
        ssr_arr = np.array(result.oos_ssr)
        returns_per_fold = [float(r.sum()) for r in result.oos_returns]
        returns_arr = np.array(returns_per_fold)

        print(f"\n--- Summary Statistics ---")
        print(f"  Mean OOS SSR:       {ssr_arr.mean():.6f}")
        print(f"  Median OOS SSR:     {np.median(ssr_arr):.6f}")
        print(f"  Std OOS SSR:        {ssr_arr.std():.6f}")
        print(f"  Min OOS SSR:        {ssr_arr.min():.6f}")
        print(f"  Max OOS SSR:        {ssr_arr.max():.6f}")

        print(f"\n  Mean fold return:   {returns_arr.mean():.6f} ({returns_arr.mean()*100:.2f}%)")
        print(f"  Positive folds:     {(returns_arr > 0).sum()} / {len(returns_arr)}")
        print(f"  Negative folds:     {(returns_arr < 0).sum()} / {len(returns_arr)}")

        # Cumulative OOS equity curve
        all_oos = pd.concat(result.oos_returns)
        cum_return = np.cumsum(all_oos.values)
        running_max = np.maximum.accumulate(cum_return)
        max_dd = np.min(cum_return - running_max)

        print(f"\n--- OOS Equity Curve ---")
        print(f"  Total OOS bars:     {len(all_oos):,}")
        print(f"  Final cum return:   {cum_return[-1]:.6f} ({cum_return[-1]*100:.2f}%)")
        print(f"  Max OOS drawdown:   {max_dd:.6f} ({max_dd*100:.2f}%)")
        print(f"  Peak cum return:    {running_max[-1]:.6f} ({running_max[-1]*100:.2f}%)")

        # Estimate dollar PnL (NQ multiplier = $20, MNQ = $2)
        avg_price = df["close"].mean()
        dollar_pnl_nq = cum_return[-1] * avg_price * 20
        dollar_pnl_mnq = cum_return[-1] * avg_price * 2
        dollar_dd_mnq = max_dd * avg_price * 2
        print(f"\n--- Estimated Dollar PnL (avg price ${avg_price:,.0f}) ---")
        print(f"  NQ  (1 contract):   ${dollar_pnl_nq:,.2f}")
        print(f"  MNQ (1 contract):   ${dollar_pnl_mnq:,.2f}")
        print(f"  MNQ max drawdown:   ${dollar_dd_mnq:,.2f}")

        # Estimated trades per day
        # Count sign changes in concatenated OOS positions
        all_positions = np.sign(all_oos.values)
        position_changes = np.sum(np.diff(all_positions) != 0)
        oos_days = len(all_oos) / bpd
        trades_per_day = position_changes / oos_days if oos_days > 0 else 0
        print(f"\n--- Trade Frequency ---")
        print(f"  Est. trades/day:    {trades_per_day:.0f}")
        print(f"  OOS trading days:   {oos_days:.0f}")

        # PRD success criteria
        print(f"\n--- PRD Success Criteria ---")
        print(f"  SSR > 1.5:          {'PASS' if result.aggregate_ssr > 1.5 else 'FAIL'} "
              f"(got {result.aggregate_ssr:.4f})")
        print(f"  OOS positive:       {'PASS' if result.total_oos_return > 0 else 'FAIL'} "
              f"(got {result.total_oos_return:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_walk_forward.py <csv> [timeframe]")
        print("  timeframe: 1min (default), 5min, 15min, 30min, 1h")
        sys.exit(1)

    filepath = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(filepath, timeframe)
