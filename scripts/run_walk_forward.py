#!/usr/bin/env python3
"""
Walk-forward validation for GA-MSSR on NQ futures data.

Trains on a rolling window, tests on unseen data, rolls forward.
This gives realistic out-of-sample performance estimates.

Usage:
    .venv/bin/python scripts/run_walk_forward.py data/futures/NQ_1min_IB.csv

Window sizes (configurable):
    - Train: 4 weeks (~26,000 bars)
    - Test:  1 week  (~6,500 bars)
    - Step:  1 week  (non-overlapping test periods)
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


def main(filepath: str):
    print("=" * 60)
    print("GA-MSSR Walk-Forward Validation")
    print("=" * 60)

    # ---- Load Data ----
    print(f"\nLoading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close")
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # ---- Window Configuration ----
    # NQ trades ~1,380 bars/day (23 hours × 60 min)
    BARS_PER_DAY = 1380
    TRAIN_DAYS = 20       # 4 weeks training
    TEST_DAYS = 5          # 1 week testing
    train_size = BARS_PER_DAY * TRAIN_DAYS   # ~27,600
    test_size = BARS_PER_DAY * TEST_DAYS     # ~6,900

    total_bars_needed = train_size + test_size
    if len(df) < total_bars_needed:
        print(f"\nERROR: Need at least {total_bars_needed:,} bars "
              f"(train={train_size:,} + test={test_size:,}), "
              f"but only have {len(df):,}")
        sys.exit(1)

    max_folds = (len(df) - train_size) // test_size
    print(f"\n  Train window: {TRAIN_DAYS} days ({train_size:,} bars)")
    print(f"  Test window:  {TEST_DAYS} days ({test_size:,} bars)")
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
    print("WALK-FORWARD RESULTS (OUT-OF-SAMPLE)")
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

        # Estimate dollar PnL (NQ multiplier = $20)
        # Using rough NQ price for dollar conversion
        avg_price = df["close"].mean()
        dollar_pnl = cum_return[-1] * avg_price * 20  # 1 contract
        dollar_dd = max_dd * avg_price * 20
        print(f"\n--- Estimated Dollar PnL (1 contract, avg price ${avg_price:,.0f}) ---")
        print(f"  Est. total PnL:     ${dollar_pnl:,.2f}")
        print(f"  Est. max drawdown:  ${dollar_dd:,.2f}")

        # PRD success criteria
        print(f"\n--- PRD Success Criteria ---")
        print(f"  SSR > 1.5:          {'PASS' if result.aggregate_ssr > 1.5 else 'FAIL'} "
              f"(got {result.aggregate_ssr:.4f})")
        print(f"  OOS positive:       {'PASS' if result.total_oos_return > 0 else 'FAIL'} "
              f"(got {result.total_oos_return:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_walk_forward.py <path_to_nq_csv>")
        sys.exit(1)
    main(sys.argv[1])
