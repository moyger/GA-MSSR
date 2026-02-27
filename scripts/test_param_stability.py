#!/usr/bin/env python3
"""
Parameter Stability Test: Run walk-forward with different GA seeds.

Verifies that strategy performance isn't an artifact of a specific seed.
If results vary wildly across seeds, the strategy is fragile.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig

SEEDS = [42, 123, 456, 789]
INSTRUMENTS = {
    "NQ": ("data/futures/NQ_1min_IB.csv", "15min", 92),
    "ETH": ("data/futures/ETHUSD_1min.csv", "15min", 96),
}

TRAIN_DAYS = 20
TEST_DAYS = 5


def run_seed_test(name, filepath, timeframe, bpd):
    print(f"\n{'=' * 70}")
    print(f"PARAMETER STABILITY TEST — {name} ({timeframe})")
    print(f"{'=' * 70}")

    print(f"Loading {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    train_size = bpd * TRAIN_DAYS
    test_size = bpd * TEST_DAYS

    results = {}
    for seed in SEEDS:
        t0 = time.time()
        ga_config = GAMSSRConfig(
            sol_per_pop=20,
            num_parents_mating=10,
            num_generations=200,
            random_seed=seed,
        )
        ga = GAMSSR(ga_config)
        result = ga.walk_forward(
            df,
            train_size=train_size,
            test_size=test_size,
            periods=[3, 7, 15, 27],
        )
        elapsed = time.time() - t0

        # Compute max drawdown
        if result.oos_returns:
            all_oos = pd.concat(result.oos_returns)
            cum = np.cumsum(all_oos.values)
            running_max = np.maximum.accumulate(cum)
            max_dd = float(np.min(cum - running_max))
        else:
            max_dd = 0.0

        results[seed] = {
            "ssr": result.aggregate_ssr,
            "return": result.total_oos_return,
            "folds": result.num_folds,
            "max_dd": max_dd,
            "time": elapsed,
        }
        print(f"  Seed {seed:>4}: SSR={result.aggregate_ssr:.6f}  "
              f"Return={result.total_oos_return*100:+.2f}%  "
              f"MaxDD={max_dd*100:.2f}%  "
              f"Folds={result.num_folds}  ({elapsed:.1f}s)")

    # Summary
    ssrs = [r["ssr"] for r in results.values()]
    rets = [r["return"] for r in results.values()]
    dds = [r["max_dd"] for r in results.values()]

    print(f"\n{'─' * 70}")
    print(f"STABILITY SUMMARY — {name}")
    print(f"{'─' * 70}")
    print(f"  SSR   — Mean: {np.mean(ssrs):.6f}  Std: {np.std(ssrs):.6f}  "
          f"Range: [{min(ssrs):.6f}, {max(ssrs):.6f}]  "
          f"CV: {np.std(ssrs)/max(np.mean(ssrs), 1e-10)*100:.1f}%")
    print(f"  Return — Mean: {np.mean(rets)*100:+.2f}%  Std: {np.std(rets)*100:.2f}%  "
          f"Range: [{min(rets)*100:+.2f}%, {max(rets)*100:+.2f}%]")
    print(f"  MaxDD  — Mean: {np.mean(dds)*100:.2f}%  "
          f"Range: [{min(dds)*100:.2f}%, {max(dds)*100:.2f}%]")

    # Verdict
    cv = np.std(ssrs) / max(np.mean(ssrs), 1e-10) * 100
    all_positive = all(r > 0 for r in rets)
    print(f"\n  SSR CV (coefficient of variation): {cv:.1f}%")
    if cv < 20 and all_positive:
        print(f"  VERDICT: STABLE — Low SSR variation, all seeds profitable")
    elif cv < 50 and all_positive:
        print(f"  VERDICT: MODERATE — Some variation but consistently profitable")
    elif all_positive:
        print(f"  VERDICT: FRAGILE — High variation across seeds (but all profitable)")
    else:
        neg_seeds = [s for s, r in results.items() if r["return"] <= 0]
        print(f"  VERDICT: UNSTABLE — Seeds {neg_seeds} produced negative returns")


if __name__ == "__main__":
    for name, (filepath, tf, bpd) in INSTRUMENTS.items():
        if Path(filepath).exists():
            run_seed_test(name, filepath, tf, bpd)
        else:
            print(f"\nSkipping {name}: {filepath} not found")

    print("\nDone.")
