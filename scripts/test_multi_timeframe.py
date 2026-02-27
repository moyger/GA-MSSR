#!/usr/bin/env python3
"""
Multi-Timeframe Consistency Test.

Runs walk-forward on 5min, 15min, and 30min for NQ and ETH.
If the strategy only works on one timeframe, it's fragile.
A robust strategy should show positive OOS returns across timeframes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig

BARS_PER_DAY_FUTURES = {"5min": 276, "15min": 92, "30min": 46}
BARS_PER_DAY_CRYPTO = {"5min": 288, "15min": 96, "30min": 48}

INSTRUMENTS = {
    "NQ": ("data/futures/NQ_1min_IB.csv", BARS_PER_DAY_FUTURES),
    "ETH": ("data/futures/ETHUSD_1min.csv", BARS_PER_DAY_CRYPTO),
}

TIMEFRAMES = ["5min", "15min", "30min"]
TRAIN_DAYS = 20
TEST_DAYS = 5


def run_timeframe_test(name, filepath, bpd_map):
    print(f"\n{'=' * 70}")
    print(f"MULTI-TIMEFRAME CONSISTENCY — {name}")
    print(f"{'=' * 70}")

    results = {}
    for tf in TIMEFRAMES:
        bpd = bpd_map[tf]
        train_size = bpd * TRAIN_DAYS
        test_size = bpd * TEST_DAYS

        print(f"\n  [{tf}] Loading and resampling...")
        try:
            df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=tf)
        except Exception as e:
            print(f"  [{tf}] ERROR loading data: {e}")
            continue

        total_needed = train_size + test_size
        if len(df) < total_needed:
            print(f"  [{tf}] Not enough data: {len(df)} < {total_needed}")
            continue

        print(f"  [{tf}] {len(df):,} bars | train={train_size} test={test_size}")

        ga_config = GAMSSRConfig(
            sol_per_pop=20,
            num_parents_mating=10,
            num_generations=200,
            random_seed=42,
        )

        t0 = time.time()
        ga = GAMSSR(ga_config)
        result = ga.walk_forward(
            df,
            train_size=train_size,
            test_size=test_size,
            periods=[3, 7, 15, 27],
        )
        elapsed = time.time() - t0

        # Compute metrics
        if result.oos_returns:
            all_oos = pd.concat(result.oos_returns)
            cum = np.cumsum(all_oos.values)
            running_max = np.maximum.accumulate(cum)
            max_dd = float(np.min(cum - running_max))

            # Trade frequency
            positions = np.sign(all_oos.values)
            pos_changes = int(np.sum(np.diff(positions) != 0))
            oos_days = len(all_oos) / bpd
            trades_per_day = pos_changes / max(oos_days, 0.01)
        else:
            max_dd = 0.0
            trades_per_day = 0.0

        results[tf] = {
            "ssr": result.aggregate_ssr,
            "return": result.total_oos_return,
            "folds": result.num_folds,
            "max_dd": max_dd,
            "trades_per_day": trades_per_day,
            "time": elapsed,
        }

        print(f"  [{tf}] SSR={result.aggregate_ssr:.6f}  "
              f"Return={result.total_oos_return*100:+.2f}%  "
              f"MaxDD={max_dd*100:.2f}%  "
              f"Folds={result.num_folds}  "
              f"Trades/day={trades_per_day:.0f}  "
              f"({elapsed:.1f}s)")

    # Summary table
    print(f"\n{'─' * 70}")
    print(f"TIMEFRAME COMPARISON — {name}")
    print(f"{'─' * 70}")
    print(f"  {'TF':>5} | {'SSR':>10} | {'Return':>10} | {'MaxDD':>8} | {'Trades/d':>9} | {'Folds':>5}")
    print(f"  {'─'*5}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*9}-+-{'─'*5}")
    for tf in TIMEFRAMES:
        if tf in results:
            r = results[tf]
            print(f"  {tf:>5} | {r['ssr']:10.6f} | {r['return']*100:+9.2f}% | "
                  f"{r['max_dd']*100:7.2f}% | {r['trades_per_day']:8.0f} | {r['folds']:5d}")

    # Verdict
    profitable = [tf for tf, r in results.items() if r["return"] > 0]
    positive_ssr = [tf for tf, r in results.items() if r["ssr"] > 0]

    print(f"\n  Profitable timeframes: {len(profitable)}/{len(results)} "
          f"({', '.join(profitable)})")
    print(f"  Positive SSR:          {len(positive_ssr)}/{len(results)} "
          f"({', '.join(positive_ssr)})")

    if len(profitable) == len(results):
        print(f"  VERDICT: CONSISTENT — Strategy profitable across all timeframes")
    elif len(profitable) >= 2:
        print(f"  VERDICT: MOSTLY CONSISTENT — Profitable on {len(profitable)}/{len(results)} timeframes")
    else:
        print(f"  VERDICT: TIMEFRAME-DEPENDENT — Only works on {', '.join(profitable)}")


if __name__ == "__main__":
    for name, (filepath, bpd_map) in INSTRUMENTS.items():
        if Path(filepath).exists():
            run_timeframe_test(name, filepath, bpd_map)
        else:
            print(f"\nSkipping {name}: {filepath} not found")

    print("\nDone.")
