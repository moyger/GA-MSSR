#!/usr/bin/env python3
"""
Out-of-Sample Holdout Test.

Reserves the last 30 days of data as a true holdout set.
Runs walk-forward on the earlier data to train, then evaluates
the FINAL fold's weights on the holdout — simulating what would
happen if we deployed today.

This is the most honest test: the model has NEVER seen this data.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR
from strategies.khushi_rules import train_rule_params, get_rule_features

HOLDOUT_DAYS = 30
TRAIN_DAYS = 20
TEST_DAYS = 5

INSTRUMENTS = {
    "NQ": ("data/futures/NQ_1min_IB.csv", "15min", 92),
    "ETH": ("data/futures/ETHUSD_1min.csv", "15min", 96),
}


def run_holdout_test(name, filepath, timeframe, bpd):
    print(f"\n{'=' * 70}")
    print(f"OUT-OF-SAMPLE HOLDOUT TEST — {name} ({timeframe})")
    print(f"{'=' * 70}")

    print(f"Loading {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    print(f"  Total: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    holdout_bars = bpd * HOLDOUT_DAYS
    train_size = bpd * TRAIN_DAYS
    test_size = bpd * TEST_DAYS

    if len(df) < holdout_bars + train_size + test_size:
        print(f"  ERROR: Not enough data. Need {holdout_bars + train_size + test_size:,} bars, "
              f"have {len(df):,}")
        return

    # Split: development vs holdout
    dev_df = df.iloc[:-holdout_bars]
    holdout_df = df.iloc[-holdout_bars:]

    print(f"  Development: {len(dev_df):,} bars ({dev_df.index[0]} to {dev_df.index[-1]})")
    print(f"  Holdout:     {len(holdout_df):,} bars ({holdout_df.index[0]} to {holdout_df.index[-1]})")

    # ---- Part 1: Walk-forward on development set ----
    print(f"\n[1/3] Walk-forward on development set...")
    t0 = time.time()
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )
    ga = GAMSSR(ga_config)
    wf_result = ga.walk_forward(
        dev_df,
        train_size=train_size,
        test_size=test_size,
        periods=[3, 7, 15, 27],
    )
    elapsed_wf = time.time() - t0
    print(f"  Dev OOS SSR:    {wf_result.aggregate_ssr:.6f}")
    print(f"  Dev OOS Return: {wf_result.total_oos_return*100:+.2f}%")
    print(f"  Folds:          {wf_result.num_folds}")
    print(f"  Time:           {elapsed_wf:.1f}s")

    # ---- Part 2: Train final model on last TRAIN_DAYS of dev set ----
    print(f"\n[2/3] Training final model on last {TRAIN_DAYS} days of dev set...")
    final_train_df = dev_df.iloc[-train_size:]
    print(f"  Training on: {final_train_df.index[0]} to {final_train_df.index[-1]}")

    rule_params = train_rule_params(final_train_df, periods=[3, 7, 15, 27])
    train_features = get_rule_features(final_train_df, rule_params)

    ga_final = GAMSSR(ga_config)
    fit_result = ga_final.fit(train_features)
    print(f"  Training SSR:   {fit_result.best_fitness:.6f}")
    print(f"  Weights: {np.round(fit_result.best_weights, 3)}")

    # ---- Part 3: Evaluate on holdout ----
    print(f"\n[3/3] Evaluating on {HOLDOUT_DAYS}-day holdout (true OOS)...")
    holdout_features = get_rule_features(holdout_df, rule_params)
    holdout_pred = ga_final.predict(holdout_features)

    holdout_returns = holdout_pred["port_return"]
    ssr_calc = SSR()
    holdout_ssr_result = ssr_calc.calculate(holdout_returns.values)

    cum = np.cumsum(holdout_returns.values)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - running_max))
    total_return = float(holdout_returns.sum())

    # Position stats
    positions = holdout_pred["position"].values
    discretized = np.where(positions > 0.30, 1, np.where(positions < -0.30, -1, 0))
    flat_pct = (discretized == 0).sum() / len(discretized) * 100
    trades = np.sum(np.diff(discretized) != 0)
    trades_per_day = trades / HOLDOUT_DAYS

    print(f"\n{'─' * 70}")
    print(f"HOLDOUT RESULTS — {name} (last {HOLDOUT_DAYS} days, NEVER SEEN)")
    print(f"{'─' * 70}")
    print(f"  Period:          {holdout_df.index[0]} to {holdout_df.index[-1]}")
    print(f"  Holdout SSR:     {holdout_ssr_result.ssr:.6f}")
    print(f"  Holdout Return:  {total_return*100:+.2f}%")
    print(f"  Max Drawdown:    {max_dd*100:.2f}%")
    print(f"  Trades:          {trades} ({trades_per_day:.1f}/day)")
    print(f"  Flat %:          {flat_pct:.1f}%")

    # Dollar estimates
    avg_price = holdout_df["close"].mean()
    if "NQ" in name.upper():
        dollar_pnl = total_return * avg_price * 2  # MNQ $2/point
        dollar_dd = max_dd * avg_price * 2
        print(f"  MNQ PnL:         ${dollar_pnl:,.2f} (1 contract)")
        print(f"  MNQ Max DD:      ${dollar_dd:,.2f}")
    else:
        dollar_pnl = total_return * avg_price * 0.5  # 0.5 ETH position
        print(f"  ETH PnL:         ${dollar_pnl:,.2f} (0.5 ETH)")

    # Compare dev vs holdout
    print(f"\n  Dev OOS SSR:     {wf_result.aggregate_ssr:.6f}")
    print(f"  Holdout SSR:     {holdout_ssr_result.ssr:.6f}")
    ratio = holdout_ssr_result.ssr / max(wf_result.aggregate_ssr, 1e-10)
    print(f"  Holdout/Dev:     {ratio:.2f}x")

    if total_return > 0 and holdout_ssr_result.ssr > 0:
        print(f"\n  VERDICT: PASS — Holdout is profitable with positive SSR")
    elif total_return > 0:
        print(f"\n  VERDICT: MARGINAL — Profitable but poor risk-adjusted return")
    else:
        print(f"\n  VERDICT: FAIL — Holdout is not profitable")


if __name__ == "__main__":
    for name, (filepath, tf, bpd) in INSTRUMENTS.items():
        if Path(filepath).exists():
            run_holdout_test(name, filepath, tf, bpd)
        else:
            print(f"\nSkipping {name}: {filepath} not found")

    print("\nDone.")
