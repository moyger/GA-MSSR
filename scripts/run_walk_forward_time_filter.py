#!/usr/bin/env python3
"""
Walk-forward validation with NY session time filter for GA-MSSR.

Compares 4 configurations:
  1. Baseline (no filter)
  2. NY session only (8:30 AM - 12:00 PM ET)
  3. ER filter only (p=14 t=0.40)
  4. NY session + ER filter (combined)

Usage:
    .venv/bin/python scripts/run_walk_forward_time_filter.py data/futures/NQ_in_15_minute.csv
    .venv/bin/python scripts/run_walk_forward_time_filter.py data/futures/NQ_1min_IB.csv 15min
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from strategies.indicators import efficiency_ratio
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

# ---- Constants ----
BARS_PER_DAY_FUTURES = {
    "1min": 1380, "5min": 276, "15min": 92, "30min": 46, "1h": 23,
}
TRAIN_DAYS = 20
TEST_DAYS = 5
POSITION_THRESHOLD = 0.10

# NY session window (Eastern Time)
NY_START_HOUR, NY_START_MIN = 8, 30
NY_END_HOUR, NY_END_MIN = 12, 0


def _infer_timeframe(filepath: str) -> str:
    fname = Path(filepath).stem.lower()
    for tf, label in [
        ("15_minute", "15min"), ("30_minute", "30min"),
        ("1_minute", "1min"), ("3_minute", "3min"), ("5_minute", "5min"),
        ("4_hour", "4h"), ("2_hour", "2h"), ("3_hour", "3h"), ("1_hour", "1h"),
    ]:
        if tf in fname:
            return label
    return "1min"


def _detect_bars_per_day(filepath, tf_label):
    return BARS_PER_DAY_FUTURES.get(tf_label, 1380)


def _discretize(position: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    return np.where(
        position > threshold, 1.0,
        np.where(position < -threshold, -1.0, 0.0),
    )


def _ny_session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True if bar falls within NY session (8:30 AM - 12:00 PM ET)."""
    if index.tz is None:
        et_index = index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        et_index = index.tz_convert("US/Eastern")

    hours = et_index.hour
    minutes = et_index.minute
    time_minutes = hours * 60 + minutes

    start = NY_START_HOUR * 60 + NY_START_MIN  # 8:30 = 510
    end = NY_END_HOUR * 60 + NY_END_MIN        # 12:00 = 720

    return (time_minutes >= start) & (time_minutes < end)


def _er_mask(close: pd.Series, period: int = 14, threshold: float = 0.40) -> np.ndarray:
    """Boolean mask: True if ER > threshold (trending)."""
    er = efficiency_ratio(close, n=period)
    return er.values > threshold


def main(filepath: str, timeframe: str = None):
    tf_label = timeframe or _infer_timeframe(filepath)

    print("=" * 80)
    print(f"GA-MSSR Walk-Forward: NY Session + ER Filter ({tf_label})")
    print("=" * 80)

    # ---- Load Data ----
    print(f"\nLoading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    bpd = _detect_bars_per_day(filepath, tf_label)
    train_size = bpd * TRAIN_DAYS
    test_size = bpd * TEST_DAYS

    if len(df) < train_size + test_size:
        print(f"\nERROR: Need {train_size + test_size:,} bars, have {len(df):,}")
        sys.exit(1)

    max_folds = (len(df) - train_size) // test_size
    print(f"  Train: {TRAIN_DAYS}d ({train_size:,} bars) | Test: {TEST_DAYS}d ({test_size:,} bars)")
    print(f"  Expected folds: ~{max_folds}")

    # Show session coverage
    ny_mask_full = _ny_session_mask(df.index)
    print(f"  NY session bars: {ny_mask_full.sum():,} / {len(df):,} "
          f"({ny_mask_full.sum()/len(df)*100:.1f}%)")

    # ---- GA Config ----
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    # ---- Filter Configs ----
    configs = [
        {"label": "Baseline", "ny_session": False, "er": False},
        {"label": "NY 8:30-12 ET", "ny_session": True, "er": False},
        {"label": "ER p=14 t=0.40", "ny_session": False, "er": True},
        {"label": "NY + ER", "ny_session": True, "er": True},
    ]

    # ---- Walk-Forward Loop ----
    print(f"\nRunning walk-forward (GA pop={ga_config.sol_per_pop}, gens={ga_config.num_generations})...")
    print("-" * 80)

    ssr_calc = SSR()
    n = len(df)
    start = 0
    fold_num = 0
    fold_data = {c["label"]: {"oos_returns": [], "oos_ssr": []} for c in configs}

    t0 = time.time()

    while start + train_size + test_size <= n:
        fold_num += 1
        train_df = df.iloc[start : start + train_size]
        test_df = df.iloc[start + train_size : start + train_size + test_size]

        # Train
        rule_params = train_rule_params(train_df, periods=[3, 7, 15, 27])
        train_features = get_rule_features(train_df, rule_params)
        test_features = get_rule_features(test_df, rule_params)

        if len(train_features) < 10 or len(test_features) < 5:
            start += test_size
            continue

        ga = GAMSSR(ga_config)
        result = ga.fit(train_features)

        # Get test positions
        test_pred = ga.predict(test_features)
        raw_position = test_pred["position"].values
        discretized = _discretize(raw_position, POSITION_THRESHOLD)
        logr = test_features["logr"].values.astype(np.float64)

        # Pre-compute masks on test data
        test_ohlc = test_df.loc[test_features.index]
        ny_mask = _ny_session_mask(test_features.index)
        er_mask = _er_mask(test_ohlc["close"], period=14, threshold=0.40)

        fold_line = f"  Fold {fold_num:2d} | SSR(train)={result.best_fitness:.4f} | "
        for config in configs:
            # Build combined mask
            mask = np.ones(len(discretized), dtype=bool)
            if config["ny_session"]:
                mask = mask & ny_mask
            if config["er"]:
                mask = mask & er_mask

            filtered_pos = np.where(mask, discretized, 0.0)
            port_ret = filtered_pos * logr
            oos_result = ssr_calc.calculate(port_ret)
            oos_ret_series = pd.Series(port_ret, index=test_features.index)

            fold_data[config["label"]]["oos_returns"].append(oos_ret_series)
            fold_data[config["label"]]["oos_ssr"].append(oos_result.ssr)

            fold_line += f"{config['label']}={oos_result.ssr:.4f} "

        print(fold_line)
        start += test_size

    elapsed = time.time() - t0
    print(f"\n  Completed {fold_num} folds in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- Aggregate Results ----
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS — OUT-OF-SAMPLE COMPARISON")
    print("=" * 80)

    print(f"\n{'Config':<22} | {'Agg SSR':>9} | {'Return%':>9} | {'MaxDD%':>8} | "
          f"{'Mean SSR':>9} | {'Med SSR':>9} | {'Win%':>5} | {'Trades':>6} | {'Flat%':>6}")
    print("-" * 110)

    summary_rows = []
    for config in configs:
        label = config["label"]
        data = fold_data[label]

        if not data["oos_returns"]:
            continue

        all_oos = pd.concat(data["oos_returns"])
        agg_result = ssr_calc.calculate(all_oos.values)
        total_ret = float(all_oos.sum())

        cum_ret = np.cumsum(all_oos.values)
        running_max = np.maximum.accumulate(cum_ret)
        max_dd = float(np.min(cum_ret - running_max))

        ssr_arr = np.array(data["oos_ssr"])
        returns_per_fold = [float(r.sum()) for r in data["oos_returns"]]
        returns_arr = np.array(returns_per_fold)
        positive_folds = int((returns_arr > 0).sum())
        win_pct = positive_folds / len(returns_arr) * 100

        # Trade and flat counts
        all_positions = np.sign(all_oos.values)
        trades = int(np.sum(np.diff(all_positions) != 0))

        # Count zero-return bars as flat
        flat_pct = float(np.sum(all_oos.values == 0)) / len(all_oos) * 100

        print(f"  {label:<20} | {agg_result.ssr:9.6f} | {total_ret*100:+8.2f}% | "
              f"{max_dd*100:+7.2f}% | {ssr_arr.mean():9.6f} | {np.median(ssr_arr):9.6f} | "
              f"{win_pct:4.0f}% | {trades:6d} | {flat_pct:5.1f}%")

        summary_rows.append({
            "label": label,
            "agg_ssr": agg_result.ssr,
            "total_return": total_ret,
            "max_dd": max_dd,
            "mean_ssr": ssr_arr.mean(),
            "median_ssr": np.median(ssr_arr),
            "win_pct": win_pct,
            "trades": trades,
            "flat_pct": flat_pct,
            "positive_folds": positive_folds,
            "total_folds": len(returns_arr),
        })

    print("-" * 110)

    # ---- Comparison ----
    if len(summary_rows) >= 2:
        baseline = summary_rows[0]
        best = max(summary_rows[1:], key=lambda r: r["agg_ssr"])

        print(f"\n  Best config (OOS): {best['label']} (SSR={best['agg_ssr']:.6f})")
        print(f"\n  vs Baseline:")

        ssr_diff = best["agg_ssr"] - baseline["agg_ssr"]
        ret_diff = (best["total_return"] - baseline["total_return"]) * 100
        dd_diff = (baseline["max_dd"] - best["max_dd"]) * 100
        print(f"    SSR:       {baseline['agg_ssr']:.6f} -> {best['agg_ssr']:.6f} ({ssr_diff:+.6f})")
        print(f"    Return:    {baseline['total_return']*100:+.2f}% -> {best['total_return']*100:+.2f}% ({ret_diff:+.2f}%)")
        print(f"    MaxDD:     {baseline['max_dd']*100:+.2f}% -> {best['max_dd']*100:+.2f}% ({dd_diff:+.2f}% better)")
        print(f"    Win folds: {baseline['positive_folds']}/{baseline['total_folds']} -> {best['positive_folds']}/{best['total_folds']}")
        print(f"    Trades:    {baseline['trades']} -> {best['trades']}")
        print(f"    Flat%:     {baseline['flat_pct']:.1f}% -> {best['flat_pct']:.1f}%")

        # Dollar PnL estimate (MNQ @ $2/point)
        avg_price = df["close"].mean()
        print(f"\n  MNQ Dollar Estimates (avg price ${avg_price:,.0f}):")
        for row in summary_rows:
            dollar_pnl = row["total_return"] * avg_price * 2
            dollar_dd = row["max_dd"] * avg_price * 2
            print(f"    {row['label']:<20} PnL: ${dollar_pnl:>10,.2f}  DD: ${dollar_dd:>10,.2f}")

    # ---- Per-Fold Breakdown ----
    if fold_num > 0:
        print(f"\n{'Fold':>6}", end="")
        for config in configs:
            print(f" | {config['label']:>20}", end="")
        print()
        print("-" * (8 + 23 * len(configs)))

        for i in range(fold_num):
            print(f"  {i+1:4d}", end="")
            for config in configs:
                label = config["label"]
                if i < len(fold_data[label]["oos_ssr"]):
                    ssr_val = fold_data[label]["oos_ssr"][i]
                    ret_val = float(fold_data[label]["oos_returns"][i].sum()) * 100
                    print(f" | {ssr_val:8.4f} ({ret_val:+6.2f}%)", end="")
                else:
                    print(f" | {'N/A':>20}", end="")
            print()

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_walk_forward_time_filter.py <csv> [timeframe]")
        print("  timeframe: 5min, 15min, 30min, 1h (omit for raw data)")
        sys.exit(1)

    filepath = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(filepath, timeframe)
