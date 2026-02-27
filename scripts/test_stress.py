#!/usr/bin/env python3
"""
Regime / Drawdown Stress Test.

Identifies the worst market periods in the data and tests whether
the strategy would have survived. Specifically:
1. Finds the worst 5-day drawdown windows in the underlying
2. Trains the model on the 20 days before each stress period
3. Evaluates on the stress period itself
4. Reports if the strategy would have blown through risk limits

Also checks: does the strategy make money in trending vs ranging regimes?
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
from strategies.indicators import efficiency_ratio

INSTRUMENTS = {
    "NQ": {
        "filepath": "data/futures/NQ_1min_IB.csv",
        "timeframe": "15min",
        "bpd": 92,
        "multiplier": 2,   # MNQ $2/point
        "label": "MNQ",
    },
    "ETH": {
        "filepath": "data/futures/ETHUSD_1min.csv",
        "timeframe": "15min",
        "bpd": 96,
        "multiplier": 0.5,  # 0.5 ETH position size
        "label": "ETH",
    },
}

TRAIN_DAYS = 20
STRESS_DAYS = 5
TOP_N_WORST = 5
POSITION_THRESHOLD = 0.30


def find_worst_windows(df, window_bars, n=5):
    """Find the n worst non-overlapping drawdown windows."""
    close = df["close"].values
    log_returns = np.log(close[1:] / close[:-1])

    windows = []
    for i in range(len(log_returns) - window_bars + 1):
        chunk = log_returns[i:i + window_bars]
        cum_ret = np.sum(chunk)
        windows.append((i, cum_ret))

    # Sort by return (worst first)
    windows.sort(key=lambda x: x[1])

    # Pick non-overlapping
    selected = []
    used_indices = set()
    for start, ret in windows:
        if any(abs(start - u) < window_bars for u in used_indices):
            continue
        selected.append((start + 1, ret))  # +1 because log_returns is offset by 1
        used_indices.add(start)
        if len(selected) >= n:
            break

    return selected


def find_regime_windows(df, bpd, er_period=20):
    """Split data into trending and ranging regimes using Efficiency Ratio."""
    er = efficiency_ratio(df["close"], n=er_period)

    # Split into 5-day chunks and classify
    chunk_size = bpd * 5
    trending_chunks = []
    ranging_chunks = []

    for i in range(0, len(df) - chunk_size, chunk_size):
        chunk_er = er.iloc[i:i + chunk_size].mean()
        if chunk_er > 0.40:
            trending_chunks.append(i)
        elif chunk_er < 0.25:
            ranging_chunks.append(i)

    return trending_chunks, ranging_chunks


def run_stress_test(name, cfg):
    filepath = cfg["filepath"]
    tf = cfg["timeframe"]
    bpd = cfg["bpd"]
    multiplier = cfg["multiplier"]
    label = cfg["label"]

    print(f"\n{'=' * 70}")
    print(f"STRESS TEST — {name} ({tf})")
    print(f"{'=' * 70}")

    print(f"Loading {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=tf)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    train_size = bpd * TRAIN_DAYS
    stress_size = bpd * STRESS_DAYS

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    # ---- Part 1: Worst Drawdown Windows ----
    print(f"\n[1/2] Finding {TOP_N_WORST} worst {STRESS_DAYS}-day market periods...")
    worst_windows = find_worst_windows(df, stress_size, TOP_N_WORST)

    print(f"\n  {'#':>3} {'Start Date':>22} {'Market Drop':>12} {'Strat Return':>13} {'Strat DD':>10} {'Dollar PnL':>11} {'Verdict':>8}")
    print(f"  {'─'*3}─{'─'*22}─{'─'*12}─{'─'*13}─{'─'*10}─{'─'*11}─{'─'*8}")

    stress_results = []
    for rank, (start_idx, market_return) in enumerate(worst_windows, 1):
        # Need train_size bars before stress window
        if start_idx < train_size:
            continue

        train_df = df.iloc[start_idx - train_size : start_idx]
        stress_df = df.iloc[start_idx : start_idx + stress_size]

        if len(stress_df) < stress_size // 2:
            continue

        # Train and evaluate
        rule_params = train_rule_params(train_df, periods=[3, 7, 15, 27])
        train_features = get_rule_features(train_df, rule_params)
        stress_features = get_rule_features(stress_df, rule_params)

        if len(train_features) < 10 or len(stress_features) < 5:
            continue

        ga = GAMSSR(ga_config)
        ga.fit(train_features)
        pred = ga.predict(stress_features)

        strat_return = float(pred["port_return"].sum())
        cum = np.cumsum(pred["port_return"].values)
        strat_dd = float(np.min(cum - np.maximum.accumulate(cum)))

        # Dollar PnL
        avg_price = stress_df["close"].mean()
        dollar_pnl = strat_return * avg_price * multiplier
        dollar_dd = strat_dd * avg_price * multiplier

        verdict = "OK" if strat_return >= 0 else ("WARN" if strat_return > -0.01 else "LOSS")

        stress_results.append({
            "rank": rank,
            "start": stress_df.index[0],
            "market_return": market_return,
            "strat_return": strat_return,
            "strat_dd": strat_dd,
            "dollar_pnl": dollar_pnl,
            "dollar_dd": dollar_dd,
            "verdict": verdict,
        })

        print(f"  {rank:3d} {str(stress_df.index[0]):>22} "
              f"{market_return*100:+10.2f}% "
              f"{strat_return*100:+11.2f}% "
              f"{strat_dd*100:8.2f}% "
              f"${dollar_pnl:>9,.2f} "
              f"  {verdict}")

    # Stress summary
    if stress_results:
        survived = [r for r in stress_results if r["strat_return"] >= 0]
        worst_strat_dd = min(r["dollar_dd"] for r in stress_results)
        avg_return = np.mean([r["strat_return"] for r in stress_results])

        print(f"\n  Survived (profitable): {len(survived)}/{len(stress_results)}")
        print(f"  Avg strategy return during stress: {avg_return*100:+.2f}%")
        print(f"  Worst strategy DD during stress: ${worst_strat_dd:,.2f}")

    # ---- Part 2: Trending vs Ranging Regimes ----
    print(f"\n[2/2] Regime analysis (trending vs ranging)...")
    trending_starts, ranging_starts = find_regime_windows(df, bpd)

    regime_results = {"trending": [], "ranging": []}

    for regime_name, starts in [("trending", trending_starts), ("ranging", ranging_starts)]:
        for start_idx in starts:
            if start_idx < train_size:
                continue
            if start_idx + stress_size > len(df):
                continue

            train_df = df.iloc[start_idx - train_size : start_idx]
            test_df = df.iloc[start_idx : start_idx + stress_size]

            rule_params = train_rule_params(train_df, periods=[3, 7, 15, 27])
            train_features = get_rule_features(train_df, rule_params)
            test_features = get_rule_features(test_df, rule_params)

            if len(train_features) < 10 or len(test_features) < 5:
                continue

            ga = GAMSSR(ga_config)
            ga.fit(train_features)
            pred = ga.predict(test_features)
            ret = float(pred["port_return"].sum())
            regime_results[regime_name].append(ret)

    print(f"\n  {'Regime':>10} | {'Windows':>7} | {'Avg Return':>10} | {'Win Rate':>8} | {'Verdict':>8}")
    print(f"  {'─'*10}-+-{'─'*7}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}")

    for regime_name in ["trending", "ranging"]:
        rets = regime_results[regime_name]
        if rets:
            avg_ret = np.mean(rets)
            win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
            verdict = "GOOD" if win_rate > 55 else ("OK" if win_rate > 40 else "WEAK")
            print(f"  {regime_name:>10} | {len(rets):>7} | {avg_ret*100:+9.2f}% | {win_rate:6.0f}% | {verdict:>8}")
        else:
            print(f"  {regime_name:>10} | {0:>7} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8}")

    # Final verdict
    print(f"\n  {'─' * 60}")
    if stress_results:
        survival_rate = len(survived) / len(stress_results) * 100
        if survival_rate >= 60:
            print(f"  VERDICT: RESILIENT — Survived {survival_rate:.0f}% of worst periods")
        elif survival_rate >= 40:
            print(f"  VERDICT: MODERATE — Survived {survival_rate:.0f}% of worst periods")
        else:
            print(f"  VERDICT: VULNERABLE — Only survived {survival_rate:.0f}% of worst periods")


if __name__ == "__main__":
    for name, cfg in INSTRUMENTS.items():
        if Path(cfg["filepath"]).exists():
            run_stress_test(name, cfg)
        else:
            print(f"\nSkipping {name}: {cfg['filepath']} not found")

    print("\nDone.")
