#!/usr/bin/env python3
"""
Trend/chop filter backtest for GA-MSSR.

Trains one model, then tests different trend filters (ADX, CHOP, Efficiency
Ratio) to see if skipping choppy markets improves performance.

The filter is applied as a post-GA gate: when the market is detected as choppy,
the position is forced to 0 regardless of the signal.

Usage:
    .venv/bin/python scripts/run_filter_backtest.py data/futures/NQ_1min_IB.csv 15min
    .venv/bin/python scripts/run_filter_backtest.py data/futures/NQ_in_15_minute.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from strategies.indicators import adx, chop, efficiency_ratio
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR


# Filter configurations to sweep
FILTER_CONFIGS = [
    # Baseline: no filter
    {"label": "NONE (baseline)", "type": "none"},
    # Efficiency Ratio sweeps
    {"label": "ER p=10 t=0.20", "type": "er", "period": 10, "threshold": 0.20},
    {"label": "ER p=10 t=0.30", "type": "er", "period": 10, "threshold": 0.30},
    {"label": "ER p=10 t=0.40", "type": "er", "period": 10, "threshold": 0.40},
    {"label": "ER p=14 t=0.20", "type": "er", "period": 14, "threshold": 0.20},
    {"label": "ER p=14 t=0.30", "type": "er", "period": 14, "threshold": 0.30},
    {"label": "ER p=14 t=0.40", "type": "er", "period": 14, "threshold": 0.40},
    {"label": "ER p=20 t=0.25", "type": "er", "period": 20, "threshold": 0.25},
    {"label": "ER p=20 t=0.35", "type": "er", "period": 20, "threshold": 0.35},
    # ADX sweeps
    {"label": "ADX p=14 t=20", "type": "adx", "period": 14, "threshold": 20.0},
    {"label": "ADX p=14 t=25", "type": "adx", "period": 14, "threshold": 25.0},
    {"label": "ADX p=14 t=30", "type": "adx", "period": 14, "threshold": 30.0},
    # ADX + DI spread
    {"label": "ADX+DI t=25 s=5", "type": "adx_di", "period": 14, "threshold": 25.0, "di_min": 5.0},
    {"label": "ADX+DI t=25 s=10", "type": "adx_di", "period": 14, "threshold": 25.0, "di_min": 10.0},
    # CHOP sweeps (inverted: lower = more trending)
    {"label": "CHOP p=14 t=45", "type": "chop", "period": 14, "threshold": 45.0},
    {"label": "CHOP p=14 t=50", "type": "chop", "period": 14, "threshold": 50.0},
    {"label": "CHOP p=14 t=55", "type": "chop", "period": 14, "threshold": 55.0},
]


def _count_trades(positions: np.ndarray) -> int:
    """Count position sign changes (each change = 1 trade)."""
    signs = np.sign(positions)
    return int(np.sum(np.diff(signs) != 0))


def _trade_stats(positions: np.ndarray, logr: np.ndarray):
    """Compute per-trade win/loss statistics."""
    signs = np.sign(positions)
    changes = np.where(np.diff(signs) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(positions)]])

    wins, losses = [], []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 1:
            continue
        seg_pos = positions[start:end]
        seg_ret = logr[start:end]
        trade_pnl = float(np.sum(seg_pos * seg_ret))
        if seg_pos[0] == 0:
            continue
        if trade_pnl > 0:
            wins.append(trade_pnl)
        elif trade_pnl < 0:
            losses.append(trade_pnl)

    total = len(wins) + len(losses)
    win_rate = len(wins) / total * 100 if total > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return win_rate, ratio, len(wins), len(losses)


def _compute_trending_mask(
    df: pd.DataFrame, config: dict,
) -> np.ndarray:
    """Compute a boolean trending mask for the given filter config.

    Returns a numpy array of bools aligned with df's index.
    True = trending (allow trading), False = choppy (force flat).
    """
    ftype = config["type"]

    if ftype == "none":
        return np.ones(len(df), dtype=bool)

    period = config["period"]

    if ftype == "er":
        er = efficiency_ratio(df["close"], n=period)
        return er.values > config["threshold"]

    if ftype in ("adx", "adx_di"):
        adx_val, di_plus, di_minus = adx(df["high"], df["low"], df["close"], n=period)
        mask = adx_val.values > config["threshold"]
        if ftype == "adx_di":
            di_spread = np.abs(di_plus.values - di_minus.values)
            mask = mask & (di_spread > config.get("di_min", 5.0))
        return mask

    if ftype == "chop":
        chop_val = chop(df["high"], df["low"], df["close"], n=period)
        # CHOP is inverted: low = trending
        return chop_val.values < config["threshold"]

    return np.ones(len(df), dtype=bool)


def main(filepath: str, timeframe: str = None):
    print("=" * 80)
    print("GA-MSSR Trend/Chop Filter Backtest")
    print("=" * 80)

    # ---- Phase 1: Load & Denoise ----
    print(f"\n[1/4] Loading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    label = timeframe or "raw"
    print(f"      {len(df):,} bars ({label}) | {df.index[0]} to {df.index[-1]}")

    # ---- Phase 2: Train Rule Parameters ----
    print("\n[2/4] Training 16 rule parameters (grid search)...")
    t0 = time.time()
    rule_params = train_rule_params(df)
    print(f"      Done in {time.time() - t0:.1f}s")

    # ---- Phase 3: Generate Features + GA Optimize ----
    print("\n[3/4] Generating features & running GA optimization...")
    features = get_rule_features(df, rule_params)
    print(f"      Feature matrix: {features.shape}")

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )
    t0 = time.time()
    ga = GAMSSR(ga_config)
    result = ga.fit(features)
    print(f"      Done in {time.time() - t0:.1f}s | SSR={result.best_fitness:.6f}")
    print(f"      Weights: {np.round(result.best_weights, 3)}")

    # ---- Phase 4: Filter Sweep ----
    print("\n[4/4] Sweeping trend/chop filters...")

    logr = features["logr"].values.astype(np.float64)
    signal_cols = [c for c in features.columns if c.startswith("rule_")]
    signals = features[signal_cols].values.astype(np.float64)

    # Raw continuous position → discretize with threshold=0.1 (standard)
    raw_position = signals @ result.best_weights
    threshold = 0.1
    discretized = np.where(
        raw_position > threshold, 1.0,
        np.where(raw_position < -threshold, -1.0, 0.0),
    )

    # Align the features DataFrame index with the discretized array.
    # The filter indicators need the original OHLC data. Since features
    # may have fewer rows than df (NaN dropped), align by index.
    features_df = df.loc[features.index].copy()

    ssr_calc = SSR()
    rows = []

    for config in FILTER_CONFIGS:
        # Compute trending mask on the aligned OHLC data
        mask = _compute_trending_mask(features_df, config)

        # Apply filter: zero-out positions where not trending
        filtered_pos = np.where(mask, discretized, 0.0)

        # Portfolio returns
        port_ret = filtered_pos * logr
        total_bars = len(filtered_pos)
        flat_bars = int(np.sum(filtered_pos == 0))
        flat_pct = flat_bars / total_bars * 100

        # Metrics
        metrics = ssr_calc.calculate(port_ret)
        trades = _count_trades(filtered_pos)
        win_rate, wl_ratio, wins, losses_count = _trade_stats(filtered_pos, logr)

        rows.append({
            "label": config["label"],
            "type": config["type"],
            "trades": trades,
            "return_pct": metrics.total_return * 100,
            "max_dd_pct": metrics.max_drawdown * 100,
            "ssr": metrics.ssr,
            "win_rate": win_rate,
            "wl_ratio": wl_ratio,
            "flat_pct": flat_pct,
            "wins": wins,
            "losses": losses_count,
        })

    # ---- Print Results ----
    print("\n" + "=" * 100)
    print("FILTER COMPARISON (position threshold = 0.10)")
    print("=" * 100)
    print(
        f"{'Filter':<22} | {'Trades':>6} | {'Return%':>8} | {'MaxDD%':>8} | "
        f"{'SSR':>9} | {'WinRate':>7} | {'W/L':>5} | {'Flat%':>6}"
    )
    print("-" * 100)

    baseline = rows[0]
    best_ssr_row = max(rows[1:], key=lambda r: r["ssr"]) if len(rows) > 1 else rows[0]

    for r in rows:
        marker = " <--" if r is best_ssr_row and r["type"] != "none" else ""
        print(
            f"  {r['label']:<20} | {r['trades']:6d} | "
            f"{r['return_pct']:+7.2f}% | {r['max_dd_pct']:+7.2f}% | "
            f"{r['ssr']:9.6f} | {r['win_rate']:5.1f}%  | "
            f"{r['wl_ratio']:5.2f} | {r['flat_pct']:5.1f}%{marker}"
        )

        # Separator between filter types
        if r["label"].startswith("ER p=20") and r["label"].endswith("t=0.35"):
            print("-" * 100)
        elif r["label"] == "ADX p=14 t=30":
            print("-" * 100)
        elif r["label"].startswith("ADX+DI") and "s=10" in r["label"]:
            print("-" * 100)

    print("-" * 100)

    # Summary: best filter vs baseline
    print(f"\n  Best filter: {best_ssr_row['label']} (SSR = {best_ssr_row['ssr']:.6f})")

    if best_ssr_row["type"] != "none":
        trade_reduction = (1 - best_ssr_row["trades"] / baseline["trades"]) * 100 if baseline["trades"] > 0 else 0
        dd_improvement = baseline["max_dd_pct"] - best_ssr_row["max_dd_pct"]
        ret_change = best_ssr_row["return_pct"] - baseline["return_pct"]

        print(f"\n  vs baseline (no filter):")
        print(f"    Trades:   {baseline['trades']} -> {best_ssr_row['trades']} "
              f"({trade_reduction:+.0f}% fewer)")
        print(f"    MaxDD:    {baseline['max_dd_pct']:+.2f}% -> {best_ssr_row['max_dd_pct']:+.2f}% "
              f"({dd_improvement:+.2f}% better)")
        print(f"    Return:   {baseline['return_pct']:+.2f}% -> {best_ssr_row['return_pct']:+.2f}% "
              f"({ret_change:+.2f}%)")
        print(f"    Win rate: {baseline['win_rate']:.1f}% -> {best_ssr_row['win_rate']:.1f}%")
        print(f"    Flat%:    {baseline['flat_pct']:.1f}% -> {best_ssr_row['flat_pct']:.1f}%")

    # Best per filter type
    print("\n  Best per filter type:")
    for ftype in ["er", "adx", "adx_di", "chop"]:
        type_rows = [r for r in rows if r["type"] == ftype]
        if type_rows:
            best = max(type_rows, key=lambda r: r["ssr"])
            improvement = best["ssr"] - baseline["ssr"]
            sign = "+" if improvement > 0 else ""
            print(f"    {ftype.upper():>6}: {best['label']:<20} SSR={best['ssr']:.6f} ({sign}{improvement:.6f})")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_filter_backtest.py <csv> [timeframe]")
        print("  timeframe: 5min, 15min, 30min, 1h (omit for raw data)")
        print("\nExamples:")
        print("  .venv/bin/python scripts/run_filter_backtest.py data/futures/NQ_1min_IB.csv 15min")
        print("  .venv/bin/python scripts/run_filter_backtest.py data/futures/NQ_in_15_minute.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(filepath, timeframe)
