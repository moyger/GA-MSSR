#!/usr/bin/env python3
"""
Threshold sweep backtest for GA-MSSR.

Trains one model, then tests different position_threshold values to find the
sweet spot between signal sensitivity and noise filtering.

Usage:
    .venv/bin/python scripts/run_threshold_backtest.py data/futures/NQ_1min_IB.csv 15min
    .venv/bin/python scripts/run_threshold_backtest.py data/futures/ETHUSD_1min.csv 15min
    .venv/bin/python scripts/run_threshold_backtest.py data/futures/NQ_in_15_minute.csv
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def _count_trades(positions: np.ndarray) -> int:
    """Count position sign changes (each change = 1 trade)."""
    signs = np.sign(positions)
    return int(np.sum(np.diff(signs) != 0))


def _trade_stats(positions: np.ndarray, logr: np.ndarray):
    """Compute per-trade win/loss statistics."""
    signs = np.sign(positions)
    changes = np.where(np.diff(signs) != 0)[0] + 1
    # Add start and end
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
            continue  # flat segment, not a trade
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


def main(filepath: str, timeframe: str = None):
    print("=" * 70)
    print("GA-MSSR Position Threshold Sweep")
    print("=" * 70)

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

    # ---- Phase 4: Threshold Sweep ----
    print("\n[4/4] Sweeping thresholds...")

    logr = features["logr"].values.astype(np.float64)
    signal_cols = [c for c in features.columns if c.startswith("rule_")]
    signals = features[signal_cols].values.astype(np.float64)

    # Raw continuous position (same for all thresholds)
    raw_position = signals @ result.best_weights

    ssr_calc = SSR()
    rows = []

    for threshold in THRESHOLDS:
        # Discretize
        pos = np.where(raw_position > threshold, 1.0,
              np.where(raw_position < -threshold, -1.0, 0.0))

        # Returns
        port_ret = pos * logr
        total_bars = len(pos)
        flat_bars = int(np.sum(pos == 0))
        flat_pct = flat_bars / total_bars * 100

        # Metrics
        metrics = ssr_calc.calculate(port_ret)
        trades = _count_trades(pos)
        win_rate, wl_ratio, wins, losses = _trade_stats(pos, logr)

        rows.append({
            "threshold": threshold,
            "trades": trades,
            "return_pct": metrics.total_return * 100,
            "max_dd_pct": metrics.max_drawdown * 100,
            "ssr": metrics.ssr,
            "win_rate": win_rate,
            "wl_ratio": wl_ratio,
            "flat_pct": flat_pct,
            "wins": wins,
            "losses": losses,
        })

    # ---- Print Results ----
    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)
    print(
        f"{'Thresh':>7} | {'Trades':>6} | {'Return%':>8} | {'MaxDD%':>8} | "
        f"{'SSR':>9} | {'WinRate':>7} | {'W/L':>5} | {'Flat%':>6}"
    )
    print("-" * 70)

    best_ssr_row = max(rows, key=lambda r: r["ssr"])

    for r in rows:
        marker = " <--" if r is best_ssr_row and r["threshold"] > 0 else ""
        print(
            f"  {r['threshold']:5.2f} | {r['trades']:6d} | "
            f"{r['return_pct']:+7.2f}% | {r['max_dd_pct']:+7.2f}% | "
            f"{r['ssr']:9.6f} | {r['win_rate']:5.1f}%  | "
            f"{r['wl_ratio']:5.2f} | {r['flat_pct']:5.1f}%{marker}"
        )

    print("-" * 70)
    print(f"  Best SSR at threshold = {best_ssr_row['threshold']:.2f} "
          f"(SSR = {best_ssr_row['ssr']:.6f})")

    # ---- Detail: threshold=0 vs best ----
    baseline = rows[0]
    best = best_ssr_row
    if best["threshold"] > 0:
        trade_reduction = (1 - best["trades"] / baseline["trades"]) * 100
        dd_improvement = baseline["max_dd_pct"] - best["max_dd_pct"]
        ret_change = best["return_pct"] - baseline["return_pct"]

        print(f"\n  vs baseline (threshold=0.0):")
        print(f"    Trades:   {baseline['trades']} -> {best['trades']} "
              f"({trade_reduction:+.0f}% fewer)")
        print(f"    MaxDD:    {baseline['max_dd_pct']:+.2f}% -> {best['max_dd_pct']:+.2f}% "
              f"({dd_improvement:+.2f}% better)")
        print(f"    Return:   {baseline['return_pct']:+.2f}% -> {best['return_pct']:+.2f}% "
              f"({ret_change:+.2f}%)")
        print(f"    Win rate: {baseline['win_rate']:.1f}% -> {best['win_rate']:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_threshold_backtest.py <csv> [timeframe]")
        print("  timeframe: 5min, 15min, 30min, 1h (omit for raw data)")
        print("\nExamples:")
        print("  .venv/bin/python scripts/run_threshold_backtest.py data/futures/NQ_1min_IB.csv 15min")
        print("  .venv/bin/python scripts/run_threshold_backtest.py data/futures/ETHUSD_1min.csv 15min")
        sys.exit(1)

    filepath = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else None
    main(filepath, timeframe)
