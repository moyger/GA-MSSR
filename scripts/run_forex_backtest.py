#!/usr/bin/env python3
"""
Forex backtest with threshold tuning for GA-MSSR.

Runs walk-forward validation + threshold sweep on EURUSD and AUDUSD,
then produces a side-by-side comparison table.

Usage:
    .venv/bin/python scripts/run_forex_backtest.py
    .venv/bin/python scripts/run_forex_backtest.py --pair eurusd
    .venv/bin/python scripts/run_forex_backtest.py --pair audusd
    .venv/bin/python scripts/run_forex_backtest.py --capital 100000
    .venv/bin/python scripts/run_forex_backtest.py --capital 100000 --daily-limit 2000
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from data.forex_config import EURUSD, AUDUSD, ForexConfig
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

# ── Constants ─────────────────────────────────────────────────────────
TIMEFRAME = "15min"
BARS_PER_DAY = 96   # forex 24h at 15min
TRAIN_DAYS = 20
TEST_DAYS = 5
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
GA_PERIODS = [3, 7, 15, 27]

PAIR_DATA = {
    "eurusd": {
        "config": EURUSD,
        "data": "data/forex/EURUSD_5min.csv",
    },
    "audusd": {
        "config": AUDUSD,
        "data": "data/forex/AUDUSD_5min.csv",
    },
}


# ── Helpers (from run_threshold_backtest.py) ──────────────────────────

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


# ── Threshold Sweep ───────────────────────────────────────────────────

def run_threshold_sweep(df, config: ForexConfig, ga_config: GAMSSRConfig,
                        lots: int = 1):
    """Train one model, sweep thresholds. Returns list of result dicts."""
    print(f"\n  [Threshold Sweep] Training 16 rule parameters...")
    t0 = time.time()
    rule_params = train_rule_params(df, periods=GA_PERIODS)
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"  [Threshold Sweep] Generating features & running GA...")
    features = get_rule_features(df, rule_params)
    t0 = time.time()
    ga = GAMSSR(ga_config)
    result = ga.fit(features)
    print(f"    Done in {time.time() - t0:.1f}s | SSR={result.best_fitness:.6f}")
    print(f"    Weights: {np.round(result.best_weights, 3)}")

    logr = features["logr"].values.astype(np.float64)
    signal_cols = [c for c in features.columns if c.startswith("rule_")]
    signals = features[signal_cols].values.astype(np.float64)
    raw_position = signals @ result.best_weights

    ssr_calc = SSR()
    rows = []

    for threshold in THRESHOLDS:
        pos = np.where(raw_position > threshold, 1.0,
              np.where(raw_position < -threshold, -1.0, 0.0))

        port_ret = pos * logr
        total_bars = len(pos)
        flat_bars = int(np.sum(pos == 0))
        flat_pct = flat_bars / total_bars * 100

        metrics = ssr_calc.calculate(port_ret)
        trades = _count_trades(pos)
        win_rate, wl_ratio, wins, losses = _trade_stats(pos, logr)

        # Dollar PnL estimate (scaled by lots)
        avg_price = df["close"].mean()
        dollar_pnl = metrics.total_return * avg_price * config.point_value * lots
        dollar_dd = metrics.max_drawdown * avg_price * config.point_value * lots

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
            "dollar_pnl": dollar_pnl,
            "dollar_dd": dollar_dd,
        })

    return rows


# ── Walk-Forward Simulation ──────────────────────────────────────────

def run_walk_forward_sim(df, config: ForexConfig, ga_config: GAMSSRConfig,
                         best_threshold: float, capital: float = 0,
                         daily_limit: float = 0, lots: int = 1):
    """Walk-forward with bar-by-bar dollar PnL. Returns summary dict."""
    train_size = BARS_PER_DAY * TRAIN_DAYS
    test_size = BARS_PER_DAY * TEST_DAYS
    n = len(df)

    cap_label = f"  Capital: ${capital:,.0f}" if capital > 0 else ""
    dl_label = f"  Daily Limit: ${daily_limit:,.0f}" if daily_limit > 0 else ""
    print(f"\n  [Walk-Forward] Train={TRAIN_DAYS}d ({train_size} bars) "
          f"Test={TEST_DAYS}d ({test_size} bars) Threshold={best_threshold:.2f}"
          f"{cap_label}{dl_label}")

    ssr_calc = SSR()
    all_pnl_curves = []
    fold_summaries = []
    all_positions = []
    all_returns = []
    days_stopped = 0

    start = 0
    fold_num = 0
    while start + train_size + test_size <= n:
        fold_num += 1
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]

        # Train
        rule_params = train_rule_params(train_df, periods=GA_PERIODS)
        train_features = get_rule_features(train_df, rule_params)
        test_features = get_rule_features(test_df, rule_params)

        if len(train_features) < 10 or len(test_features) < 5:
            start += test_size
            continue

        ga = GAMSSR(ga_config)
        ga_result = ga.fit(train_features)
        weights = ga_result.best_weights

        # Predict on test data with threshold
        logr = test_features["logr"].values
        signal_cols = [c for c in test_features.columns if c.startswith("rule_")]
        signals = test_features[signal_cols].values
        raw_pos = signals @ weights
        positions = np.where(raw_pos > best_threshold, 1.0,
                    np.where(raw_pos < -best_threshold, -1.0, 0.0))

        # Dollar PnL (bar-by-bar) with daily loss limit
        raw_col = "close_raw" if "close_raw" in test_df.columns else "close"
        prices = test_df[raw_col].iloc[-len(test_features):].values

        bar_pnl = np.zeros(len(positions))
        total_fees = 0.0
        daily_pnl = 0.0
        bar_in_day = 0
        daily_stopped = False
        fold_days_stopped = 0
        actual_positions = positions.copy()

        for i in range(1, len(positions)):
            # New day reset
            if bar_in_day >= BARS_PER_DAY:
                bar_in_day = 0
                daily_pnl = 0.0
                daily_stopped = False

            # If daily stopped, flatten
            if daily_stopped:
                actual_positions[i] = 0
                bar_in_day += 1
                continue

            # PnL from position held (scaled by lots)
            point_val = config.point_value * lots
            comm = config.commission_rt * lots
            if actual_positions[i - 1] != 0:
                price_change = prices[i] - prices[i - 1]
                pnl = actual_positions[i - 1] * price_change * point_val
                bar_pnl[i] += pnl
                daily_pnl += pnl

            # Fee on position change (scaled by lots)
            if actual_positions[i] != actual_positions[i - 1]:
                if actual_positions[i - 1] != 0:
                    bar_pnl[i] -= comm
                    total_fees += comm
                    daily_pnl -= comm
                if actual_positions[i] != 0:
                    bar_pnl[i] -= comm
                    total_fees += comm
                    daily_pnl -= comm

            # Daily loss limit check
            if daily_limit > 0 and daily_pnl <= -daily_limit:
                daily_stopped = True
                fold_days_stopped += 1
                # Flatten position
                if actual_positions[i] != 0:
                    bar_pnl[i] -= comm
                    total_fees += comm
                    actual_positions[i] = 0

            bar_in_day += 1

        days_stopped += fold_days_stopped
        port_return = actual_positions * logr

        net_pnl = float(np.sum(bar_pnl))
        cum_pnl = np.cumsum(bar_pnl)
        running_peak = np.maximum.accumulate(cum_pnl)
        max_dd = float(np.min(cum_pnl - running_peak))

        all_pnl_curves.append(cum_pnl)
        all_positions.extend(actual_positions.tolist())
        all_returns.extend(port_return.tolist())

        fold_trades = _count_trades(actual_positions)

        oos_ssr = ssr_calc.calculate(port_return)
        fold_summaries.append({
            "fold": fold_num,
            "net_pnl": net_pnl,
            "max_dd": max_dd,
            "ssr": oos_ssr.ssr,
            "return_pct": float(port_return.sum()) * 100,
            "trades": fold_trades,
            "fees": total_fees,
            "train_ssr": ga_result.best_fitness,
            "days_stopped": fold_days_stopped,
        })

        status = "+" if net_pnl > 0 else "-"
        stop_label = f"  Stop={fold_days_stopped}d" if daily_limit > 0 else ""
        print(f"    Fold {fold_num:2d}: PnL=${net_pnl:8.2f}  DD=${max_dd:8.2f}  "
              f"SSR={oos_ssr.ssr:8.4f}  Trades={fold_trades:3d}{stop_label}  [{status}]")

        start += test_size

    if not fold_summaries:
        print("    ERROR: No folds completed (insufficient data).")
        return {"net_pnl": 0, "max_dd": 0, "num_folds": 0, "capital": capital}

    # Aggregate
    total_net_pnl = sum(f["net_pnl"] for f in fold_summaries)
    total_fees = sum(f["fees"] for f in fold_summaries)
    total_trades = sum(f["trades"] for f in fold_summaries)
    positive_folds = sum(1 for f in fold_summaries if f["net_pnl"] > 0)

    # Global equity curve
    global_curve = np.concatenate(all_pnl_curves)
    running_peak = np.maximum.accumulate(global_curve)
    global_max_dd = float(np.min(global_curve - running_peak))

    # OOS returns
    all_ret = np.array(all_returns)
    oos_ssr_agg = ssr_calc.calculate(all_ret)

    # Daily PnL stats
    total_bars = len(all_positions)
    oos_days = total_bars / BARS_PER_DAY
    trades_per_day = total_trades / oos_days if oos_days > 0 else 0

    # Win rate on positions
    win_rate, wl_ratio, wins, losses = _trade_stats(
        np.array(all_positions), all_ret)

    # Capital-relative metrics
    roc = (total_net_pnl / capital * 100) if capital > 0 else 0
    dd_pct = (abs(global_max_dd) / capital * 100) if capital > 0 else 0
    final_balance = capital + total_net_pnl if capital > 0 else total_net_pnl

    return {
        "net_pnl": total_net_pnl,
        "total_fees": total_fees,
        "max_dd": global_max_dd,
        "num_folds": fold_num,
        "positive_folds": positive_folds,
        "total_trades": total_trades,
        "trades_per_day": trades_per_day,
        "oos_ssr": oos_ssr_agg.ssr,
        "total_return_pct": float(all_ret.sum()) * 100,
        "win_rate": win_rate,
        "wl_ratio": wl_ratio,
        "oos_days": oos_days,
        "fold_summaries": fold_summaries,
        "equity_curve": global_curve,
        "capital": capital,
        "roc": roc,
        "dd_pct": dd_pct,
        "final_balance": final_balance,
        "days_stopped": days_stopped,
    }


# ── Per-Pair Backtest ─────────────────────────────────────────────────

def run_pair_backtest(pair_name: str, ga_config: GAMSSRConfig,
                      capital: float = 0, daily_limit: float = 0,
                      lots: int = 1):
    """Run full backtest for one pair. Returns results dict."""
    pair_info = PAIR_DATA[pair_name]
    config = pair_info["config"]
    data_path = pair_info["data"]

    pip_val = config.pip_value * lots
    comm_rt = config.commission_rt * lots

    print(f"\n{'='*70}")
    lot_label = (f"{lots} mini lot{'s' if lots > 1 else ''} "
                 f"({lots * 10}K units)")
    print(f"  {config.symbol} — {lot_label} — {TIMEFRAME}")
    print(f"  Pip value: ${pip_val:.2f}/pip  |  "
          f"Commission: ${comm_rt:.2f}/RT  |  "
          f"Spread: ~{config.spread_cost_pips} pips")
    if capital > 0:
        print(f"  Capital: ${capital:,.0f}  |  Daily Limit: ${daily_limit:,.0f}")
    print(f"{'='*70}")

    # Load and denoise
    print(f"\n  Loading {data_path}...")
    df = build_denoised_dataset(data_path, denoise_columns="close",
                                timeframe=TIMEFRAME)
    print(f"  {len(df):,} bars | {df.index[0]} to {df.index[-1]}")

    # Phase 1: Threshold sweep (scaled by lots)
    threshold_results = run_threshold_sweep(df, config, ga_config, lots=lots)

    # Print threshold table
    print(f"\n  {'Thresh':>7} | {'Trades':>6} | {'Return%':>8} | {'MaxDD%':>8} | "
          f"{'SSR':>9} | {'WinRate':>7} | {'W/L':>5} | {'Flat%':>6} | {'$ PnL':>9}")
    print("  " + "-" * 85)

    best_ssr_row = max(threshold_results, key=lambda r: r["ssr"])
    for r in threshold_results:
        marker = " <--" if r is best_ssr_row and r["threshold"] > 0 else ""
        print(
            f"    {r['threshold']:5.2f} | {r['trades']:6d} | "
            f"{r['return_pct']:+7.2f}% | {r['max_dd_pct']:+7.2f}% | "
            f"{r['ssr']:9.6f} | {r['win_rate']:5.1f}%  | "
            f"{r['wl_ratio']:5.2f} | {r['flat_pct']:5.1f}% | "
            f"${r['dollar_pnl']:8.2f}{marker}"
        )

    best_threshold = best_ssr_row["threshold"]
    print(f"\n  Best threshold: {best_threshold:.2f} (SSR={best_ssr_row['ssr']:.6f})")

    # Phase 2: Walk-forward with best threshold
    wf_results = run_walk_forward_sim(df, config, ga_config, best_threshold,
                                      capital=capital, daily_limit=daily_limit,
                                      lots=lots)

    # Print walk-forward summary
    if wf_results["num_folds"] > 0:
        print(f"\n  --- Walk-Forward Summary ({config.symbol}) ---")
        if capital > 0:
            print(f"  Starting capital:  ${capital:,.2f}")
            print(f"  Final balance:     ${wf_results['final_balance']:,.2f}")
            print(f"  Return on capital: {wf_results['roc']:+.2f}%")
        print(f"  Net PnL:           ${wf_results['net_pnl']:,.2f}")
        print(f"  Total fees:        ${wf_results['total_fees']:,.2f}")
        print(f"  Max drawdown:      ${wf_results['max_dd']:,.2f}"
              + (f" ({wf_results['dd_pct']:.2f}% of capital)" if capital > 0 else ""))
        if daily_limit > 0:
            print(f"  Days stopped:      {wf_results['days_stopped']}")
        print(f"  OOS SSR:           {wf_results['oos_ssr']:.6f}")
        print(f"  Return:            {wf_results['total_return_pct']:+.2f}%")
        print(f"  Folds:             {wf_results['num_folds']} "
              f"({wf_results['positive_folds']} profitable)")
        print(f"  Total trades:      {wf_results['total_trades']:,}")
        print(f"  Trades/day:        {wf_results['trades_per_day']:.1f}")
        print(f"  Win rate:          {wf_results['win_rate']:.1f}%")
        print(f"  W/L ratio:         {wf_results['wl_ratio']:.2f}")

    return {
        "pair": config.symbol,
        "config": config,
        "bars": len(df),
        "lots": lots,
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "threshold_results": threshold_results,
        "best_threshold": best_ssr_row,
        "walk_forward": wf_results,
    }


# ── Comparison ────────────────────────────────────────────────────────

def print_comparison(results: list[dict]):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  FOREX COMPARISON — Side by Side")
    print(f"{'='*70}")

    col_w = 18
    header = f"  {'Metric':<28}"
    for r in results:
        header += f" {r['pair']:>{col_w}}"
    print(header)
    print("  " + "-" * (28 + (col_w + 1) * len(results)))

    metrics = [
        ("Date range",        lambda r: r["date_range"]),
        ("Bars (15min)",       lambda r: f"{r['bars']:,}"),
        ("",                   lambda r: ""),
        ("--- Threshold Sweep ---", lambda r: ""),
        ("Best Threshold",     lambda r: f"{r['best_threshold']['threshold']:.2f}"),
        ("SSR (best)",         lambda r: f"{r['best_threshold']['ssr']:.6f}"),
        ("Return % (best)",    lambda r: f"{r['best_threshold']['return_pct']:+.2f}%"),
        ("Max DD % (best)",    lambda r: f"{r['best_threshold']['max_dd_pct']:+.2f}%"),
        ("Win Rate (best)",    lambda r: f"{r['best_threshold']['win_rate']:.1f}%"),
        ("W/L Ratio (best)",   lambda r: f"{r['best_threshold']['wl_ratio']:.2f}"),
        ("Trades (best)",      lambda r: f"{r['best_threshold']['trades']:,}"),
        ("$ PnL (best)",       lambda r: f"${r['best_threshold']['dollar_pnl']:,.2f}"),
        ("Flat % (best)",      lambda r: f"{r['best_threshold']['flat_pct']:.1f}%"),
        ("",                   lambda r: ""),
        ("--- Walk-Forward ---", lambda r: ""),
        ("Net PnL ($)",        lambda r: f"${r['walk_forward']['net_pnl']:,.2f}"),
        ("Max DD ($)",         lambda r: f"${r['walk_forward']['max_dd']:,.2f}"),
        ("OOS SSR",            lambda r: f"{r['walk_forward']['oos_ssr']:.6f}"),
        ("OOS Return %",       lambda r: f"{r['walk_forward']['total_return_pct']:+.2f}%"),
        ("Folds (profitable)", lambda r: f"{r['walk_forward']['num_folds']} "
                                         f"({r['walk_forward']['positive_folds']})"),
        ("Total Trades",       lambda r: f"{r['walk_forward']['total_trades']:,}"),
        ("Trades/day",         lambda r: f"{r['walk_forward']['trades_per_day']:.1f}"),
        ("Win Rate",           lambda r: f"{r['walk_forward']['win_rate']:.1f}%"),
        ("W/L Ratio",          lambda r: f"{r['walk_forward']['wl_ratio']:.2f}"),
    ]

    # Add capital metrics if capital was set
    cap = results[0]["walk_forward"].get("capital", 0)
    if cap > 0:
        metrics.extend([
            ("",                   lambda r: ""),
            ("--- Account ---",    lambda r: ""),
            ("Starting Capital",   lambda r: f"${r['walk_forward']['capital']:,.0f}"),
            ("Final Balance",      lambda r: f"${r['walk_forward']['final_balance']:,.2f}"),
            ("Return on Capital",  lambda r: f"{r['walk_forward']['roc']:+.2f}%"),
            ("Max DD (% capital)", lambda r: f"{r['walk_forward']['dd_pct']:.2f}%"),
            ("Return/DD Ratio",    lambda r: f"{abs(r['walk_forward']['net_pnl'] / r['walk_forward']['max_dd']):.1f}x"
                                             if r['walk_forward']['max_dd'] != 0 else "N/A"),
            ("Days Stopped",       lambda r: f"{r['walk_forward']['days_stopped']}"),
        ])

    for label, extractor in metrics:
        if label == "":
            print()
            continue
        if label.startswith("---"):
            row = f"  {label:<28}"
            print(row)
            continue
        row = f"  {label:<28}"
        for r in results:
            row += f" {extractor(r):>{col_w}}"
        print(row)


def generate_comparison_chart(results: list[dict], output_dir: Path):
    """Generate comparison equity curve chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    num_pairs = len(results)
    fig, axes = plt.subplots(num_pairs + 1, 1, figsize=(16, 6 * (num_pairs + 1)),
                             gridspec_kw={"height_ratios": [3] * num_pairs + [2]})
    if num_pairs == 1:
        axes = [axes, plt.subplot(num_pairs + 1, 1, num_pairs + 1)]

    lots = results[0].get("lots", 1) if results else 1
    lot_label = f"{lots}x Mini Lot ({lots*10}K)" if lots > 1 else "Mini Lot"
    fig.suptitle(f"GA-MSSR Forex Backtest Comparison — {TIMEFRAME} {lot_label}",
                 fontsize=16, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    # Equity curves
    for i, r in enumerate(results):
        ax = axes[i]
        wf = r["walk_forward"]
        if "equity_curve" in wf and len(wf["equity_curve"]) > 0:
            curve = wf["equity_curve"]
            ax.plot(range(len(curve)), curve, color=colors[i % len(colors)],
                    linewidth=1.2, label=r["pair"])
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"{r['pair']} — Walk-Forward Equity (${wf['net_pnl']:,.2f} net)")
            ax.set_ylabel("PnL ($)")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            # Mark fold boundaries
            if "fold_summaries" in wf:
                offset = 0
                for fold in wf["fold_summaries"]:
                    ax.axvline(x=offset, color="orange", alpha=0.2, linestyle=":")
                    offset += BARS_PER_DAY * TEST_DAYS

    # Per-fold comparison bar chart
    ax_folds = axes[-1]
    if num_pairs > 1:
        fold_counts = [r["walk_forward"]["num_folds"] for r in results]
        max_folds = max(fold_counts) if fold_counts else 0
        width = 0.35
        for i, r in enumerate(results):
            wf = r["walk_forward"]
            if "fold_summaries" in wf:
                fold_pnls = [f["net_pnl"] for f in wf["fold_summaries"]]
                x = np.arange(len(fold_pnls)) + i * width
                bar_colors = [colors[i % len(colors)] if p > 0 else "#F44336"
                              for p in fold_pnls]
                ax_folds.bar(x, fold_pnls, width, alpha=0.7, label=r["pair"],
                             color=colors[i % len(colors)])
        ax_folds.set_title("Per-Fold OOS PnL ($) by Pair")
        ax_folds.set_xlabel("Fold")
        ax_folds.set_ylabel("PnL ($)")
        ax_folds.legend()
        ax_folds.grid(True, alpha=0.3)
        ax_folds.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = output_dir / f"forex_comparison_{TIMEFRAME}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart saved: {chart_path}")
    return chart_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GA-MSSR Forex Backtest with Threshold Tuning")
    parser.add_argument("--pair", choices=["eurusd", "audusd", "both"],
                        default="both",
                        help="Which pair(s) to backtest (default: both)")
    parser.add_argument("--capital", type=float, default=0,
                        help="Starting capital in USD (default: 0 = no tracking)")
    parser.add_argument("--daily-limit", type=float, default=0,
                        help="Max daily loss in USD (default: 0 = no limit)")
    parser.add_argument("--lots", type=int, default=1,
                        help="Number of mini lots (1=10K/$1pip, 10=100K/$10pip)")
    args = parser.parse_args()

    # Default daily limit to 1% of capital if capital set but limit not
    daily_limit = args.daily_limit
    if args.capital > 0 and daily_limit == 0:
        daily_limit = args.capital * 0.01  # 1% of capital

    lots = args.lots
    lot_label = (f"{lots} mini lot{'s' if lots > 1 else ''} "
                 f"({lots * 10}K units) — ${lots:.0f}/pip")

    print("=" * 70)
    print("  GA-MSSR Forex Backtest — Threshold Tuning + Walk-Forward")
    print(f"  Position sizing: {lot_label}")
    if args.capital > 0:
        print(f"  Capital: ${args.capital:,.0f}  |  "
              f"Daily Limit: ${daily_limit:,.0f} (1% of capital)")
    print("=" * 70)

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    pairs = ["eurusd", "audusd"] if args.pair == "both" else [args.pair]

    t0_total = time.time()
    results = []
    for pair in pairs:
        results.append(run_pair_backtest(pair, ga_config,
                                         capital=args.capital,
                                         daily_limit=daily_limit,
                                         lots=lots))

    elapsed = time.time() - t0_total

    # Comparison table
    if len(results) > 1:
        print_comparison(results)

    # Generate chart
    output_dir = Path("reports")
    generate_comparison_chart(results, output_dir)

    # Save fold details CSV
    for r in results:
        wf = r["walk_forward"]
        if "fold_summaries" in wf and wf["fold_summaries"]:
            fold_df = pd.DataFrame(wf["fold_summaries"])
            csv_path = output_dir / f"forex_{r['pair'].lower()}_folds_{TIMEFRAME}.csv"
            fold_df.to_csv(csv_path, index=False)
            print(f"Fold details saved: {csv_path}")

    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
