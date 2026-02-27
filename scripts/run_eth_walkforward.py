#!/usr/bin/env python3
"""
Walk-forward validation for GA-MSSR on ETH/USD with trade-size sweep.

Runs the walk-forward once, then scales the OOS log returns to different
trade sizes to find the optimal risk per trade for a given account size.
Includes commission impact for both market (taker) and limit (maker) orders.

Usage:
    .venv/bin/python scripts/run_eth_walkforward.py
    .venv/bin/python scripts/run_eth_walkforward.py data/futures/ETHUSD_1min.csv
    .venv/bin/python scripts/run_eth_walkforward.py data/futures/ETHUSD_1min.csv --capital 5000
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA = "data/futures/ETHUSD_1min.csv"
TIMEFRAME = "15min"
BARS_PER_DAY = 96  # crypto 24/7, 15-min bars
TRAIN_DAYS = 20
TEST_DAYS = 5
TRADE_SIZES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
MAX_DD_PCT_LIMIT = 0.20  # recommend sizes where DD < 20% of capital

# Bybit perpetual futures fees (non-VIP)
TAKER_FEE = 0.00055   # 0.055% — market orders
MAKER_FEE = 0.00020   # 0.020% — limit orders


def compute_metrics(oos_returns: np.ndarray, avg_price: float,
                    trade_size: float, capital: float,
                    num_trades: int, fee_rate: float):
    """Convert log returns to dollar P&L metrics for a given trade size."""
    # Gross dollar P&L per bar = log_return * price * qty
    dollar_pnl = oos_returns * avg_price * trade_size
    gross_pnl = float(np.sum(dollar_pnl))

    # Commission: fee_rate * notional * num_trades (each trade = open or close)
    notional_per_trade = avg_price * trade_size
    total_fees = fee_rate * notional_per_trade * num_trades

    # Net P&L after fees
    # Distribute fee drag proportionally across bars for DD calculation
    fee_per_bar = total_fees / len(oos_returns) if len(oos_returns) > 0 else 0
    net_pnl_per_bar = dollar_pnl - fee_per_bar
    cum_pnl = np.cumsum(net_pnl_per_bar)

    # Max drawdown
    running_peak = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_peak
    max_dd = float(np.min(drawdowns))

    net_pnl = gross_pnl - total_fees
    max_dd_pct = abs(max_dd) / capital if capital > 0 else 0

    # Profit factor (on gross, fees shown separately)
    gains = dollar_pnl[dollar_pnl > 0].sum()
    losses = abs(dollar_pnl[dollar_pnl < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    # Return on capital
    roc = net_pnl / capital * 100

    # Return-to-drawdown ratio
    ret_dd_ratio = abs(net_pnl / max_dd) if max_dd != 0 else float("inf")

    return {
        "trade_size": trade_size,
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "roc": roc,
        "profit_factor": profit_factor,
        "ret_dd_ratio": ret_dd_ratio,
    }


def print_sweep(title: str, results: list, avg_price: float, capital: float):
    """Print a formatted trade-size sweep table."""
    print(f"\n  {title}")
    header = (f"  {'Size':>6} {'Notional':>9} {'Gross':>9} {'Fees':>8} "
              f"{'Net P&L':>9} {'Max DD':>9} {'DD%':>6} "
              f"{'ROC%':>7} {'PF':>5} {'R/DD':>5}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in results:
        notional = m["trade_size"] * avg_price
        flag = " *" if m["max_dd_pct"] <= MAX_DD_PCT_LIMIT else ""
        print(
            f"  {m['trade_size']:6.2f}"
            f"  ${notional:7,.0f}"
            f"  ${m['gross_pnl']:7,.0f}"
            f"  ${m['total_fees']:6,.0f}"
            f"  ${m['net_pnl']:7,.0f}"
            f"  ${m['max_dd']:7,.0f}"
            f"  {m['max_dd_pct']*100:4.1f}%"
            f"  {m['roc']:5.1f}%"
            f"  {m['profit_factor']:4.1f}"
            f"  {m['ret_dd_ratio']:4.1f}{flag}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="GA-MSSR ETH Walk-Forward with Trade Size Sweep")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA,
                        help="Path to ETH OHLCV CSV")
    parser.add_argument("--capital", type=float, default=2000,
                        help="Starting capital in USD (default: 2000)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated trade sizes, e.g. '0.05,0.1,0.2'")
    args = parser.parse_args()

    capital = args.capital
    trade_sizes = (
        [float(s) for s in args.sizes.split(",")]
        if args.sizes else TRADE_SIZES
    )

    print("=" * 70)
    print(f"  GA-MSSR ETH Walk-Forward  |  Capital: ${capital:,.0f}")
    print("=" * 70)

    # ── Load & Resample ──────────────────────────────────────────────
    print(f"\nLoading {args.data}...")
    df = build_denoised_dataset(args.data, denoise_columns="close",
                                timeframe=TIMEFRAME)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    train_size = BARS_PER_DAY * TRAIN_DAYS
    test_size = BARS_PER_DAY * TEST_DAYS
    max_folds = (len(df) - train_size) // test_size

    print(f"  Timeframe:    {TIMEFRAME} ({BARS_PER_DAY} bars/day)")
    print(f"  Train:        {TRAIN_DAYS} days ({train_size:,} bars)")
    print(f"  Test:         {TEST_DAYS} days ({test_size:,} bars)")
    print(f"  Max folds:    ~{max_folds}")

    # ── Run Walk-Forward (once) ──────────────────────────────────────
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    print(f"\nRunning walk-forward (pop={ga_config.sol_per_pop}, "
          f"gens={ga_config.num_generations})...")
    print("-" * 70)

    t0 = time.time()
    ga = GAMSSR(ga_config)
    result = ga.walk_forward(
        df,
        train_size=train_size,
        test_size=test_size,
        periods=[3, 7, 15, 27],
    )
    elapsed = time.time() - t0

    if result.num_folds == 0:
        print("ERROR: No folds completed. Check data length.")
        sys.exit(1)

    # ── Aggregate OOS Returns ────────────────────────────────────────
    all_oos = pd.concat(result.oos_returns).values
    avg_price = float(df["close_raw"].mean() if "close_raw" in df.columns
                      else df["close"].mean())

    # Position changes → trade count
    all_positions = np.sign(all_oos)
    position_changes = int(np.sum(np.diff(all_positions) != 0))
    oos_days = len(all_oos) / BARS_PER_DAY
    trades_per_day = position_changes / oos_days if oos_days > 0 else 0
    # Each position change = close old + open new = 2 fills
    total_fills = position_changes * 2

    # Per-fold stats
    ssr_arr = np.array(result.oos_ssr)
    positive_folds = sum(1 for r in result.oos_returns if float(r.sum()) > 0)

    print(f"\n  Folds:          {result.num_folds} "
          f"({positive_folds} positive, "
          f"{result.num_folds - positive_folds} negative)")
    print(f"  Aggregate SSR:  {result.aggregate_ssr:.4f}")
    print(f"  Avg price:      ${avg_price:,.2f}")
    print(f"  OOS bars:       {len(all_oos):,} ({oos_days:.0f} days)")
    print(f"  Trades/day:     ~{trades_per_day:.0f}")
    print(f"  Total fills:    {total_fills:,} over {oos_days:.0f} days")
    print(f"  Time:           {elapsed:.1f}s")

    # ── Per-Fold Breakdown ───────────────────────────────────────────
    print(f"\n{'Fold':>6} {'OOS SSR':>10} {'Return':>10}")
    print("-" * 30)
    for i, (ssr_val, oos_ret) in enumerate(
        zip(result.oos_ssr, result.oos_returns), 1
    ):
        fold_ret = float(oos_ret.sum())
        print(f"  {i:4d}   {ssr_val:9.4f}   {fold_ret:9.4f}")

    print(f"\n  Mean SSR:   {ssr_arr.mean():.4f}")
    print(f"  Median SSR: {np.median(ssr_arr):.4f}")
    print(f"  Min SSR:    {ssr_arr.min():.4f}")
    print(f"  Max SSR:    {ssr_arr.max():.4f}")

    # ── Trade Size Sweep ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRADE SIZE SWEEP  |  Capital: ${capital:,.0f}  |  Avg ETH: ${avg_price:,.2f}")
    print(f"  Bybit fees — Taker: {TAKER_FEE*100:.3f}%  |  Maker: {MAKER_FEE*100:.3f}%")
    print(f"  Total fills: {total_fills:,}")
    print(f"{'=' * 70}")

    # Market orders (taker)
    taker_results = []
    for size in trade_sizes:
        m = compute_metrics(all_oos, avg_price, size, capital,
                            total_fills, TAKER_FEE)
        taker_results.append(m)

    print_sweep(f"MARKET ORDERS (taker {TAKER_FEE*100:.3f}%)", taker_results,
                avg_price, capital)

    # Limit orders (maker)
    maker_results = []
    for size in trade_sizes:
        m = compute_metrics(all_oos, avg_price, size, capital,
                            total_fills, MAKER_FEE)
        maker_results.append(m)

    print_sweep(f"LIMIT ORDERS (maker {MAKER_FEE*100:.3f}%)", maker_results,
                avg_price, capital)

    # ── Fee Savings Summary ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FEE SAVINGS: LIMIT vs MARKET")
    print(f"{'=' * 70}")
    print(f"\n  {'Size':>6} {'Market Fee':>11} {'Limit Fee':>11} {'Saved':>9} {'Net Diff':>10}")
    print("  " + "-" * 50)
    for t, mk in zip(taker_results, maker_results):
        saved = t["total_fees"] - mk["total_fees"]
        net_diff = mk["net_pnl"] - t["net_pnl"]
        print(f"  {t['trade_size']:6.2f}"
              f"  ${t['total_fees']:9,.2f}"
              f"  ${mk['total_fees']:9,.2f}"
              f"  ${saved:7,.2f}"
              f"  +${net_diff:7,.2f}")

    # ── Recommendation ───────────────────────────────────────────────
    # Use limit order results for recommendation
    safe = [r for r in maker_results if r["max_dd_pct"] <= MAX_DD_PCT_LIMIT]
    if safe:
        best = max(safe, key=lambda r: r["trade_size"])
    else:
        best = min(maker_results, key=lambda r: r["max_dd_pct"])

    # Also find best market order option
    safe_taker = [r for r in taker_results if r["max_dd_pct"] <= MAX_DD_PCT_LIMIT]
    best_taker = (max(safe_taker, key=lambda r: r["trade_size"])
                  if safe_taker
                  else min(taker_results, key=lambda r: r["max_dd_pct"]))

    print(f"\n{'=' * 70}")
    print(f"  RECOMMENDATION (with limit orders)")
    print(f"{'=' * 70}")
    print(f"\n  Trade size:     {best['trade_size']:.2f} ETH "
          f"(${best['trade_size'] * avg_price:,.0f} notional)")
    print(f"  Gross P&L:      ${best['gross_pnl']:,.2f}")
    print(f"  Fees:           ${best['total_fees']:,.2f}")
    print(f"  Net P&L:        ${best['net_pnl']:,.2f} "
          f"({best['roc']:.1f}% return on ${capital:,.0f})")
    print(f"  Max drawdown:   ${best['max_dd']:,.2f} "
          f"({best['max_dd_pct']*100:.1f}% of account)")
    print(f"  Profit factor:  {best['profit_factor']:.1f}")
    print(f"  Ret/DD ratio:   {best['ret_dd_ratio']:.1f}")

    if best_taker["trade_size"] != best["trade_size"]:
        print(f"\n  With market orders, max safe size: "
              f"{best_taker['trade_size']:.2f} ETH")

    if best["max_dd_pct"] > MAX_DD_PCT_LIMIT:
        print(f"\n  WARNING: Even smallest size exceeds "
              f"{MAX_DD_PCT_LIMIT*100:.0f}% DD limit.")
        print(f"  Consider increasing capital or reducing leverage.")

    print()


if __name__ == "__main__":
    main()
