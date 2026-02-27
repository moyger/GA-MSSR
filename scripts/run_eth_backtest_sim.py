#!/usr/bin/env python3
"""
Realistic walk-forward simulation for GA-MSSR on ETH/USD.

Simulates the live bot's exact behavior:
- Walk-forward GA training (20-day train, 5-day test)
- Fixed trade size in ETH
- Daily loss limit with halt-until-next-day
- Per-trade commission (maker or taker)
- Bar-by-bar P&L tracking with position management

Usage:
    .venv/bin/python scripts/run_eth_backtest_sim.py
    .venv/bin/python scripts/run_eth_backtest_sim.py --size 1.0 --daily-limit 100
    .venv/bin/python scripts/run_eth_backtest_sim.py --size 0.2 --daily-limit 50
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA = "data/futures/ETHUSD_1min.csv"
TIMEFRAME = "15min"
BARS_PER_DAY = 96
TRAIN_DAYS = 20
TEST_DAYS = 5

MAKER_FEE = 0.00020   # 0.020%
TAKER_FEE = 0.00055   # 0.055%


def simulate_oos_fold(prices: np.ndarray, signals: np.ndarray,
                      trade_size: float, daily_limit: float,
                      fee_rate: float, bars_per_day: int):
    """
    Simulate bar-by-bar trading on an OOS fold with daily loss limit.

    Parameters
    ----------
    prices : array of close prices for each bar
    signals : array of target positions {-1, 0, +1} for each bar
    trade_size : ETH per trade
    daily_limit : max daily loss in USD (positive number)
    fee_rate : commission rate per fill
    bars_per_day : bars in one trading day

    Returns
    -------
    dict with simulation results
    """
    n = len(prices)
    position = 0        # current position: -1, 0, +1
    daily_pnl = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    daily_stopped = False
    bar_in_day = 0

    pnl_curve = np.zeros(n)
    trades = []
    daily_summaries = []
    current_day_trades = 0
    current_day_pnl = 0.0
    days_stopped = 0

    for i in range(n):
        price = prices[i]
        target = int(signals[i])

        # New day reset
        if bar_in_day >= bars_per_day:
            daily_summaries.append({
                "day_pnl": current_day_pnl,
                "day_trades": current_day_trades,
                "stopped": daily_stopped,
            })
            bar_in_day = 0
            daily_pnl = 0.0
            daily_stopped = False
            current_day_pnl = 0.0
            current_day_trades = 0

        # P&L from price movement (if holding a position)
        if i > 0 and position != 0:
            price_change = prices[i] - prices[i - 1]
            bar_pnl = position * price_change * trade_size
            daily_pnl += bar_pnl
            total_pnl += bar_pnl

        # Daily loss limit check
        if not daily_stopped and daily_pnl <= -daily_limit:
            daily_stopped = True
            days_stopped += 1
            # Close position if holding
            if position != 0:
                fee = fee_rate * price * trade_size
                total_fees += fee
                total_pnl -= fee
                daily_pnl -= fee
                trades.append({
                    "bar": i, "side": "close_stop",
                    "price": price, "fee": fee,
                })
                current_day_trades += 1
                position = 0

        # Skip trading if daily stopped
        if daily_stopped:
            pnl_curve[i] = total_pnl - total_fees if i == 0 else total_pnl
            bar_in_day += 1
            continue

        # Execute position change
        if target != position:
            # Close existing position
            if position != 0:
                fee = fee_rate * price * trade_size
                total_fees += fee
                total_pnl -= fee
                daily_pnl -= fee
                current_day_trades += 1

            # Open new position
            if target != 0:
                fee = fee_rate * price * trade_size
                total_fees += fee
                total_pnl -= fee
                daily_pnl -= fee
                current_day_trades += 1
                trades.append({
                    "bar": i,
                    "side": "buy" if target > 0 else "sell",
                    "price": price,
                    "fee": fee,
                })

            position = target

        pnl_curve[i] = total_pnl
        current_day_pnl = daily_pnl
        bar_in_day += 1

    # Final day
    if bar_in_day > 0:
        daily_summaries.append({
            "day_pnl": current_day_pnl,
            "day_trades": current_day_trades,
            "stopped": daily_stopped,
        })

    # Metrics
    running_peak = np.maximum.accumulate(pnl_curve)
    drawdowns = pnl_curve - running_peak
    max_dd = float(np.min(drawdowns))

    winning_days = sum(1 for d in daily_summaries if d["day_pnl"] > 0)
    losing_days = sum(1 for d in daily_summaries if d["day_pnl"] < 0)
    flat_days = sum(1 for d in daily_summaries if d["day_pnl"] == 0)

    daily_pnls = [d["day_pnl"] for d in daily_summaries]
    avg_win = np.mean([p for p in daily_pnls if p > 0]) if winning_days else 0
    avg_loss = np.mean([p for p in daily_pnls if p < 0]) if losing_days else 0

    return {
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "net_pnl": total_pnl,  # fees already subtracted
        "max_dd": max_dd,
        "pnl_curve": pnl_curve,
        "num_trades": len(trades),
        "num_days": len(daily_summaries),
        "winning_days": winning_days,
        "losing_days": losing_days,
        "flat_days": flat_days,
        "days_stopped": days_stopped,
        "avg_daily_win": avg_win,
        "avg_daily_loss": avg_loss,
        "daily_summaries": daily_summaries,
        "best_day": max(daily_pnls) if daily_pnls else 0,
        "worst_day": min(daily_pnls) if daily_pnls else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GA-MSSR ETH Realistic Backtest Simulation")
    parser.add_argument("data", nargs="?", default=DEFAULT_DATA)
    parser.add_argument("--size", type=float, default=1.0,
                        help="Trade size in ETH (default: 1.0)")
    parser.add_argument("--daily-limit", type=float, default=100.0,
                        help="Daily loss limit in USD (default: 100)")
    parser.add_argument("--capital", type=float, default=2000,
                        help="Starting capital (default: 2000)")
    parser.add_argument("--fee", type=str, default="maker",
                        choices=["maker", "taker"],
                        help="Fee type: maker (limit) or taker (market)")
    args = parser.parse_args()

    fee_rate = MAKER_FEE if args.fee == "maker" else TAKER_FEE
    fee_label = f"{'maker' if args.fee == 'maker' else 'taker'} ({fee_rate*100:.3f}%)"

    print("=" * 70)
    print(f"  GA-MSSR ETH Backtest Simulation")
    print(f"  Size: {args.size} ETH  |  Daily Limit: ${args.daily_limit:,.0f}"
          f"  |  Capital: ${args.capital:,.0f}")
    print(f"  Fees: {fee_label}")
    print("=" * 70)

    # ── Load Data ────────────────────────────────────────────────────
    print(f"\nLoading {args.data}...")
    df = build_denoised_dataset(args.data, denoise_columns="close",
                                timeframe=TIMEFRAME)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    train_size = BARS_PER_DAY * TRAIN_DAYS
    test_size = BARS_PER_DAY * TEST_DAYS

    # ── GA Config ────────────────────────────────────────────────────
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )
    periods = [3, 7, 15, 27]

    # ── Walk-Forward with Bar-by-Bar Simulation ──────────────────────
    print(f"\nRunning walk-forward simulation...")
    print(f"  Train: {TRAIN_DAYS} days ({train_size} bars)")
    print(f"  Test:  {TEST_DAYS} days ({test_size} bars)")
    print("-" * 70)

    t0 = time.time()
    all_sim_results = []
    fold_summaries = []
    n = len(df)
    start = 0
    fold_num = 0

    while start + train_size + test_size <= n:
        fold_num += 1
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]

        # Train rule params and GA on training data
        rule_params = train_rule_params(train_df, periods=periods)
        train_features = get_rule_features(train_df, rule_params)

        ga = GAMSSR(ga_config)
        ga_result = ga.fit(train_features)
        weights = ga_result.best_weights

        # Generate signals on test data
        test_features = get_rule_features(test_df, rule_params)
        signals_col = [c for c in test_features.columns if c != "logr"]
        raw_signal = test_features[signals_col].values @ weights
        positions = np.sign(raw_signal)

        # Get raw prices aligned to feature rows (features drop first bar for logr)
        raw_col = "close_raw" if "close_raw" in test_df.columns else "close"
        test_prices = test_df[raw_col].iloc[-len(test_features):].values

        # Simulate
        sim = simulate_oos_fold(
            prices=test_prices,
            signals=positions,
            trade_size=args.size,
            daily_limit=args.daily_limit,
            fee_rate=fee_rate,
            bars_per_day=BARS_PER_DAY,
        )
        all_sim_results.append(sim)

        fold_summaries.append({
            "fold": fold_num,
            "net_pnl": sim["net_pnl"],
            "max_dd": sim["max_dd"],
            "trades": sim["num_trades"],
            "days": sim["num_days"],
            "win_days": sim["winning_days"],
            "loss_days": sim["losing_days"],
            "stopped": sim["days_stopped"],
            "train_ssr": ga_result.best_fitness,
        })

        status = "+" if sim["net_pnl"] > 0 else "-"
        print(f"  Fold {fold_num:3d}: P&L=${sim['net_pnl']:8,.2f}  "
              f"DD=${sim['max_dd']:8,.2f}  "
              f"W/L={sim['winning_days']}/{sim['losing_days']}  "
              f"Stopped={sim['days_stopped']}d  "
              f"SSR={ga_result.best_fitness:.2f}  [{status}]")

        start += test_size

    elapsed = time.time() - t0

    # ── Aggregate Results ────────────────────────────────────────────
    total_net_pnl = sum(s["net_pnl"] for s in all_sim_results)
    total_fees = sum(s["total_fees"] for s in all_sim_results)
    total_trades = sum(s["num_trades"] for s in all_sim_results)
    total_days = sum(s["num_days"] for s in all_sim_results)
    total_winning = sum(s["winning_days"] for s in all_sim_results)
    total_losing = sum(s["losing_days"] for s in all_sim_results)
    total_stopped = sum(s["days_stopped"] for s in all_sim_results)
    positive_folds = sum(1 for s in all_sim_results if s["net_pnl"] > 0)

    # Global equity curve
    all_curves = np.concatenate([s["pnl_curve"] for s in all_sim_results])
    running_peak = np.maximum.accumulate(all_curves)
    global_max_dd = float(np.min(all_curves - running_peak))

    # Daily P&L stats
    all_daily = []
    for s in all_sim_results:
        all_daily.extend([d["day_pnl"] for d in s["daily_summaries"]])
    daily_arr = np.array(all_daily)
    avg_daily = float(np.mean(daily_arr))
    std_daily = float(np.std(daily_arr))
    best_day = float(np.max(daily_arr))
    worst_day = float(np.min(daily_arr))

    # Sharpe-like ratio (daily)
    daily_sharpe = avg_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0

    roc = total_net_pnl / args.capital * 100
    max_dd_pct = abs(global_max_dd) / args.capital * 100

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {args.size} ETH  |  ${args.daily_limit:,.0f} daily limit"
          f"  |  {fee_label}")
    print(f"{'=' * 70}")

    print(f"\n  --- Performance ---")
    print(f"  Net P&L:            ${total_net_pnl:,.2f}")
    print(f"  Total fees:         ${total_fees:,.2f}")
    print(f"  Return on capital:  {roc:.1f}%  (on ${args.capital:,.0f})")
    print(f"  Max drawdown:       ${global_max_dd:,.2f}  ({max_dd_pct:.1f}% of account)")
    print(f"  Return/DD ratio:    {abs(total_net_pnl / global_max_dd):.1f}" if global_max_dd != 0 else "")
    print(f"  Annualized Sharpe:  {daily_sharpe:.2f}")

    print(f"\n  --- Trading Activity ---")
    print(f"  Folds:              {fold_num} ({positive_folds} profitable)")
    print(f"  Total trades:       {total_trades:,}")
    print(f"  Trading days:       {total_days}")
    print(f"  Trades/day:         {total_trades / total_days:.1f}" if total_days else "")

    print(f"\n  --- Daily P&L ---")
    print(f"  Winning days:       {total_winning} ({total_winning/total_days*100:.0f}%)" if total_days else "")
    print(f"  Losing days:        {total_losing} ({total_losing/total_days*100:.0f}%)" if total_days else "")
    print(f"  Days stopped:       {total_stopped} ({total_stopped/total_days*100:.0f}%)" if total_days else "")
    print(f"  Avg daily P&L:      ${avg_daily:,.2f}")
    print(f"  Std daily P&L:      ${std_daily:,.2f}")
    print(f"  Best day:           ${best_day:,.2f}")
    print(f"  Worst day:          ${worst_day:,.2f}")

    # Equity curve milestones
    final_balance = args.capital + total_net_pnl
    print(f"\n  --- Account ---")
    print(f"  Starting balance:   ${args.capital:,.2f}")
    print(f"  Final balance:      ${final_balance:,.2f}")
    print(f"  Time:               {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n{'=' * 70}")
    if total_net_pnl > 0:
        print(f"  PROFITABLE: ${total_net_pnl:,.2f} net over {total_days} days "
              f"(~${avg_daily:,.2f}/day)")
    else:
        print(f"  UNPROFITABLE: ${total_net_pnl:,.2f} net over {total_days} days")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
