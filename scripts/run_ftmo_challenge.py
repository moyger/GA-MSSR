#!/usr/bin/env python3
"""
FTMO Prop Firm Challenge Simulator using GA-MSSR.

Simulates both 2-Step (Challenge + Verification) and 1-Step FTMO challenges
on EURUSD and/or AUDUSD with full rule enforcement:

  2-Step Challenge:  10% profit target, 5% daily loss, 10% total loss, 4 min days
  2-Step Verify:      5% profit target, 5% daily loss, 10% total loss, 4 min days
  1-Step:            10% profit target, 3% daily loss, 10% total loss, best-day rule

Usage:
    .venv/bin/python scripts/run_ftmo_challenge.py
    .venv/bin/python scripts/run_ftmo_challenge.py --pair audusd --lots 50
    .venv/bin/python scripts/run_ftmo_challenge.py --account 200000 --mode 1step
    .venv/bin/python scripts/run_ftmo_challenge.py --pair both --lots auto
"""
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from data.forex_config import EURUSD, AUDUSD, ForexConfig
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

# ── FTMO Rules ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FTMOPhaseRules:
    """Rules for a single FTMO phase."""
    name: str
    profit_target_pct: float    # % of initial balance
    max_daily_loss_pct: float   # % of initial balance
    max_total_loss_pct: float   # % of initial balance
    min_trading_days: int       # minimum days with at least 1 trade
    best_day_rule: bool = False # 1-step: best day < 50% of total profit
    best_day_pct: float = 0.50 # max share of profit from single day

# 2-Step: Challenge Phase
CHALLENGE_2STEP = FTMOPhaseRules(
    name="FTMO Challenge (2-Step)",
    profit_target_pct=0.10,     # 10%
    max_daily_loss_pct=0.05,    # 5%
    max_total_loss_pct=0.10,    # 10%
    min_trading_days=4,
)

# 2-Step: Verification Phase
VERIFY_2STEP = FTMOPhaseRules(
    name="FTMO Verification (2-Step)",
    profit_target_pct=0.05,     # 5%
    max_daily_loss_pct=0.05,    # 5%
    max_total_loss_pct=0.10,    # 10%
    min_trading_days=4,
)

# 1-Step Challenge
CHALLENGE_1STEP = FTMOPhaseRules(
    name="FTMO Challenge (1-Step)",
    profit_target_pct=0.10,     # 10%
    max_daily_loss_pct=0.03,    # 3%
    max_total_loss_pct=0.10,    # 10%
    min_trading_days=4,
    best_day_rule=True,
    best_day_pct=0.50,
)

# ── Constants ─────────────────────────────────────────────────────────
TIMEFRAME = "15min"
BARS_PER_DAY = 96   # forex 24h at 15min
TRAIN_DAYS = 20
TEST_DAYS = 5
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
GA_PERIODS = [3, 7, 15, 27]

PAIR_DATA = {
    "eurusd": {"config": EURUSD, "data": "data/forex/EURUSD_5min.csv"},
    "audusd": {"config": AUDUSD, "data": "data/forex/AUDUSD_5min.csv"},
}


# ── Helpers ───────────────────────────────────────────────────────────

def _count_trades(positions: np.ndarray) -> int:
    signs = np.sign(positions)
    return int(np.sum(np.diff(signs) != 0))


def _trade_stats(positions: np.ndarray, logr: np.ndarray):
    signs = np.sign(positions)
    changes = np.where(np.diff(signs) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(positions)]])

    wins, losses = [], []
    # Track win/loss sequence for consecutive streak calculation
    outcomes = []  # 'W' or 'L' per trade
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
            outcomes.append("W")
        elif trade_pnl < 0:
            losses.append(trade_pnl)
            outcomes.append("L")

    total = len(wins) + len(losses)
    win_rate = len(wins) / total * 100 if total > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Consecutive streaks
    max_consec_wins, max_consec_losses = 0, 0
    cur_streak, cur_type = 0, None
    for o in outcomes:
        if o == cur_type:
            cur_streak += 1
        else:
            cur_type = o
            cur_streak = 1
        if o == "W":
            max_consec_wins = max(max_consec_wins, cur_streak)
        else:
            max_consec_losses = max(max_consec_losses, cur_streak)

    return win_rate, ratio, len(wins), len(losses), max_consec_wins, max_consec_losses


def auto_lot_size(account_size: float, daily_loss_pct: float,
                  avg_daily_range_pips: float = 80.0) -> int:
    """Calculate aggressive-but-safe lot sizing for FTMO.

    Target: use 60% of the daily loss budget as the expected max adverse
    daily excursion. This leaves a 40% buffer for extreme days.
    """
    daily_loss_limit = account_size * daily_loss_pct
    usable_budget = daily_loss_limit * 0.60  # 60% of daily limit
    # Each mini lot = $1/pip; adverse day ~ avg_daily_range_pips * 0.5
    expected_adverse = avg_daily_range_pips * 0.50  # half ADR as typical adverse
    lots = int(usable_budget / expected_adverse)
    return max(1, lots)


# ── Threshold Sweep (find best threshold) ────────────────────────────

def run_threshold_sweep(df, config: ForexConfig, ga_config: GAMSSRConfig,
                        lots: int):
    """Train one model, sweep thresholds. Returns best threshold + details."""
    print(f"\n  [Threshold Sweep] Training 16 rules + GA optimization...")
    t0 = time.time()
    rule_params = train_rule_params(df, periods=GA_PERIODS)
    features = get_rule_features(df, rule_params)
    ga = GAMSSR(ga_config)
    result = ga.fit(features)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s | SSR={result.best_fitness:.6f}")

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
        metrics = ssr_calc.calculate(port_ret)
        trades = _count_trades(pos)
        flat_pct = float(np.sum(pos == 0)) / len(pos) * 100
        win_rate, wl_ratio, wins, loss_count, _, _ = _trade_stats(pos, logr)

        avg_price = df["close"].mean()
        dollar_pnl = metrics.total_return * avg_price * config.point_value * lots
        dollar_dd = metrics.max_drawdown * avg_price * config.point_value * lots

        rows.append({
            "threshold": threshold, "trades": trades,
            "return_pct": metrics.total_return * 100,
            "max_dd_pct": metrics.max_drawdown * 100,
            "ssr": metrics.ssr, "win_rate": win_rate,
            "wl_ratio": wl_ratio, "flat_pct": flat_pct,
            "dollar_pnl": dollar_pnl, "dollar_dd": dollar_dd,
        })

    # Print table
    print(f"\n  {'Thr':>5} | {'Trades':>6} | {'Ret%':>7} | {'DD%':>7} | "
          f"{'SSR':>9} | {'WR%':>5} | {'W/L':>5} | {'$PnL':>10}")
    print("  " + "-" * 72)
    best = max(rows, key=lambda r: r["ssr"])
    for r in rows:
        marker = " <--" if r is best and r["threshold"] > 0 else ""
        print(f"  {r['threshold']:5.2f} | {r['trades']:6d} | "
              f"{r['return_pct']:+6.2f}% | {r['max_dd_pct']:+6.2f}% | "
              f"{r['ssr']:9.6f} | {r['win_rate']:4.1f}% | "
              f"{r['wl_ratio']:5.2f} | ${r['dollar_pnl']:9.2f}{marker}")

    print(f"\n  Best threshold: {best['threshold']:.2f} "
          f"(SSR={best['ssr']:.6f})")
    return best["threshold"], rows


# ── FTMO Walk-Forward Simulation ─────────────────────────────────────

def run_ftmo_walkforward(df, config: ForexConfig, ga_config: GAMSSRConfig,
                         threshold: float, account_size: float, lots: int,
                         rules: FTMOPhaseRules, custom_daily_limit: float = 0):
    """
    Walk-forward simulation with full FTMO rule enforcement.

    Returns dict with detailed results including pass/fail status.
    """
    train_size = BARS_PER_DAY * TRAIN_DAYS
    test_size = BARS_PER_DAY * TEST_DAYS
    n = len(df)

    # FTMO dollar limits (custom daily limit overrides FTMO default if set)
    ftmo_daily_limit = account_size * rules.max_daily_loss_pct
    daily_loss_limit = custom_daily_limit if custom_daily_limit > 0 else ftmo_daily_limit
    total_loss_limit = account_size * rules.max_total_loss_pct
    profit_target = account_size * rules.profit_target_pct
    equity_floor = account_size - total_loss_limit

    print(f"\n  [FTMO Walk-Forward] {rules.name}")
    print(f"    Account:       ${account_size:,.0f}")
    print(f"    Profit target: ${profit_target:,.0f} "
          f"({rules.profit_target_pct*100:.0f}%)")
    dl_label = (f"${daily_loss_limit:,.0f} (custom)"
                if custom_daily_limit > 0
                else f"${daily_loss_limit:,.0f} ({rules.max_daily_loss_pct*100:.0f}%)")
    print(f"    Daily loss:    {dl_label}")
    print(f"    Total loss:    ${total_loss_limit:,.0f} "
          f"({rules.max_total_loss_pct*100:.0f}%) → floor ${equity_floor:,.0f}")
    print(f"    Min days:      {rules.min_trading_days}")
    print(f"    Lots:          {lots} mini ({lots*10}K, ${lots}/pip)")
    if rules.best_day_rule:
        print(f"    Best-day rule: single day < {rules.best_day_pct*100:.0f}% "
              f"of total profit")

    ssr_calc = SSR()
    balance = account_size
    high_water_mark = account_size
    all_pnl_bars = []
    fold_summaries = []
    all_positions = []
    all_returns = []

    # Daily tracking
    daily_pnls = []       # list of (date_label, daily_pnl)
    trading_days = set()   # days where at least 1 trade was placed
    days_stopped = 0

    # FTMO breach tracking
    challenge_failed = False
    fail_reason = ""
    target_hit = False
    target_hit_day = None

    start = 0
    fold_num = 0
    global_bar = 0

    while start + train_size + test_size <= n:
        if challenge_failed:
            break

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

        # Predict on test data
        logr = test_features["logr"].values
        signal_cols = [c for c in test_features.columns if c.startswith("rule_")]
        signals = test_features[signal_cols].values
        raw_pos = signals @ weights
        positions = np.where(raw_pos > threshold, 1.0,
                    np.where(raw_pos < -threshold, -1.0, 0.0))

        # Raw prices for dollar PnL
        raw_col = "close_raw" if "close_raw" in test_df.columns else "close"
        prices = test_df[raw_col].iloc[-len(test_features):].values
        test_timestamps = test_df.index[-len(test_features):]

        # Bar-by-bar simulation with FTMO enforcement
        bar_pnl = np.zeros(len(positions))
        actual_positions = positions.copy()
        fold_fees = 0.0
        daily_pnl = 0.0
        daily_stopped = False
        fold_days_stopped = 0
        current_day = None

        for i in range(len(positions)):
            if challenge_failed:
                actual_positions[i] = 0
                continue

            bar_day = test_timestamps[i].date() if i < len(test_timestamps) else None

            # New day check
            if bar_day != current_day:
                # Save previous day
                if current_day is not None and daily_pnl != 0:
                    daily_pnls.append((str(current_day), daily_pnl))

                current_day = bar_day
                daily_pnl = 0.0
                daily_stopped = False

            # If daily stopped, flatten
            if daily_stopped:
                actual_positions[i] = 0
                continue

            # PnL from held position
            point_val = config.point_value * lots
            comm = config.commission_rt * lots

            if i > 0 and actual_positions[i - 1] != 0:
                price_change = prices[i] - prices[i - 1]
                pnl = actual_positions[i - 1] * price_change * point_val
                bar_pnl[i] += pnl
                daily_pnl += pnl
                balance += pnl

            # Fee on position change
            if i > 0 and actual_positions[i] != actual_positions[i - 1]:
                if actual_positions[i - 1] != 0:
                    bar_pnl[i] -= comm
                    fold_fees += comm
                    daily_pnl -= comm
                    balance -= comm
                if actual_positions[i] != 0:
                    bar_pnl[i] -= comm
                    fold_fees += comm
                    daily_pnl -= comm
                    balance -= comm
                    # Record trading day
                    if bar_day:
                        trading_days.add(bar_day)

            # Track high water mark
            if balance > high_water_mark:
                high_water_mark = balance

            # ── FTMO Rule Checks ──

            # 1) Total loss check (equity floor)
            if balance <= equity_floor:
                challenge_failed = True
                fail_reason = (f"TOTAL LOSS BREACHED: balance ${balance:,.2f} "
                               f"<= floor ${equity_floor:,.0f}")
                # Flatten
                if actual_positions[i] != 0:
                    bar_pnl[i] -= comm
                    fold_fees += comm
                    balance -= comm
                    actual_positions[i] = 0
                break

            # 2) Daily loss check
            if daily_pnl <= -daily_loss_limit:
                daily_stopped = True
                fold_days_stopped += 1
                # Flatten
                if actual_positions[i] != 0:
                    bar_pnl[i] -= comm
                    fold_fees += comm
                    daily_pnl -= comm
                    balance -= comm
                    actual_positions[i] = 0

            # 3) Profit target check
            total_profit = balance - account_size
            if not target_hit and total_profit >= profit_target:
                target_hit = True
                target_hit_day = str(bar_day) if bar_day else f"fold_{fold_num}"

        # Save last day of fold
        if current_day is not None and daily_pnl != 0:
            daily_pnls.append((str(current_day), daily_pnl))

        days_stopped += fold_days_stopped
        port_return = actual_positions * logr

        net_pnl = float(np.sum(bar_pnl))
        cum_pnl = np.cumsum(bar_pnl)
        all_pnl_bars.append(bar_pnl)
        all_positions.extend(actual_positions.tolist())
        all_returns.extend(port_return.tolist())

        fold_trades = _count_trades(actual_positions)
        oos_ssr = ssr_calc.calculate(port_return)

        fold_summaries.append({
            "fold": fold_num, "net_pnl": net_pnl,
            "ssr": oos_ssr.ssr, "trades": fold_trades,
            "fees": fold_fees, "days_stopped": fold_days_stopped,
            "balance_end": balance,
        })

        pnl_sign = "+" if net_pnl > 0 else "-"
        stop_label = f"  Stop={fold_days_stopped}d" if fold_days_stopped > 0 else ""
        tgt_label = "  *** TARGET HIT ***" if target_hit and target_hit_day else ""
        print(f"    Fold {fold_num:2d}: PnL=${net_pnl:8.2f}  "
              f"Bal=${balance:10,.2f}  Trades={fold_trades:3d}"
              f"{stop_label}{tgt_label}  [{pnl_sign}]")

        if challenge_failed:
            print(f"    *** CHALLENGE FAILED: {fail_reason}")
            break

        start += test_size

    if not fold_summaries:
        return {"passed": False, "fail_reason": "No folds completed"}

    # ── Aggregate results ──
    total_net_pnl = balance - account_size
    total_fees = sum(f["fees"] for f in fold_summaries)
    total_trades = sum(f["trades"] for f in fold_summaries)
    num_trading_days = len(trading_days)

    # Global equity curve
    all_bars = np.concatenate(all_pnl_bars)
    global_curve = np.cumsum(all_bars) + account_size  # absolute balance
    running_peak = np.maximum.accumulate(global_curve)
    max_dd_dollar = float(np.min(global_curve - running_peak))
    max_dd_pct = abs(max_dd_dollar) / account_size * 100

    # Daily PnL analysis
    daily_pnl_values = [d[1] for d in daily_pnls]
    worst_day = min(daily_pnl_values) if daily_pnl_values else 0
    best_day = max(daily_pnl_values) if daily_pnl_values else 0
    positive_days = sum(1 for d in daily_pnl_values if d > 0)
    negative_days = sum(1 for d in daily_pnl_values if d < 0)

    # Best-day rule check (1-step)
    best_day_violation = False
    if rules.best_day_rule and total_net_pnl > 0:
        total_positive = sum(d for d in daily_pnl_values if d > 0)
        if total_positive > 0 and best_day / total_positive > rules.best_day_pct:
            best_day_violation = True

    # OOS aggregate stats
    all_ret = np.array(all_returns)
    all_pos = np.array(all_positions)
    win_rate, wl_ratio, wins, loss_count, max_consec_wins, max_consec_losses = \
        _trade_stats(all_pos, all_ret)
    oos_ssr = ssr_calc.calculate(all_ret)

    total_bars = len(all_positions)
    oos_days = total_bars / BARS_PER_DAY
    trades_per_day = total_trades / oos_days if oos_days > 0 else 0

    # Consecutive winning/losing days
    max_consec_win_days, max_consec_loss_days = 0, 0
    cur_day_streak, cur_day_type = 0, None
    for _, dpnl in daily_pnls:
        day_type = "W" if dpnl > 0 else "L" if dpnl < 0 else None
        if day_type is None:
            continue
        if day_type == cur_day_type:
            cur_day_streak += 1
        else:
            cur_day_type = day_type
            cur_day_streak = 1
        if day_type == "W":
            max_consec_win_days = max(max_consec_win_days, cur_day_streak)
        else:
            max_consec_loss_days = max(max_consec_loss_days, cur_day_streak)

    # ── Pass/Fail determination ──
    passed = True
    fail_reasons = []

    if challenge_failed:
        passed = False
        fail_reasons.append(fail_reason)

    if not target_hit:
        passed = False
        fail_reasons.append(
            f"Profit target not reached: ${total_net_pnl:,.2f} / "
            f"${profit_target:,.0f} needed "
            f"({total_net_pnl/profit_target*100:.1f}% of target)")

    if num_trading_days < rules.min_trading_days:
        passed = False
        fail_reasons.append(
            f"Min trading days not met: {num_trading_days} / "
            f"{rules.min_trading_days} required")

    if best_day_violation:
        passed = False
        fail_reasons.append(
            f"Best-day rule violated: best day ${best_day:,.2f} = "
            f"{best_day/sum(d for d in daily_pnl_values if d > 0)*100:.1f}% "
            f"of total profit (max {rules.best_day_pct*100:.0f}%)")

    return {
        "passed": passed,
        "fail_reasons": fail_reasons,
        "target_hit": target_hit,
        "target_hit_day": target_hit_day,
        "account_size": account_size,
        "final_balance": balance,
        "net_pnl": total_net_pnl,
        "roc_pct": total_net_pnl / account_size * 100,
        "profit_target": profit_target,
        "profit_progress_pct": total_net_pnl / profit_target * 100,
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "total_fees": total_fees,
        "total_trades": total_trades,
        "trades_per_day": trades_per_day,
        "num_trading_days": num_trading_days,
        "oos_days": oos_days,
        "win_rate": win_rate,
        "wl_ratio": wl_ratio,
        "oos_ssr": oos_ssr.ssr,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "max_consec_win_days": max_consec_win_days,
        "max_consec_loss_days": max_consec_loss_days,
        "worst_day": worst_day,
        "best_day": best_day,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "days_stopped": days_stopped,
        "daily_pnls": daily_pnls,
        "equity_curve": global_curve,
        "fold_summaries": fold_summaries,
        "best_day_violation": best_day_violation,
        "high_water_mark": high_water_mark,
        "daily_loss_limit": daily_loss_limit,
        "num_folds": fold_num,
        "positive_folds": sum(1 for f in fold_summaries if f["net_pnl"] > 0),
    }


# ── Report Printing ──────────────────────────────────────────────────

def print_ftmo_report(pair: str, rules: FTMOPhaseRules, lots: int,
                      result: dict):
    """Print detailed FTMO compliance report."""
    account = result["account_size"]
    daily_lim = result.get("daily_loss_limit", account * rules.max_daily_loss_pct)
    total_lim = account * rules.max_total_loss_pct

    print(f"\n{'='*70}")
    verdict = "PASSED" if result["passed"] else "FAILED"
    print(f"  {rules.name} — {pair} — {verdict}")
    print(f"{'='*70}")

    print(f"\n  --- Account ---")
    print(f"  Starting balance:  ${account:>12,.2f}")
    print(f"  Final balance:     ${result['final_balance']:>12,.2f}")
    print(f"  Net PnL:           ${result['net_pnl']:>12,.2f}")
    print(f"  ROC:               {result['roc_pct']:>11.2f}%")
    print(f"  High water mark:   ${result['high_water_mark']:>12,.2f}")

    print(f"\n  --- FTMO Rule Compliance ---")

    # Profit target
    target = account * rules.profit_target_pct
    tgt_status = "PASS" if result["target_hit"] else "FAIL"
    print(f"  Profit target:     ${result['net_pnl']:>10,.2f} / "
          f"${target:>8,.0f}  ({result['profit_progress_pct']:5.1f}%)  "
          f"[{tgt_status}]")
    if result["target_hit"]:
        print(f"    Target hit on:   {result['target_hit_day']}")

    # Daily loss
    worst = result["worst_day"]
    dl_status = "PASS" if abs(worst) < daily_lim else "FAIL"
    print(f"  Max daily loss:    ${worst:>10,.2f} / "
          f"${-daily_lim:>8,.0f}  "
          f"({abs(worst)/daily_lim*100:5.1f}% of limit)  [{dl_status}]")

    # Total loss (max drawdown)
    dd = result["max_dd_dollar"]
    tl_status = "PASS" if abs(dd) < total_lim else "FAIL"
    print(f"  Max drawdown:      ${dd:>10,.2f} / "
          f"${-total_lim:>8,.0f}  "
          f"({abs(dd)/total_lim*100:5.1f}% of limit)  [{tl_status}]")

    # Min trading days
    td_status = ("PASS" if result["num_trading_days"] >= rules.min_trading_days
                 else "FAIL")
    print(f"  Trading days:      {result['num_trading_days']:>10d} / "
          f"{rules.min_trading_days:>8d} min  "
          f"{'':>16}  [{td_status}]")

    # Best-day rule (1-step only)
    if rules.best_day_rule:
        bd_status = "PASS" if not result["best_day_violation"] else "FAIL"
        total_pos = sum(d for d in [p[1] for p in result["daily_pnls"]] if d > 0)
        bd_pct = (result["best_day"] / total_pos * 100) if total_pos > 0 else 0
        print(f"  Best-day rule:     {bd_pct:>9.1f}% / "
              f"{rules.best_day_pct*100:>7.0f}% max  "
              f"{'':>16}  [{bd_status}]")

    print(f"\n  --- Trading Stats ---")
    print(f"  Total trades:      {result['total_trades']:>10,}")
    print(f"  Trades/day:        {result['trades_per_day']:>10.1f}")
    print(f"  Win rate:          {result['win_rate']:>9.1f}%")
    print(f"  W/L ratio:         {result['wl_ratio']:>10.2f}")
    print(f"  OOS SSR:           {result['oos_ssr']:>10.6f}")
    print(f"  Folds:             {result['num_folds']} "
          f"({result['positive_folds']} profitable)")
    print(f"  Total fees:        ${result['total_fees']:>10,.2f}")
    print(f"  Days stopped:      {result['days_stopped']:>10}")

    print(f"\n  --- Consecutive Streaks ---")
    print(f"  Consec wins:       {result['max_consec_wins']:>10} trades")
    print(f"  Consec losses:     {result['max_consec_losses']:>10} trades")
    print(f"  Consec win days:   {result['max_consec_win_days']:>10} days")
    print(f"  Consec loss days:  {result['max_consec_loss_days']:>10} days")

    print(f"\n  --- Daily P&L Summary ---")
    print(f"  Profitable days:   {result['positive_days']:>10}")
    print(f"  Losing days:       {result['negative_days']:>10}")
    print(f"  Best day:          ${result['best_day']:>10,.2f}")
    print(f"  Worst day:         ${result['worst_day']:>10,.2f}")
    avg_daily = (result["net_pnl"] / result["oos_days"]
                 if result["oos_days"] > 0 else 0)
    print(f"  Avg daily PnL:     ${avg_daily:>10,.2f}")

    # Overall verdict
    print(f"\n  {'='*50}")
    if result["passed"]:
        print(f"  RESULT: *** CHALLENGE PASSED ***")
        print(f"  Profit: ${result['net_pnl']:,.2f} "
              f"({result['roc_pct']:.2f}% ROC)")
    else:
        print(f"  RESULT: *** CHALLENGE FAILED ***")
        for reason in result["fail_reasons"]:
            print(f"    - {reason}")
    print(f"  {'='*50}")


# ── Chart Generation ─────────────────────────────────────────────────

def generate_ftmo_chart(pair: str, rules: FTMOPhaseRules, lots: int,
                        result: dict, output_dir: Path):
    """Generate FTMO challenge equity curve with rule lines."""
    output_dir.mkdir(parents=True, exist_ok=True)

    account = result["account_size"]
    daily_lim = result.get("daily_loss_limit", account * rules.max_daily_loss_pct)
    total_lim = account * rules.max_total_loss_pct
    target = account * rules.profit_target_pct
    equity_floor = account - total_lim
    target_line = account + target

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    verdict = "PASSED" if result["passed"] else "FAILED"
    lot_label = f"{lots}x Mini ({lots*10}K, ${lots}/pip)"
    fig.suptitle(f"{rules.name} — {pair} — {verdict}\n"
                 f"{lot_label} | Account: ${account:,.0f}",
                 fontsize=14, fontweight="bold")

    # Equity curve
    curve = result["equity_curve"]
    x = range(len(curve))

    # Color the line green when above start, red when below
    ax1.plot(x, curve, color="#2196F3", linewidth=1.2, label="Balance")

    # FTMO rule lines
    ax1.axhline(y=account, color="gray", linestyle="--", alpha=0.5,
                label=f"Starting ${account:,.0f}")
    ax1.axhline(y=target_line, color="#4CAF50", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Target ${target_line:,.0f} (+{rules.profit_target_pct*100:.0f}%)")
    ax1.axhline(y=equity_floor, color="#F44336", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Floor ${equity_floor:,.0f} (-{rules.max_total_loss_pct*100:.0f}%)")

    # Fill danger zone
    ax1.fill_between(x, equity_floor, account - total_lim * 0.5,
                      color="#F44336", alpha=0.05)
    # Fill target zone
    ax1.fill_between(x, target_line, target_line + target * 0.3,
                      color="#4CAF50", alpha=0.05)

    # Mark target hit
    if result["target_hit"]:
        # Find first bar where balance >= target
        target_bar = None
        for i, b in enumerate(curve):
            if b >= target_line:
                target_bar = i
                break
        if target_bar is not None:
            ax1.axvline(x=target_bar, color="#4CAF50", alpha=0.5,
                        linestyle=":", linewidth=2)
            ax1.annotate("TARGET HIT", xy=(target_bar, curve[target_bar]),
                         fontsize=10, color="#4CAF50", fontweight="bold",
                         xytext=(10, 20), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color="#4CAF50"))

    # Mark fold boundaries
    if result["fold_summaries"]:
        offset = 0
        for fold in result["fold_summaries"]:
            ax1.axvline(x=offset, color="orange", alpha=0.15, linestyle=":")
            offset += BARS_PER_DAY * TEST_DAYS

    ax1.set_ylabel("Account Balance ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity Curve")

    # Daily PnL bar chart
    daily_dates = [d[0] for d in result["daily_pnls"]]
    daily_vals = [d[1] for d in result["daily_pnls"]]
    bar_colors = ["#4CAF50" if v > 0 else "#F44336" for v in daily_vals]

    ax2.bar(range(len(daily_vals)), daily_vals, color=bar_colors, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.axhline(y=-daily_lim, color="#F44336", linestyle="--", alpha=0.6,
                label=f"Daily limit -${daily_lim:,.0f}")
    ax2.set_ylabel("Daily PnL ($)")
    ax2.set_xlabel("Trading Day")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Daily P&L")

    # Set x-tick labels (show every 5th day)
    if daily_dates:
        step = max(1, len(daily_dates) // 15)
        ax2.set_xticks(range(0, len(daily_dates), step))
        ax2.set_xticklabels([daily_dates[i] for i in range(0, len(daily_dates), step)],
                             rotation=45, fontsize=7)

    plt.tight_layout()
    mode = "1step" if rules.best_day_rule else "2step"
    chart_path = output_dir / f"ftmo_{pair.lower()}_{mode}_{TIMEFRAME}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {chart_path}")
    return chart_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FTMO Prop Firm Challenge Simulator — GA-MSSR")
    parser.add_argument("--pair", choices=["eurusd", "audusd", "both"],
                        default="both",
                        help="Pair(s) to simulate (default: both)")
    parser.add_argument("--account", type=float, default=100_000,
                        help="FTMO account size (default: 100000)")
    parser.add_argument("--mode", choices=["2step", "1step", "full"],
                        default="2step",
                        help="Challenge mode: 2step, 1step, or full (default: 2step)")
    parser.add_argument("--lots", type=str, default="auto",
                        help="Mini lots: integer or 'auto' (default: auto)")
    parser.add_argument("--daily-limit", type=float, default=0,
                        help="Custom daily loss limit in $ (overrides FTMO %%; default: use FTMO rule)")
    args = parser.parse_args()

    account = args.account
    custom_daily_limit = args.daily_limit

    # Determine phases to run
    if args.mode == "2step":
        phases = [CHALLENGE_2STEP, VERIFY_2STEP]
    elif args.mode == "1step":
        phases = [CHALLENGE_1STEP]
    else:  # full = try both modes
        phases = [CHALLENGE_2STEP, VERIFY_2STEP, CHALLENGE_1STEP]

    # Auto lot sizing
    if args.lots == "auto":
        lots = auto_lot_size(account, phases[0].max_daily_loss_pct)
        print(f"\n  Auto lot sizing: {lots} mini lots "
              f"({lots*10}K units, ${lots}/pip)")
    else:
        lots = int(args.lots)

    pairs = ["eurusd", "audusd"] if args.pair == "both" else [args.pair]

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    print("=" * 70)
    print("  FTMO PROP FIRM CHALLENGE SIMULATOR — GA-MSSR")
    print(f"  Account: ${account:,.0f}  |  Mode: {args.mode}  |  "
          f"Lots: {lots} ({lots*10}K, ${lots}/pip)")
    if custom_daily_limit > 0:
        print(f"  Custom daily loss limit: ${custom_daily_limit:,.0f}")
    print(f"  Pairs: {', '.join(p.upper() for p in pairs)}")
    print("=" * 70)

    t0_total = time.time()
    all_results = []  # list of (pair, phase_rules, lots, result)

    for pair_name in pairs:
        pair_info = PAIR_DATA[pair_name]
        config = pair_info["config"]
        symbol = config.symbol

        print(f"\n{'#'*70}")
        print(f"  {symbol} — Loading & preprocessing")
        print(f"{'#'*70}")

        df = build_denoised_dataset(pair_info["data"],
                                     denoise_columns="close",
                                     timeframe=TIMEFRAME)
        print(f"  {len(df):,} bars | {df.index[0]} to {df.index[-1]}")

        # Find best threshold first
        best_threshold, sweep_rows = run_threshold_sweep(
            df, config, ga_config, lots)

        # Run each FTMO phase
        for rules in phases:
            result = run_ftmo_walkforward(
                df, config, ga_config, best_threshold,
                account, lots, rules,
                custom_daily_limit=custom_daily_limit)

            print_ftmo_report(symbol, rules, lots, result)

            # Generate chart
            output_dir = Path("reports")
            generate_ftmo_chart(symbol, rules, lots, result, output_dir)

            all_results.append((symbol, rules, lots, result))

    # ── Final Summary ──
    elapsed = time.time() - t0_total

    print(f"\n{'='*70}")
    print(f"  FTMO CHALLENGE SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Pair':<10} {'Phase':<30} {'Lots':>5} "
          f"{'PnL':>10} {'ROC%':>7} {'MaxDD%':>7} {'Days':>5} {'Result':>8}")
    print(f"  {'-'*88}")

    for symbol, rules, lot_count, result in all_results:
        verdict = "PASSED" if result["passed"] else "FAILED"
        phase_short = ("Challenge" if "Challenge" in rules.name else "Verify")
        step = "1S" if rules.best_day_rule else "2S"
        label = f"{phase_short} ({step})"
        print(f"  {symbol:<10} {label:<30} {lot_count:>5} "
              f"${result['net_pnl']:>9,.2f} "
              f"{result['roc_pct']:>6.2f}% "
              f"{result['max_dd_pct']:>6.2f}% "
              f"{result['num_trading_days']:>5} "
              f"{'** ' + verdict + ' **':>8}")

    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("  Done.")


if __name__ == "__main__":
    main()
