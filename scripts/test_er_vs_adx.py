#!/usr/bin/env python3
"""
A/B Test: ER Filter vs ER + ADX Filter on ETH/USD.

Compares three configurations across walk-forward folds:
  A) ER only (current production: ER(14) > 0.40)
  B) ADX only (ADX(14) > 25)
  C) ER + ADX combined (both must pass)

For each fold, trains GA on the training window, generates OOS predictions,
then applies each filter to the OOS predictions and measures performance.

Usage:
    .venv/bin/python scripts/test_er_vs_adx.py
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
from strategies.indicators import efficiency_ratio, adx

# ── Config ─────────────────────────────────────────────────────────────
DATA_PATH = "data/futures/ETHUSD_1min.csv"
TIMEFRAME = "15min"
BARS_PER_DAY = 96
TRAIN_DAYS = 20
TEST_DAYS = 5
PERIODS = [3, 7, 15, 27]

# Filter parameters
ER_PERIOD = 14
ER_THRESHOLD = 0.40
ADX_PERIOD = 14
ADX_THRESHOLD = 25


def apply_filter(positions, df_chunk, filter_type):
    """Apply a filter to position series. Returns filtered positions."""
    filtered = positions.copy()

    if filter_type == "none":
        return filtered

    er_vals = efficiency_ratio(df_chunk["close"], n=ER_PERIOD)
    adx_vals, _, _ = adx(df_chunk["high"], df_chunk["low"], df_chunk["close"], n=ADX_PERIOD)

    # Align to position index
    er_aligned = er_vals.reindex(filtered.index)
    adx_aligned = adx_vals.reindex(filtered.index)

    if filter_type == "er":
        mask = er_aligned < ER_THRESHOLD
        filtered[mask] = 0
    elif filter_type == "adx":
        mask = adx_aligned < ADX_THRESHOLD
        filtered[mask] = 0
    elif filter_type == "er+adx":
        # Both must pass: ER >= threshold AND ADX >= threshold
        mask = (er_aligned < ER_THRESHOLD) | (adx_aligned < ADX_THRESHOLD)
        filtered[mask] = 0

    return filtered


def compute_fold_metrics(positions, log_returns):
    """Compute metrics from filtered positions and log returns."""
    port_returns = positions * log_returns
    port_returns = port_returns.dropna()

    if len(port_returns) == 0:
        return {"total_return": 0, "ssr": 0, "max_dd": 0,
                "trades": 0, "win_rate": 0, "flat_pct": 0}

    total_return = float(port_returns.sum())
    cum = np.cumsum(port_returns.values)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    # SSR
    ssr_calc = SSR()
    ssr_result = ssr_calc.calculate(port_returns.values)
    ssr_val = ssr_result.ssr

    # Trade count (position changes)
    pos_changes = (positions.diff().abs() > 0).sum()

    # Win rate (per-bar)
    wins = (port_returns > 0).sum()
    total_nonzero = (port_returns != 0).sum()
    win_rate = wins / total_nonzero * 100 if total_nonzero > 0 else 0

    # Flat percentage
    flat_pct = (positions == 0).sum() / len(positions) * 100

    return {
        "total_return": total_return,
        "ssr": ssr_val,
        "max_dd": max_dd,
        "trades": int(pos_changes),
        "win_rate": float(win_rate),
        "flat_pct": float(flat_pct),
    }


def main():
    print("=" * 70)
    print("  A/B TEST: ER vs ADX vs ER+ADX Filter  |  ETH/USD 15min")
    print("=" * 70)

    print(f"\nFilter configs:")
    print(f"  A) No filter (baseline)")
    print(f"  B) ER only: ER({ER_PERIOD}) >= {ER_THRESHOLD}")
    print(f"  C) ADX only: ADX({ADX_PERIOD}) >= {ADX_THRESHOLD}")
    print(f"  D) ER + ADX: ER({ER_PERIOD}) >= {ER_THRESHOLD} AND ADX({ADX_PERIOD}) >= {ADX_THRESHOLD}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = build_denoised_dataset(DATA_PATH, denoise_columns="close", timeframe=TIMEFRAME)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    train_size = BARS_PER_DAY * TRAIN_DAYS
    test_size = BARS_PER_DAY * TEST_DAYS

    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )

    filters = ["none", "er", "adx", "er+adx"]
    filter_labels = {
        "none": "No Filter",
        "er": f"ER({ER_PERIOD})>={ER_THRESHOLD}",
        "adx": f"ADX({ADX_PERIOD})>={ADX_THRESHOLD}",
        "er+adx": f"ER+ADX",
    }

    # Collect per-fold results for each filter
    results = {f: [] for f in filters}

    n_folds = 0
    start_idx = train_size

    print(f"\nRunning walk-forward folds...")
    print("-" * 70)

    t0 = time.time()

    while start_idx + test_size <= len(df):
        n_folds += 1
        train_df = df.iloc[start_idx - train_size: start_idx]
        test_df = df.iloc[start_idx: start_idx + test_size]

        # Train
        rule_params = train_rule_params(train_df, periods=PERIODS)
        train_features = get_rule_features(train_df, rule_params)
        test_features = get_rule_features(test_df, rule_params)

        if len(train_features) < 10 or len(test_features) < 5:
            start_idx += test_size
            continue

        ga = GAMSSR(ga_config)
        ga.fit(train_features)
        pred = ga.predict(test_features)

        # Raw unfiltered positions
        raw_positions = pred["position"]
        log_returns = test_features["logr"]

        # Apply each filter
        for filt in filters:
            filtered_pos = apply_filter(raw_positions, test_df, filt)
            port_returns = filtered_pos * log_returns
            metrics = compute_fold_metrics(filtered_pos, log_returns)
            results[filt].append(metrics)

        fold_rets = {f: results[f][-1]["total_return"] for f in filters}
        print(f"  Fold {n_folds:2d}: "
              f"none={fold_rets['none']*100:+6.2f}%  "
              f"ER={fold_rets['er']*100:+6.2f}%  "
              f"ADX={fold_rets['adx']*100:+6.2f}%  "
              f"ER+ADX={fold_rets['er+adx']*100:+6.2f}%")

        start_idx += test_size

    elapsed = time.time() - t0
    print(f"\n  {n_folds} folds completed in {elapsed:.1f}s")

    # ── Aggregate Results ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE RESULTS ({n_folds} folds)")
    print(f"{'=' * 70}")

    header = (f"  {'Filter':>12} {'Total Ret':>10} {'Avg SSR':>8} {'Max DD':>8} "
              f"{'Win Rate':>9} {'Trades/d':>9} {'Flat%':>7} {'Folds+':>7}")
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    oos_days = n_folds * TEST_DAYS
    summary = {}

    for filt in filters:
        fold_data = results[filt]
        total_ret = sum(m["total_return"] for m in fold_data)
        avg_ssr = np.mean([m["ssr"] for m in fold_data])
        cum_returns = np.cumsum([m["total_return"] for m in fold_data])
        max_dd = float(np.min(cum_returns - np.maximum.accumulate(cum_returns)))
        avg_win_rate = np.mean([m["win_rate"] for m in fold_data])
        total_trades = sum(m["trades"] for m in fold_data)
        trades_per_day = total_trades / oos_days if oos_days > 0 else 0
        avg_flat = np.mean([m["flat_pct"] for m in fold_data])
        positive_folds = sum(1 for m in fold_data if m["total_return"] > 0)

        summary[filt] = {
            "total_ret": total_ret,
            "avg_ssr": avg_ssr,
            "max_dd": max_dd,
            "avg_win_rate": avg_win_rate,
            "trades_per_day": trades_per_day,
            "avg_flat": avg_flat,
            "positive_folds": positive_folds,
        }

        print(f"  {filter_labels[filt]:>12}"
              f"  {total_ret*100:+8.2f}%"
              f"  {avg_ssr:7.3f}"
              f"  {max_dd*100:6.2f}%"
              f"  {avg_win_rate:7.1f}%"
              f"  {trades_per_day:7.1f}"
              f"  {avg_flat:5.1f}%"
              f"  {positive_folds:3d}/{n_folds}")

    # ── Dollar P&L Comparison (at 0.5 ETH) ────────────────────────────
    avg_price = float(df["close_raw"].mean() if "close_raw" in df.columns
                      else df["close"].mean())
    trade_size = 0.5
    maker_fee = 0.00020

    print(f"\n{'=' * 70}")
    print(f"  DOLLAR P&L (0.5 ETH, maker fee {maker_fee*100:.3f}%)")
    print(f"{'=' * 70}")

    print(f"\n  {'Filter':>12} {'Gross $':>10} {'Fees $':>9} {'Net $':>10} {'Net/Day':>9}")
    print("  " + "-" * 52)

    for filt in filters:
        s = summary[filt]
        gross = s["total_ret"] * avg_price * trade_size
        total_trades = sum(m["trades"] for m in results[filt])
        fees = maker_fee * avg_price * trade_size * total_trades * 2
        net = gross - fees
        net_per_day = net / oos_days if oos_days > 0 else 0

        print(f"  {filter_labels[filt]:>12}"
              f"  ${gross:8,.2f}"
              f"  ${fees:7,.2f}"
              f"  ${net:8,.2f}"
              f"  ${net_per_day:7,.2f}")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    best_filter = max(filters, key=lambda f: summary[f]["avg_ssr"])
    best_ret_filter = max(filters, key=lambda f: summary[f]["total_ret"])

    print(f"\n  Best by SSR:     {filter_labels[best_filter]} "
          f"(SSR={summary[best_filter]['avg_ssr']:.3f})")
    print(f"  Best by return:  {filter_labels[best_ret_filter]} "
          f"(ret={summary[best_ret_filter]['total_ret']*100:+.2f}%)")

    # Compare ER+ADX vs ER alone
    er_ret = summary["er"]["total_ret"]
    combo_ret = summary["er+adx"]["total_ret"]
    er_ssr = summary["er"]["avg_ssr"]
    combo_ssr = summary["er+adx"]["avg_ssr"]

    if combo_ret > er_ret and combo_ssr > er_ssr:
        print(f"\n  ER+ADX IMPROVES over ER alone:")
        print(f"    Return: {er_ret*100:+.2f}% → {combo_ret*100:+.2f}% "
              f"({(combo_ret-er_ret)*100:+.2f}%)")
        print(f"    SSR:    {er_ssr:.3f} → {combo_ssr:.3f}")
    elif combo_ret < er_ret and combo_ssr < er_ssr:
        print(f"\n  ER+ADX HURTS performance vs ER alone:")
        print(f"    Return: {er_ret*100:+.2f}% → {combo_ret*100:+.2f}% "
              f"({(combo_ret-er_ret)*100:+.2f}%)")
        print(f"    SSR:    {er_ssr:.3f} → {combo_ssr:.3f}")
    else:
        print(f"\n  ER+ADX is MIXED vs ER alone:")
        print(f"    Return: {er_ret*100:+.2f}% → {combo_ret*100:+.2f}%")
        print(f"    SSR:    {er_ssr:.3f} → {combo_ssr:.3f}")

    adx_ret = summary["adx"]["total_ret"]
    adx_ssr = summary["adx"]["avg_ssr"]
    print(f"\n  ADX alone vs ER alone:")
    print(f"    Return: ER {er_ret*100:+.2f}% vs ADX {adx_ret*100:+.2f}%")
    print(f"    SSR:    ER {er_ssr:.3f} vs ADX {adx_ssr:.3f}")

    print()


if __name__ == "__main__":
    main()
