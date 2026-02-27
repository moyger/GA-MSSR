#!/usr/bin/env python3
"""
Generate detailed walk-forward trade report with equity chart and trade log.

Usage:
    .venv/bin/python scripts/generate_report.py data/futures/NQ_1min_IB.csv 15min
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from optimizers.ssr import SSR

BARS_PER_DAY = {
    "1min": 1380, "5min": 276, "15min": 92, "30min": 46, "1h": 23,
}

# MNQ specs
MNQ_POINT_VALUE = 2.0   # $2 per point
MNQ_COMMISSION_RT = 1.42  # Tradovate Membership round-trip


def run_detailed_walk_forward(df, timeframe, ga_config):
    """Run walk-forward and capture per-bar positions + trade details."""
    from strategies.khushi_rules import train_rule_params, get_rule_features

    bpd = BARS_PER_DAY.get(timeframe, 92)
    train_size = bpd * 20  # 4 weeks
    test_size = bpd * 5    # 1 week

    ssr = SSR()
    n = len(df)
    all_rows = []  # per-bar data
    fold_summaries = []

    start = 0
    fold_num = 0
    while start + train_size + test_size <= n:
        fold_num += 1
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]

        # Train
        rule_params = train_rule_params(train_df, periods=[3, 7, 15, 27])
        train_features = get_rule_features(train_df, rule_params)
        test_features = get_rule_features(test_df, rule_params)

        if len(train_features) < 10 or len(test_features) < 5:
            start += test_size
            continue

        ga = GAMSSR(ga_config)
        result = ga.fit(train_features)
        weights = result.best_weights

        # Predict on test data
        logr = test_features["logr"].values
        signal_cols = [c for c in test_features.columns if c.startswith("rule_")]
        signals = test_features[signal_cols].values

        raw_position = signals @ weights
        position = np.sign(raw_position)  # discretized -1/0/+1
        port_return = position * logr

        # Record per-bar data
        for i, (idx, row) in enumerate(test_df.iloc[-len(test_features):].iterrows()):
            all_rows.append({
                "timestamp": idx,
                "fold": fold_num,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "position": int(position[i]),
                "raw_signal": float(raw_position[i]),
                "bar_return": float(logr[i]),
                "port_return": float(port_return[i]),
            })

        oos_ssr = ssr.calculate(port_return)
        fold_summaries.append({
            "fold": fold_num,
            "ssr": oos_ssr.ssr,
            "return": float(port_return.sum()),
            "bars": len(test_features),
            "train_start": train_df.index[0],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
        })

        print(f"  Fold {fold_num:2d}: SSR={oos_ssr.ssr:8.2f}  Return={port_return.sum()*100:6.2f}%")
        start += test_size

    bars_df = pd.DataFrame(all_rows)
    folds_df = pd.DataFrame(fold_summaries)
    return bars_df, folds_df


def extract_trades(bars_df):
    """Extract individual trades from per-bar position data."""
    trades = []
    current_pos = 0
    entry_time = None
    entry_price = None
    trade_pnl = 0.0

    for _, row in bars_df.iterrows():
        pos = row["position"]
        if pos != current_pos:
            # Close existing trade
            if current_pos != 0 and entry_time is not None:
                exit_price = row["close"]
                if current_pos == 1:
                    pnl_points = exit_price - entry_price
                else:
                    pnl_points = entry_price - exit_price
                pnl_dollar = pnl_points * MNQ_POINT_VALUE - MNQ_COMMISSION_RT

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": row["timestamp"],
                    "side": "LONG" if current_pos == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_points": round(pnl_points, 2),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "commission": MNQ_COMMISSION_RT,
                    "fold": row["fold"],
                })

            # Open new trade
            if pos != 0:
                entry_time = row["timestamp"]
                entry_price = row["close"]
            else:
                entry_time = None
                entry_price = None

            current_pos = pos

    return pd.DataFrame(trades)


def generate_charts(bars_df, trades_df, folds_df, timeframe, output_dir):
    """Generate equity curve and trade distribution charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate cumulative equity
    bars_df = bars_df.copy()
    bars_df["cum_return"] = bars_df["port_return"].cumsum()
    bars_df["equity_mnq"] = bars_df["cum_return"] * bars_df["close"].mean() * MNQ_POINT_VALUE

    # Subtract cumulative commissions
    position_changes = (bars_df["position"].diff().fillna(0) != 0).astype(int)
    bars_df["cum_commission"] = position_changes.cumsum() * MNQ_COMMISSION_RT
    bars_df["equity_net"] = bars_df["equity_mnq"] - bars_df["cum_commission"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    fig.suptitle(f"GA-MSSR Walk-Forward Report — {timeframe} MNQ", fontsize=16, fontweight="bold")

    # 1. Equity Curve
    ax1 = axes[0]
    timestamps = pd.to_datetime(bars_df["timestamp"])
    ax1.plot(timestamps, bars_df["equity_mnq"], label="Gross PnL", color="#2196F3", linewidth=1)
    ax1.plot(timestamps, bars_df["equity_net"], label="Net PnL (after commissions)", color="#4CAF50", linewidth=1.5)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Mark fold boundaries
    for _, fold in folds_df.iterrows():
        ax1.axvline(x=fold["test_start"], color="orange", alpha=0.3, linestyle=":")

    ax1.set_title("Cumulative Equity Curve (1 MNQ Contract)")
    ax1.set_ylabel("PnL ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.tick_params(axis="x", rotation=45)

    # 2. Per-Trade PnL
    ax2 = axes[1]
    if len(trades_df) > 0:
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades_df["pnl_dollar"]]
        ax2.bar(range(len(trades_df)), trades_df["pnl_dollar"], color=colors, width=1.0, alpha=0.7)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title(f"Per-Trade PnL ({len(trades_df):,} trades)")
        ax2.set_ylabel("PnL ($)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

    # 3. PnL Distribution
    ax3 = axes[2]
    if len(trades_df) > 0:
        ax3.hist(trades_df["pnl_dollar"], bins=80, color="#2196F3", alpha=0.7, edgecolor="white")
        ax3.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax3.axvline(x=trades_df["pnl_dollar"].mean(), color="green", linestyle="--",
                    linewidth=1.5, label=f"Mean: ${trades_df['pnl_dollar'].mean():.2f}")
        ax3.set_title("Trade PnL Distribution")
        ax3.set_xlabel("PnL ($)")
        ax3.set_ylabel("Count")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Per-Fold Returns
    ax4 = axes[3]
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in folds_df["return"]]
    ax4.bar(folds_df["fold"], folds_df["return"] * 100, color=colors, alpha=0.8)
    ax4.set_title("Per-Fold OOS Return (%)")
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Return (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / f"walk_forward_{timeframe}_report.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {chart_path}")
    return chart_path


def print_trade_summary(trades_df):
    """Print trade statistics."""
    if len(trades_df) == 0:
        print("No trades.")
        return

    winners = trades_df[trades_df["pnl_dollar"] > 0]
    losers = trades_df[trades_df["pnl_dollar"] < 0]
    flat = trades_df[trades_df["pnl_dollar"] == 0]

    total_pnl = trades_df["pnl_dollar"].sum()
    total_commission = trades_df["commission"].sum()

    print(f"\n{'='*60}")
    print(f"TRADE SUMMARY (MNQ, {MNQ_COMMISSION_RT}/RT commission)")
    print(f"{'='*60}")
    print(f"  Total trades:       {len(trades_df):,}")
    print(f"  Winners:            {len(winners):,} ({len(winners)/len(trades_df)*100:.1f}%)")
    print(f"  Losers:             {len(losers):,} ({len(losers)/len(trades_df)*100:.1f}%)")
    print(f"  Breakeven:          {len(flat):,}")
    print(f"\n  Total PnL (net):    ${total_pnl:,.2f}")
    print(f"  Total commissions:  ${total_commission:,.2f}")
    print(f"  Gross PnL:          ${total_pnl + total_commission:,.2f}")
    print(f"\n  Avg win:            ${winners['pnl_dollar'].mean():.2f}" if len(winners) > 0 else "")
    print(f"  Avg loss:           ${losers['pnl_dollar'].mean():.2f}" if len(losers) > 0 else "")
    print(f"  Largest win:        ${winners['pnl_dollar'].max():.2f}" if len(winners) > 0 else "")
    print(f"  Largest loss:       ${losers['pnl_dollar'].min():.2f}" if len(losers) > 0 else "")

    if len(winners) > 0 and len(losers) > 0:
        profit_factor = winners["pnl_dollar"].sum() / abs(losers["pnl_dollar"].sum())
        print(f"\n  Profit factor:      {profit_factor:.2f}")
        print(f"  Avg win/avg loss:   {abs(winners['pnl_dollar'].mean() / losers['pnl_dollar'].mean()):.2f}")

    # Daily stats
    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["entry_time"]).dt.date
    daily = trades_df.groupby("date").agg(
        trades=("pnl_dollar", "count"),
        pnl=("pnl_dollar", "sum"),
    )
    print(f"\n  Trading days:       {len(daily)}")
    print(f"  Avg trades/day:     {daily['trades'].mean():.1f}")
    print(f"  Avg daily PnL:      ${daily['pnl'].mean():.2f}")
    print(f"  Profitable days:    {(daily['pnl'] > 0).sum()} / {len(daily)} "
          f"({(daily['pnl'] > 0).sum()/len(daily)*100:.1f}%)")
    print(f"  Best day:           ${daily['pnl'].max():.2f}")
    print(f"  Worst day:          ${daily['pnl'].min():.2f}")


def main(filepath, timeframe="15min"):
    print("=" * 60)
    print(f"GA-MSSR Walk-Forward Trade Report ({timeframe})")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close", timeframe=timeframe)
    print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # GA config
    ga_config = GAMSSRConfig(
        sol_per_pop=20, num_parents_mating=10,
        num_generations=200, random_seed=42,
    )

    # Run walk-forward with detail capture
    print(f"\nRunning walk-forward...")
    t0 = time.time()
    bars_df, folds_df = run_detailed_walk_forward(df, timeframe, ga_config)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Extract trades
    print(f"\nExtracting trades...")
    trades_df = extract_trades(bars_df)
    print(f"  {len(trades_df):,} trades extracted")

    # Print summary
    print_trade_summary(trades_df)

    # Save trade log CSV
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"trades_{timeframe}.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"\nTrade log saved: {csv_path}")

    # Generate charts
    chart_path = generate_charts(bars_df, trades_df, folds_df, timeframe, output_dir)

    # Save daily summary
    trades_df_copy = trades_df.copy()
    trades_df_copy["date"] = pd.to_datetime(trades_df_copy["entry_time"]).dt.date
    daily = trades_df_copy.groupby("date").agg(
        trades=("pnl_dollar", "count"),
        gross_pnl=("pnl_points", lambda x: (x.sum() * MNQ_POINT_VALUE)),
        net_pnl=("pnl_dollar", "sum"),
        winners=("pnl_dollar", lambda x: (x > 0).sum()),
        losers=("pnl_dollar", lambda x: (x < 0).sum()),
    ).round(2)
    daily["cum_pnl"] = daily["net_pnl"].cumsum().round(2)
    daily_path = output_dir / f"daily_summary_{timeframe}.csv"
    daily.to_csv(daily_path)
    print(f"Daily summary saved: {daily_path}")

    print(f"\nAll reports saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/generate_report.py <csv> [timeframe]")
        sys.exit(1)
    filepath = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15min"
    main(filepath, timeframe)
