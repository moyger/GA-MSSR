#!/usr/bin/env python3
"""
End-to-end GA-MSSR backtest script.

Usage:
    .venv/bin/python scripts/run_backtest.py <path_to_nq_1m_csv>

Example:
    .venv/bin/python scripts/run_backtest.py data/NQ_1min_2024.csv

The CSV must have columns: timestamp, open, high, low, close, volume

Pipeline:
    1. Load NQ 1-minute OHLCV data
    2. Apply wavelet denoising (db4)
    3. Grid-search optimal parameters for 16 trading rules
    4. GA-optimize rule weights (maximize SSR with drawdown constraint)
    5. Run NautilusTrader backtest with trained model
    6. Print performance report
"""
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_nq_data
from data.pipeline import build_denoised_dataset
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig
from strategies.backtest import run_backtest
from strategies.khushi_strategy import KhushiStrategyConfig

from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.data import BarType


def main(filepath: str):
    print("=" * 60)
    print("GA-MSSR NQ Futures Backtest")
    print("=" * 60)

    # ---- Phase 1: Load & Denoise ----
    print(f"\n[1/5] Loading data from {filepath}...")
    df = build_denoised_dataset(filepath, denoise_columns="close")
    print(f"      {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")

    # ---- Phase 2: Train Rule Parameters ----
    print("\n[2/5] Training 16 rule parameters (grid search)...")
    t0 = time.time()
    rule_params = train_rule_params(df)
    print(f"      Done in {time.time() - t0:.1f}s")
    for i, p in enumerate(rule_params, 1):
        print(f"      Rule {i:2d}: {p}")

    # ---- Phase 3: Generate Features ----
    print("\n[3/5] Generating rule feature matrix...")
    features = get_rule_features(df, rule_params)
    print(f"      Feature matrix: {features.shape} (bars x features)")

    # ---- Phase 4: GA Optimization ----
    print("\n[4/5] Running GA optimization (SSR fitness)...")
    ga_config = GAMSSRConfig(
        sol_per_pop=20,
        num_parents_mating=10,
        num_generations=200,
        random_seed=42,
    )
    print(f"      Population: {ga_config.sol_per_pop}, "
          f"Generations: {ga_config.num_generations}")

    t0 = time.time()
    ga = GAMSSR(ga_config)
    result = ga.fit(features)
    ga_time = time.time() - t0

    print(f"      Done in {ga_time:.1f}s")
    print(f"      Best SSR: {result.best_fitness:.6f}")
    print(f"      Max drawdown: {result.ssr_result.max_drawdown:.6f}")
    print(f"      Total return: {result.ssr_result.total_return:.6f}")
    print(f"      Weights: {np.round(result.best_weights, 4)}")

    # ---- Phase 5: NautilusTrader Backtest ----
    print("\n[5/5] Running NautilusTrader backtest...")

    # Convert rule_params for config serialization
    config_params = []
    for p in rule_params:
        if isinstance(p, tuple):
            config_params.append(list(p))
        elif isinstance(p, (int, float)):
            config_params.append([p])
        else:
            config_params.append(p)

    # Warmup needs enough bars for longest indicator lookback.
    # Default 200, but cap at half the data so backtesting can actually trade.
    warmup = min(200, len(df) // 2)

    strategy_config = KhushiStrategyConfig(
        instrument_id=InstrumentId.from_str("NQ.SIM"),
        bar_type=BarType.from_str("NQ.SIM-1-MINUTE-LAST-EXTERNAL"),
        trade_size=Decimal("1"),
        ga_weights=result.best_weights.tolist(),
        rule_params=config_params,
        warmup_bars=warmup,
        denoise=True,
        order_id_tag="001",
    )
    print(f"      Warmup bars: {warmup}")

    t0 = time.time()
    engine = run_backtest(df, strategy_config, starting_balance=100_000)
    bt_time = time.time() - t0
    print(f"      Done in {bt_time:.1f}s")

    # ---- Reports ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", None,
        "display.width", 300,
    ):
        fills = engine.trader.generate_order_fills_report()
        positions = engine.trader.generate_positions_report()
        account = engine.trader.generate_account_report(Venue("SIM"))

        print(f"\nTotal orders filled: {len(fills)}")
        print(f"Total positions: {len(positions)}")

        if len(account) > 0:
            print("\n--- Account Report ---")
            print(account.to_string())

        if len(positions) > 0:
            print("\n--- Positions Report ---")
            # Show key columns
            cols = [c for c in ["entry", "side", "avg_px_open", "avg_px_close",
                                "realized_return", "realized_pnl", "duration_ns",
                                "ts_opened", "ts_closed"] if c in positions.columns]
            print(positions[cols].to_string())

            # Summary stats
            if "realized_pnl" in positions.columns:
                pnl_values = positions["realized_pnl"].apply(
                    lambda x: float(str(x).replace(" USD", ""))
                    if isinstance(x, str) else float(x)
                )
                total_pnl = pnl_values.sum()
                winners = (pnl_values > 0).sum()
                losers = (pnl_values < 0).sum()
                win_rate = winners / len(pnl_values) * 100 if len(pnl_values) > 0 else 0

                print(f"\n--- Summary ---")
                print(f"Total PnL:    ${total_pnl:,.2f}")
                print(f"Winners:      {winners}")
                print(f"Losers:       {losers}")
                print(f"Win Rate:     {win_rate:.1f}%")
                if losers > 0:
                    avg_win = pnl_values[pnl_values > 0].mean()
                    avg_loss = abs(pnl_values[pnl_values < 0].mean())
                    print(f"Avg Win:      ${avg_win:,.2f}")
                    print(f"Avg Loss:     ${avg_loss:,.2f}")
                    print(f"Profit Factor: {(pnl_values[pnl_values > 0].sum() / abs(pnl_values[pnl_values < 0].sum())):.2f}")

    engine.reset()
    engine.dispose()
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/run_backtest.py <path_to_nq_csv>")
        print("\nQuick test with fixture data:")
        print("  .venv/bin/python scripts/run_backtest.py tests/fixtures/sample_nq_1m.csv")
        sys.exit(1)

    main(sys.argv[1])
