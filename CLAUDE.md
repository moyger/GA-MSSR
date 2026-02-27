# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GA-MSSR is an algorithmic trading system for E-mini Nasdaq 100 (NQ) futures implementing Dr. Matloob Khushi's GA-MSSR (Genetic Algorithm Maximizing Sharpe and Sterling Ratio) framework. It combines wavelet denoising, 16 technical trading rules, and genetic algorithm optimization.

## Commands

```bash
# Install (editable, with all extras)
pip install -e .[nautilus,ibkr,optimize,live,dev]

# Run all tests
pytest

# Run a single test file
pytest tests/test_ssr.py

# Run with coverage
pytest --cov=. tests/

# Run backtest pipeline
python scripts/run_backtest.py

# Run walk-forward analysis
python scripts/run_walk_forward.py

# Run live futures bot (CME Micro NQ)
python scripts/run_live_futures.py
```

## Architecture

### Data Flow

Raw OHLCV → `data/loader.py` → `data/wavelet_filter.py` (db4 denoise) → `strategies/khushi_rules.py` (16-rule signals) → `optimizers/ga_mssr.py` (GA weight optimization) → `strategies/khushi_strategy.py` (position computation) → NautilusTrader backtest or TradersPost live execution

### Key Modules

- **`data/`** — Data loading (`loader.py`), wavelet denoising (`wavelet_filter.py`), and end-to-end pipeline (`pipeline.py`). Contract configs: `nq_config.py`, `mnq_config.py`, `forex_config.py`.
- **`strategies/`** — `indicators.py` (20+ technical indicators), `khushi_rules.py` (16 signal rules), `khushi_strategy.py` (contains both `KhushiSignalEngine` for pure-Python signal computation and `KhushiStrategy` for NautilusTrader integration), `backtest.py` (NautilusTrader runner).
- **`optimizers/`** — `ssr.py` (Sharpe-Sterling Ratio fitness calculator), `ga_mssr.py` (PyGAD-based GA optimizer producing a 16-dim weight vector).
- **`live_futures/`** — Primary live trading bot for CME Micro NQ via TradersPost webhooks + Polygon.io data. Includes FundedNext risk management, CME trading hours guard, Slack notifications, and daily retraining.
- **`live/`** — Legacy Bybit crypto bot using CCXT.
- **`scripts/`** — Entry points for backtests, walk-forward analysis, live bots, data downloads, and reports.
- **`reference/`** — Original `zzzac/Rule-based-forex-trading-system` repo used as source of truth for the 16 rules and GA structure.

### Design Patterns

- **KhushiSignalEngine** is decoupled from NautilusTrader — usable standalone for position computation in both live and backtest contexts.
- **Configuration objects** use frozen dataclasses (`NQConfig`, `GAMSSRConfig`, `KhushiStrategyConfig`, `LiveFuturesConfig`).
- **Two-layer execution**: offline training (`train_rule_params()` → `GAMSSR.fit()` → best_weights) and online inference (`KhushiSignalEngine.compute_position()` with trained weights).

### The 16 Trading Rules (`khushi_rules.py`)

Rules 1-6: Moving average crossovers (MA, EMA, DEMA, TEMA variants). Rule 7: Stochastic crossover. Rule 8: Vortex indicator. Rule 9: Ichimoku. Rules 10-11: RSI/CCI threshold. Rules 12-13: RSI/CCI dual threshold. Rules 14-16: Channel breakouts (Keltner, Donchian, Bollinger).

### GA Optimizer Defaults

Population: 20, Generations: 200, Mutation probability: 0.1, Parent selection: Steady-State, Crossover: Single-point. Fitness function: SS Ratio. Constraint: max drawdown 15%.

## NQ Contract Parameters

- Tick size: 0.25, Tick value: $5.00 (E-mini) / $0.50 (Micro)
- Commission: $2.40 round-turn
- Hard stop: $400 daily loss per contract

## Live Bot Management

Three bots run concurrently. The Bybit bot is managed by launchd; futures and forex are started manually.

```bash
# --- Bybit ETHUSD (launchd-managed, auto-restarts) ---
launchctl stop com.ga-mssr.live        # Stop
launchctl start com.ga-mssr.live       # Start
launchctl unload ~/Library/LaunchAgents/com.ga-mssr.live.plist  # Disable auto-start
launchctl load ~/Library/LaunchAgents/com.ga-mssr.live.plist    # Re-enable auto-start

# --- NQ Futures (manual, background) ---
.venv/bin/python scripts/run_live_futures.py --env .env.propfirm.fundednext &

# --- AUDUSD Forex (manual, background) ---
.venv/bin/python scripts/run_live_forex.py --pair AUDUSD &
```

**IMPORTANT**: Never use `kill` + manual restart for the Bybit bot — launchd will auto-respawn it (KeepAlive=true), creating duplicate instances that double-trade the same account. Always use `launchctl stop/start`.

### ER Filter Configuration

All bots use the Efficiency Ratio (ER) trend filter to skip choppy markets. Parameters are set per-instrument in each bot's config:

| Bot | Config File | ER Period | ER Threshold |
|-----|-------------|-----------|--------------|
| NQ Futures | `live_futures/config.py` | 14 | 0.40 |
| ETHUSD | `live/config.py` | 14 | 0.40 |
| AUDUSD | `live_forex/config.py` | 20 | 0.35 |

### PID Lockfiles

Each bot creates a lockfile at startup to prevent duplicate instances:
- `/tmp/ga_mssr_live.pid` — Bybit ETHUSD
- `/tmp/ga_mssr_futures.pid` — NQ Futures
- `/tmp/ga_mssr_forex.pid` — AUDUSD Forex

## Environment

- Python 3.10+ required
- `.env` / `.env.futures` hold API credentials (never commit)
- Optional dependency groups: `nautilus`, `ibkr`, `optimize`, `live`, `dev`
- `conftest.py` adds project root to `sys.path`
