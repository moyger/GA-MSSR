"""Monitoring configuration -- registry of all bot artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BotDescriptor:
    """Describes one running bot for monitoring purposes."""

    name: str                     # Human-readable, e.g. "Futures (MNQ)"
    bot_id: str                   # Machine-safe key, e.g. "futures"
    log_file: str                 # Relative to project root
    state_dir: str                # Relative to project root
    model_state_file: str         # Relative to project root
    heartbeat_file: str           # Relative to project root
    position_state_file: str      # Relative to project root, "" if none
    trade_log_file: str           # Relative to project root, "" if none
    log_name: str                 # Logger name in log lines (for grep)
    process_keyword: str          # Unique substring in `ps` output


@dataclass(frozen=True)
class MonitorConfig:
    """Immutable registry of all monitored bots."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
    )

    bots: tuple[BotDescriptor, ...] = (
        BotDescriptor(
            name="Futures (MNQ)",
            bot_id="futures",
            log_file="logs/ga_mssr_futures.log",
            state_dir="live_futures/state/fundednext",
            model_state_file="live_futures/state/fundednext/model_state.json",
            heartbeat_file="live_futures/state/fundednext/heartbeat.json",
            position_state_file="live_futures/state/fundednext/position_state.json",
            trade_log_file="live_futures/state/fundednext/trade_log.jsonl",
            log_name="ga_mssr_futures",
            process_keyword="run_live_futures.py",
        ),
        BotDescriptor(
            name="Bybit (ETH/USDT)",
            bot_id="bybit",
            log_file="logs/ga_mssr_live.log",
            state_dir="live/state",
            model_state_file="live/state/model_state.json",
            heartbeat_file="live/state/heartbeat.json",
            position_state_file="",
            trade_log_file="live/state/trade_log.jsonl",
            log_name="ga_mssr_bot",
            process_keyword="run_live.py",
        ),
        BotDescriptor(
            name="Forex (AUDUSD)",
            bot_id="forex",
            log_file="logs/ga_mssr_forex.log",
            state_dir="live_forex/state",
            model_state_file="live_forex/state/model_state.json",
            heartbeat_file="live_forex/state/heartbeat.json",
            position_state_file="",
            trade_log_file="live_forex/state/trade_log.jsonl",
            log_name="ga_mssr_forex",
            process_keyword="run_live_forex.py",
        ),
    )
