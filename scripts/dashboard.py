#!/usr/bin/env python3
"""GA-MSSR Live Dashboard -- real-time terminal UI.

Shows all 3 bots' status in a continuously-updating table
plus the most recent log line from each.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --refresh 5

Requires: pip install rich   (included in dev extras)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from monitoring.config import MonitorConfig
from monitoring.readers import get_bot_status, get_last_log_line


# ── helpers ──

def _fmt(sec: float | None) -> str:
    if sec is None:
        return "N/A"
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


# ── builders ──

def _build_table(statuses: list[dict]) -> Table:
    table = Table(
        title="GA-MSSR Trading Bots",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        expand=True,
    )
    table.add_column("Bot", style="bold", width=20)
    table.add_column("Process", width=14)
    table.add_column("Position", width=10)
    table.add_column("Last Bar", width=26)
    table.add_column("Signal", width=12)
    table.add_column("Bars", width=6, justify="right")
    table.add_column("Log Age", width=8, justify="right")
    table.add_column("Model", width=10, justify="right")
    table.add_column("HB Age", width=8, justify="right")
    table.add_column("PnL", width=10, justify="right")

    for s in statuses:
        bot = s["bot"]

        # Process
        proc = Text(f"PID {s['pid']}", style="green") if s["alive"] else Text("DEAD", style="bold red")

        # Position
        pos = s["position_label"]
        pos_style = {"long": "green", "short": "red", "flat": "dim"}.get(pos, "")
        position_text = Text(pos, style=pos_style)

        # Last bar
        bar = s["last_bar"]
        bar_str = bar["bar_time"] if bar else "N/A"

        # Signal
        sig_str = f"T={bar['target']:+d} C={bar['current']:+d}" if bar else "N/A"

        # Log freshness
        lf = s["log_freshness_sec"]
        log_style = "red" if lf and lf > 120 else ""

        # Model age
        ma = s["model_age_hours"]
        if ma is not None:
            model_str = f"{ma:.1f}h"
            model_style = "red" if ma > 24 else ""
        else:
            model_str = "N/A"
            model_style = "dim"

        # Heartbeat age
        hb_age = s["heartbeat_age_sec"]
        hb_str = _fmt(hb_age) if hb_age is not None else "N/A"
        hb_style = "red" if hb_age and hb_age > 120 else ""

        # PnL
        hb = s.get("heartbeat") or {}
        pnl = hb.get("daily_pnl")
        if pnl is not None:
            pnl_str = f"${pnl:,.2f}"
            pnl_style = "green" if pnl >= 0 else "red"
        else:
            pnl_str = "N/A"
            pnl_style = "dim"

        # Bars today (from heartbeat first, else 0)
        bars = hb.get("bars_today", 0) if hb else 0

        table.add_row(
            bot.name,
            proc,
            position_text,
            bar_str,
            sig_str,
            str(bars),
            Text(_fmt(lf), style=log_style),
            Text(model_str, style=model_style),
            Text(hb_str, style=hb_style),
            Text(pnl_str, style=pnl_style),
        )

    return table


def _build_log_panel(statuses: list[dict], root: Path) -> Panel:
    lines = []
    for s in statuses:
        bot = s["bot"]
        log_path = root / bot.log_file
        last_line = get_last_log_line(log_path)
        if last_line:
            display = last_line[:120] + ("..." if len(last_line) > 120 else "")
        else:
            display = "(no log data)"
        lines.append(f"[bold]{bot.name}[/bold]: {display}")
    return Panel("\n".join(lines), title="Latest Log Lines", border_style="blue")


def _build(config: MonitorConfig) -> Layout:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    statuses = [get_bot_status(bot, config.project_root) for bot in config.bots]
    alive = sum(1 for s in statuses if s["alive"])
    total = len(statuses)
    colour = "green" if alive == total else "red"

    layout = Layout()
    layout.split_column(
        Layout(
            Text(
                f" GA-MSSR Dashboard | {now} | {alive}/{total} bots running ",
                style="bold white on blue",
                justify="center",
            ),
            name="header",
            size=1,
        ),
        Layout(_build_table(statuses), name="table", size=len(statuses) + 5),
        Layout(_build_log_panel(statuses, config.project_root), name="logs"),
    )
    return layout


# ── main ──

def main() -> None:
    parser = argparse.ArgumentParser(description="GA-MSSR Live Dashboard")
    parser.add_argument(
        "--refresh", type=int, default=10,
        help="Refresh interval in seconds (default: 10)",
    )
    args = parser.parse_args()

    config = MonitorConfig()
    console = Console()
    console.print("[bold]GA-MSSR Dashboard[/bold] starting... (Ctrl+C to exit)\n")

    with Live(
        _build(config),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            while True:
                time.sleep(args.refresh)
                live.update(_build(config))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
