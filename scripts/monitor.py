#!/usr/bin/env python3
"""GA-MSSR Bot Health Check CLI.

Run anytime to see all 3 bots' status at a glance.
Reads only log files and state files -- no running bot required.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.config import MonitorConfig
from monitoring.readers import get_bot_status


# ── helpers ──

def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


G = "\033[92m"   # green
R = "\033[91m"   # red
D = "\033[2m"    # dim
X = "\033[0m"    # reset


def _print_bot(status: dict) -> None:
    bot = status["bot"]
    alive = status["alive"]

    print(f"\n{'=' * 60}")
    print(f"  {bot.name}")
    print(f"{'=' * 60}")

    # Process
    if alive:
        print(f"  Process:       {G}RUNNING (PID {status['pid']}){X}")
    else:
        print(f"  Process:       {R}DEAD{X}")

    # Log freshness
    lf = status["log_freshness_sec"]
    warn = f" {R}(STALE!){X}" if lf and lf > 120 else ""
    print(f"  Log freshness: {_fmt_duration(lf)} ago{warn}")

    # Last bar
    bar = status["last_bar"]
    if bar:
        print(f"  Last bar:      {bar['bar_time']} | close={bar['close']}")
        print(f"  Last signal:   target={bar['target']:+d} current={bar['current']:+d}")
    else:
        print(f"  Last bar:      {D}N/A{X}")

    # Position
    print(f"  Position:      {status['position_label']}")

    # Position state (futures)
    ps = status.get("position_state")
    if ps:
        print(f"  Contract:      {ps.get('contract_name', '?')} | Entry: {ps.get('entry_price', 0)}")

    # Model
    ma = status["model_age_hours"]
    if ma is not None:
        m_warn = f" {R}(STALE! >24h){X}" if ma > 24 else ""
        ssr = status["model"].get("best_fitness", 0) if status["model"] else 0
        print(f"  Model age:     {ma:.1f}h{m_warn} | SSR={ssr:.4f}")
    else:
        print(f"  Model age:     {D}N/A{X}")

    # Heartbeat
    hb = status["heartbeat"]
    if hb:
        hb_age = status["heartbeat_age_sec"]
        hb_warn = f" {R}(MISSED!){X}" if hb_age and hb_age > 120 else ""
        uptime = _fmt_duration(hb.get("uptime_sec"))
        pnl = hb.get("daily_pnl")
        pnl_str = f"${pnl:,.2f}" if pnl is not None else "N/A"
        print(f"  Heartbeat:     {_fmt_duration(hb_age)} ago{hb_warn} | Uptime: {uptime}")
        print(f"  Daily PnL:     {pnl_str}")
    else:
        print(f"  Heartbeat:     {D}No heartbeat file yet{X}")

    # Trades today
    trades = status["trades_today"]
    if trades:
        print(f"  Trades today:  {len(trades)}")


# ── main ──

def main() -> None:
    parser = argparse.ArgumentParser(description="GA-MSSR Bot Health Check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    config = MonitorConfig()
    now = datetime.now(timezone.utc)

    statuses = [get_bot_status(bot, config.project_root) for bot in config.bots]

    if args.json:
        out = []
        for s in statuses:
            d = {k: v for k, v in s.items() if k != "bot"}
            d["bot_name"] = s["bot"].name
            d["bot_id"] = s["bot"].bot_id
            out.append(d)
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"\nGA-MSSR Bot Health Check | {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    for s in statuses:
        _print_bot(s)

    alive = sum(1 for s in statuses if s["alive"])
    total = len(statuses)
    colour = G if alive == total else R
    print(f"\n{'=' * 60}")
    print(f"  Summary: {colour}{alive}/{total} bots running{X}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
