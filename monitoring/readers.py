"""Read-only filesystem readers for monitoring.

Parses log files, state files, and heartbeat files.
No bot imports needed -- consumed by scripts/monitor.py and scripts/dashboard.py.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from monitoring.config import BotDescriptor

# ── Process detection ──────────────────────────────────────────────────


def is_process_running(keyword: str) -> Optional[int]:
    """Return PID of a Python process matching *keyword*, or None."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", keyword],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p for p in result.stdout.strip().split("\n") if p.strip()]
        return int(pids[0]) if pids else None
    except Exception:
        return None


# ── Log parsing ────────────────────────────────────────────────────────

_BAR_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| .+ \| INFO \| "
    r"Bar (\S+ \S+) \| close=([0-9.]+) \| target=([+-]?\d+) \| current=([+-]?\d+)"
)


def get_last_log_line(log_path: Path) -> Optional[str]:
    """Read the last non-empty line from a log file (reads tail 4 KB)."""
    if not log_path.exists():
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            f.seek(-min(4096, size), 2)
            data = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in data.split("\n") if ln.strip()]
        return lines[-1] if lines else None
    except Exception:
        return None


def get_log_freshness(log_path: Path) -> Optional[float]:
    """Seconds since the log file was last modified."""
    if not log_path.exists():
        return None
    try:
        mtime = os.path.getmtime(log_path)
        return datetime.now(timezone.utc).timestamp() - mtime
    except Exception:
        return None


def get_last_bar_from_log(log_path: Path) -> Optional[dict]:
    """Parse the last ``Bar ...`` line from a log file (reads tail 64 KB).

    Returns dict with keys: log_time, bar_time, close, target, current.
    """
    if not log_path.exists():
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(-min(65536, size), 2)
            data = f.read().decode("utf-8", errors="replace")
        matches = list(_BAR_RE.finditer(data))
        if not matches:
            return None
        m = matches[-1]
        return {
            "log_time": m.group(1),
            "bar_time": m.group(2),
            "close": float(m.group(3)),
            "target": int(m.group(4)),
            "current": int(m.group(5)),
        }
    except Exception:
        return None


# ── State file readers ─────────────────────────────────────────────────


def read_json_file(path: Path) -> Optional[dict]:
    """Safely read and parse a JSON file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_model_age_hours(model_state: dict) -> Optional[float]:
    """Hours since model was trained."""
    ts = model_state.get("training_timestamp")
    if not ts:
        return None
    try:
        trained = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - trained).total_seconds() / 3600
    except Exception:
        return None


def get_heartbeat_age_sec(heartbeat: dict) -> Optional[float]:
    """Seconds since last heartbeat."""
    ts = heartbeat.get("timestamp")
    if not ts:
        return None
    try:
        hb_time = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - hb_time).total_seconds()
    except Exception:
        return None


# ── Trade log ──────────────────────────────────────────────────────────


def get_trades_today(trade_log_path: Path) -> list[dict]:
    """Read today's trades from a JSONL trade log."""
    if not trade_log_path.exists():
        return []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trades: list[dict] = []
    try:
        with open(trade_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("timestamp", "").startswith(today):
                    trades.append(entry)
    except Exception:
        pass
    return trades


# ── Composite bot status ───────────────────────────────────────────────

_POS_LABEL = {1: "long", -1: "short", 0: "flat"}


def get_bot_status(bot: BotDescriptor, root: Path) -> dict:
    """Gather full status for one bot.  Returns a flat dict."""
    log_path = root / bot.log_file
    model_path = root / bot.model_state_file
    hb_path = root / bot.heartbeat_file

    pid = is_process_running(bot.process_keyword)
    log_freshness = get_log_freshness(log_path)
    last_bar = get_last_bar_from_log(log_path)

    heartbeat = read_json_file(hb_path)
    hb_age = get_heartbeat_age_sec(heartbeat) if heartbeat else None

    model = read_json_file(model_path)
    model_age = get_model_age_hours(model) if model else None

    position_state = None
    if bot.position_state_file:
        position_state = read_json_file(root / bot.position_state_file)

    trades_today: list[dict] = []
    if bot.trade_log_file:
        trades_today = get_trades_today(root / bot.trade_log_file)

    # Best-effort position from heartbeat > position_state > last log bar
    position = None
    position_label = "unknown"
    if heartbeat and heartbeat.get("position") is not None:
        position = heartbeat["position"]
        position_label = heartbeat.get("position_label", _POS_LABEL.get(position, "unknown"))
    elif position_state and position_state.get("position") is not None:
        position = position_state["position"]
        position_label = _POS_LABEL.get(position, "unknown")
    elif last_bar:
        position = last_bar.get("current")
        position_label = _POS_LABEL.get(position, "unknown")

    return {
        "bot": bot,
        "pid": pid,
        "alive": pid is not None,
        "log_freshness_sec": log_freshness,
        "last_bar": last_bar,
        "heartbeat": heartbeat,
        "heartbeat_age_sec": hb_age,
        "model": model,
        "model_age_hours": model_age,
        "position": position,
        "position_label": position_label,
        "position_state": position_state,
        "trades_today": trades_today,
    }
