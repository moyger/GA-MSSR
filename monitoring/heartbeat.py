"""Heartbeat writer + Slack sender for bot health monitoring.

Each bot creates one HeartbeatMonitor and calls tick() on every loop
iteration.  Internally throttled:
  - File write  every *write_interval_sec*  (default 60 s)
  - Slack send  every *interval_sec*        (default 3600 s)

Uses only stdlib -- no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_POS_LABEL = {1: "long", -1: "short", 0: "flat"}


class HeartbeatMonitor:
    """Lightweight heartbeat that writes state to disk and pings Slack."""

    def __init__(
        self,
        heartbeat_path: Path,
        bot_name: str,
        slack_notifier=None,
        interval_sec: int = 3600,
        write_interval_sec: int = 60,
    ):
        self._path = heartbeat_path
        self._bot_name = bot_name
        self._slack = slack_notifier
        self._interval = interval_sec
        self._write_interval = write_interval_sec
        self._last_slack_send: float = 0.0
        self._last_file_write: float = 0.0
        self._bars_today: int = 0
        self._bars_total: int = 0
        self._last_bar_time: Optional[str] = None
        self._start_time: float = time.time()
        self._current_date: str = ""

    # ── public API ──

    def tick(
        self,
        position: int = 0,
        daily_pnl: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Call on every loop iteration.  Internally throttled."""
        now = time.time()

        if now - self._last_file_write >= self._write_interval:
            self._write_heartbeat(position, daily_pnl, extra)
            self._last_file_write = now

        if self._slack and now - self._last_slack_send >= self._interval:
            self._send_slack_heartbeat(position, daily_pnl)
            self._last_slack_send = now

    def record_bar(self, bar_time: str) -> None:
        """Record that a new bar was processed (call from _on_bar)."""
        self._last_bar_time = bar_time
        self._bars_total += 1

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._bars_today = 0
        self._bars_today += 1

    # ── internals ──

    def _write_heartbeat(
        self,
        position: int,
        daily_pnl: Optional[float],
        extra: Optional[dict],
    ) -> None:
        state = {
            "bot_name": self._bot_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_sec": int(time.time() - self._start_time),
            "position": position,
            "position_label": _POS_LABEL.get(position, str(position)),
            "last_bar_time": self._last_bar_time,
            "bars_today": self._bars_today,
            "bars_total": self._bars_total,
            "daily_pnl": daily_pnl,
        }
        if extra:
            state.update(extra)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug("Heartbeat write failed: %s", e)

    def _send_slack_heartbeat(
        self, position: int, daily_pnl: Optional[float]
    ) -> None:
        pos_label = _POS_LABEL.get(position, str(position))
        uptime_h = (time.time() - self._start_time) / 3600
        pnl_str = f" | PnL: ${daily_pnl:,.2f}" if daily_pnl is not None else ""

        msg = (
            f"*{self._bot_name} Heartbeat*\n"
            f"Position: {pos_label} | Last bar: {self._last_bar_time or 'N/A'}\n"
            f"Bars today: {self._bars_today} | Uptime: {uptime_h:.1f}h{pnl_str}"
        )
        try:
            self._slack.send(msg)
        except Exception as e:
            logger.debug("Slack heartbeat failed: %s", e)
