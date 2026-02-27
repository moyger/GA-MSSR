"""
Slack webhook notifier for the GA-MSSR live bot.

Sends trade alerts, startup/shutdown, retrain, and error notifications.
Uses only stdlib (urllib) — no extra dependencies.
Silently no-ops if no webhook URL is configured.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger("ga_mssr_bot")

_POS_LABEL = {1: "long", -1: "short", 0: "flat"}


class SlackNotifier:
    def __init__(self, webhook_url: str = ""):
        self._url = webhook_url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def send(self, text: str) -> None:
        """Post a message to Slack. Never raises."""
        if not self._url:
            return
        try:
            payload = json.dumps({"text": text}).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.debug("Slack notification failed: %s", e)

    def notify_trade(
        self,
        side: str,
        qty: float,
        symbol: str,
        price: Optional[float],
        current: int,
        target: int,
        balance: Optional[float] = None,
    ) -> None:
        price_str = f" @ ${price:,.2f}" if price else ""
        bal_str = f" | Balance: ${balance:,.2f}" if balance else ""
        curr = _POS_LABEL.get(current, str(current))
        tgt = _POS_LABEL.get(target, str(target))
        self.send(
            f"*GA-MSSR | {side.upper()} {qty:.4f} {symbol}{price_str}*\n"
            f"Signal: {curr} -> {tgt}{bal_str}"
        )

    def notify_startup(
        self,
        symbol: str,
        trade_size: float,
        leverage: int,
        testnet: bool,
        balance: float,
        position: int,
    ) -> None:
        mode = "TESTNET" if testnet else "MAINNET"
        pos = _POS_LABEL.get(position, str(position))
        self.send(
            f"*GA-MSSR Bot Started*\n"
            f"Symbol: {symbol} | Size: {trade_size} | Leverage: {leverage}x\n"
            f"Balance: ${balance:,.2f} | Position: {pos}\n"
            f"Mode: {mode}"
        )

    def notify_shutdown(self, position_side: str, position_size: float) -> None:
        if position_side == "flat":
            self.send("*GA-MSSR Bot Stopped* | Position: flat")
        else:
            self.send(
                f"*GA-MSSR Bot Stopped* | Open {position_side} {position_size:.4f}\n"
                f"_Position remains open on exchange_"
            )

    def notify_retrain(self, ssr: float, timestamp: str) -> None:
        self.send(
            f"*GA-MSSR Model Retrained*\n"
            f"SSR: {ssr:.4f} | Trained: {timestamp}"
        )

    def notify_daily_limit(self, pnl: float, limit: float) -> None:
        self.send(
            f"*GA-MSSR Daily Loss Limit Hit*\n"
            f"P&L: ${pnl:,.2f} | Limit: ${limit:,.2f}\n"
            f"_Trading halted until next UTC day_"
        )

    def notify_error(self, message: str) -> None:
        self.send(f"*GA-MSSR Error*\n{message}")
