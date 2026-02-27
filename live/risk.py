"""
Risk management for the live trading bot.

Enforces daily loss limit and maximum position notional.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from live.config import LiveConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Tracks daily PnL and enforces risk limits."""

    def __init__(self, config: LiveConfig):
        self._config = config
        self._daily_pnl: float = 0.0
        self._daily_stopped: bool = False
        self._current_date: str = ""
        self._realized_pnl_today: float = 0.0
        self._trade_count_today: int = 0

    def check_new_day(self) -> None:
        """Reset daily tracking if the UTC date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._current_date:
                logger.info(
                    "Day rolled: %s -> %s | PnL=$%.2f | Trades=%d",
                    self._current_date, today,
                    self._daily_pnl, self._trade_count_today,
                )
            self._current_date = today
            self._daily_pnl = 0.0
            self._daily_stopped = False
            self._realized_pnl_today = 0.0
            self._trade_count_today = 0

    def record_trade(self, realized_pnl: float) -> None:
        """Record a completed trade's realized PnL."""
        self._realized_pnl_today += realized_pnl
        self._trade_count_today += 1
        self._daily_pnl = self._realized_pnl_today
        logger.info(
            "Trade recorded: PnL=$%.2f | Daily total=$%.2f",
            realized_pnl, self._daily_pnl,
        )

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update daily PnL with current unrealized PnL."""
        self._daily_pnl = self._realized_pnl_today + unrealized_pnl

    @property
    def is_daily_stopped(self) -> bool:
        return self._daily_stopped

    def check_daily_limit(self) -> bool:
        """Returns True if trading should stop (limit breached)."""
        if self._daily_stopped:
            return True
        if self._daily_pnl <= -self._config.max_daily_loss_usd:
            self._daily_stopped = True
            logger.warning(
                "DAILY LOSS LIMIT HIT: $%.2f (limit: $%.2f). "
                "Trading halted until next UTC day.",
                self._daily_pnl, self._config.max_daily_loss_usd,
            )
            return True
        return False

    def check_position_size(self, notional_usd: float) -> bool:
        """Returns True if the position is within limits."""
        if notional_usd > self._config.max_position_notional_usd:
            logger.warning(
                "Position notional $%.2f exceeds limit $%.2f",
                notional_usd, self._config.max_position_notional_usd,
            )
            return False
        return True

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trade_count_today(self) -> int:
        return self._trade_count_today
