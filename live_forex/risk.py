"""Risk management for the IBKR forex bot.

Enforces FTMO-style rules:
  - Custom daily loss limit ($500)
  - Max total loss (10% of account = $10K)
  - Profit target tracking (10% = $10K)

Persists trailing drawdown state to disk so it survives restarts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from live_forex.config import LiveForexConfig

logger = logging.getLogger(__name__)


class ForexRiskManager:
    """Tracks daily PnL and enforces FTMO risk limits."""

    def __init__(self, config: LiveForexConfig):
        self._config = config
        self._daily_pnl: float = 0.0
        self._daily_stopped: bool = False
        self._current_date: str = ""
        self._realized_pnl_today: float = 0.0
        self._trade_count_today: int = 0

        # Cumulative P&L tracking
        self._total_pnl: float = 0.0

        # Trailing drawdown (EOD high-water mark)
        self._eod_high_balance: float = config.ftmo_account_size
        self._trailing_floor: float = (
            config.ftmo_account_size - config.ftmo_max_total_loss
        )

        # State persistence
        self._risk_state_path = Path(config.state_dir) / "risk_state.json"
        self._load_risk_state()

    # ------------------------------------------------------------------
    # Daily tracking
    # ------------------------------------------------------------------

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
        self._total_pnl += realized_pnl
        logger.info(
            "Trade recorded: PnL=$%.2f | Daily=$%.2f | Total=$%.2f",
            realized_pnl, self._daily_pnl, self._total_pnl,
        )

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update daily PnL with current unrealized PnL."""
        self._daily_pnl = self._realized_pnl_today + unrealized_pnl

    # ------------------------------------------------------------------
    # Daily loss limit
    # ------------------------------------------------------------------

    @property
    def is_daily_stopped(self) -> bool:
        return self._daily_stopped

    def check_daily_limit(self) -> bool:
        """Returns True if daily loss limit is breached."""
        if self._daily_stopped:
            return True
        if self._daily_pnl <= -self._config.ftmo_daily_loss_limit:
            self._daily_stopped = True
            logger.warning(
                "DAILY LOSS LIMIT HIT: $%.2f (limit: $%.2f). "
                "Trading halted until next UTC day.",
                self._daily_pnl, self._config.ftmo_daily_loss_limit,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Total loss / equity floor
    # ------------------------------------------------------------------

    def check_total_loss(self, current_balance: float) -> bool:
        """Check if balance has dropped below the equity floor.

        Returns True if max total loss breached (challenge failed).
        """
        if current_balance <= self._trailing_floor:
            logger.critical(
                "TOTAL LOSS BREACHED: balance=$%.2f floor=$%.2f "
                "(EOD high=$%.2f, max loss=$%.2f)",
                current_balance, self._trailing_floor,
                self._eod_high_balance, self._config.ftmo_max_total_loss,
            )
            return True
        return False

    def check_profit_target(self, current_balance: float) -> bool:
        """Check if profit target has been reached."""
        profit = current_balance - self._config.ftmo_account_size
        if profit >= self._config.ftmo_profit_target:
            logger.info(
                "PROFIT TARGET REACHED: $%.2f (target: $%.2f)",
                profit, self._config.ftmo_profit_target,
            )
            return True
        return False

    def update_eod_balance(self, balance: float) -> None:
        """Update the EOD high-water mark. Floor only moves up."""
        if balance > self._eod_high_balance:
            self._eod_high_balance = balance
            self._trailing_floor = balance - self._config.ftmo_max_total_loss
            logger.info(
                "EOD high updated: $%.2f → new floor: $%.2f",
                balance, self._trailing_floor,
            )
        self._save_risk_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_risk_state(self) -> None:
        state = {
            "eod_high_balance": self._eod_high_balance,
            "trailing_floor": self._trailing_floor,
            "total_pnl": self._total_pnl,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._risk_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._risk_state_path.write_text(json.dumps(state, indent=2))

    def _load_risk_state(self) -> None:
        if not self._risk_state_path.exists():
            return
        try:
            state = json.loads(self._risk_state_path.read_text())
            self._eod_high_balance = state["eod_high_balance"]
            self._trailing_floor = state["trailing_floor"]
            self._total_pnl = state.get("total_pnl", 0.0)
            logger.info(
                "Loaded risk state: EOD high=$%.2f, floor=$%.2f, total=$%.2f",
                self._eod_high_balance, self._trailing_floor, self._total_pnl,
            )
        except Exception as e:
            logger.warning("Could not load risk state: %s", e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def trade_count_today(self) -> int:
        return self._trade_count_today

    @property
    def eod_high_balance(self) -> float:
        return self._eod_high_balance

    @property
    def trailing_floor(self) -> float:
        return self._trailing_floor
