"""
Risk management for the Tradovate futures bot.

Enforces FundedNext prop firm rules:
  - Daily loss limit ($1,200 for $50K account)
  - Trailing EOD max loss ($2,500)
  - Consistency rule (no single day > 40% of profit target)
  - Position size limits

Persists trailing drawdown state to disk so it survives restarts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from live_futures.config import LiveFuturesConfig

logger = logging.getLogger(__name__)


class FuturesRiskManager:
    """Tracks daily PnL and enforces FundedNext risk limits."""

    def __init__(self, config: LiveFuturesConfig):
        self._config = config
        self._daily_pnl: float = 0.0
        self._daily_stopped: bool = False
        self._current_date: str = ""
        self._realized_pnl_today: float = 0.0
        self._trade_count_today: int = 0

        # FundedNext trailing drawdown
        self._eod_high_balance: float = config.fn_account_size
        self._trailing_floor: float = config.fn_account_size - config.fn_trailing_max_loss

        # Try to restore persisted state
        self._risk_state_path = Path(config.state_dir) / "risk_state.json"
        self._load_risk_state()

    # ------------------------------------------------------------------
    # Daily tracking
    # ------------------------------------------------------------------

    def check_new_day(self, current_balance: float | None = None) -> None:
        """Reset daily tracking if the UTC date has changed.

        If current_balance is provided and the day has rolled,
        update the EOD high-water mark for trailing drawdown.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._current_date:
                logger.info(
                    "Day rolled: %s -> %s | PnL=$%.2f | Trades=%d",
                    self._current_date, today,
                    self._daily_pnl, self._trade_count_today,
                )
                # Update EOD high-water mark on day roll
                if current_balance is not None:
                    self.update_eod_balance(current_balance)
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

    # ------------------------------------------------------------------
    # FundedNext daily loss limit
    # ------------------------------------------------------------------

    @property
    def is_daily_stopped(self) -> bool:
        return self._daily_stopped

    def check_daily_limit(self) -> bool:
        """Returns True if daily loss limit is breached."""
        if self._daily_stopped:
            return True
        if self._daily_pnl <= -self._config.fn_daily_loss_limit:
            self._daily_stopped = True
            logger.warning(
                "DAILY LOSS LIMIT HIT: $%.2f (limit: $%.2f). "
                "Trading halted until next UTC day.",
                self._daily_pnl, self._config.fn_daily_loss_limit,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # FundedNext trailing EOD drawdown
    # ------------------------------------------------------------------

    def check_trailing_drawdown(self, current_balance: float) -> bool:
        """Check if the trailing max loss has been breached.

        Returns True if balance has dropped below the trailing floor.
        This would mean the FundedNext challenge is failed.
        """
        if current_balance <= self._trailing_floor:
            logger.critical(
                "TRAILING DRAWDOWN BREACHED: balance=$%.2f floor=$%.2f "
                "(EOD high=$%.2f, max loss=$%.2f)",
                current_balance, self._trailing_floor,
                self._eod_high_balance, self._config.fn_trailing_max_loss,
            )
            return True
        return False

    def update_eod_balance(self, balance: float) -> None:
        """Update the EOD high-water mark at end of trading day.

        The trailing floor moves UP (never down) as the account grows.
        """
        if balance > self._eod_high_balance:
            self._eod_high_balance = balance
            self._trailing_floor = balance - self._config.fn_trailing_max_loss
            logger.info(
                "EOD high updated: $%.2f → new floor: $%.2f",
                balance, self._trailing_floor,
            )
        self._save_risk_state()

    # ------------------------------------------------------------------
    # FundedNext consistency rule
    # ------------------------------------------------------------------

    def check_consistency_rule(self) -> bool:
        """Check if daily profit exceeds 40% of profit target.

        Returns True if we should stop trading to stay consistent.
        The 40% cap for a $50K account with $2,500 target = $1,000/day.
        """
        if self._config.fn_profit_target <= 0 or self._config.fn_consistency_pct <= 0:
            return False
        max_daily_profit = self._config.fn_profit_target * self._config.fn_consistency_pct
        if self._daily_pnl >= max_daily_profit:
            logger.warning(
                "CONSISTENCY RULE: daily profit $%.2f reached cap $%.2f (%.0f%% of target). "
                "Stopping trading for the day.",
                self._daily_pnl, max_daily_profit,
                self._config.fn_consistency_pct * 100,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Position size check
    # ------------------------------------------------------------------

    def check_position_size(self, contracts: int) -> bool:
        """Returns True if the position is within limits."""
        if contracts > self._config.max_position_contracts:
            logger.warning(
                "Position size %d contracts exceeds limit %d",
                contracts, self._config.max_position_contracts,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_risk_state(self) -> None:
        """Persist trailing drawdown state to disk."""
        state = {
            "eod_high_balance": self._eod_high_balance,
            "trailing_floor": self._trailing_floor,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._risk_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._risk_state_path.write_text(json.dumps(state, indent=2))

    def _load_risk_state(self) -> None:
        """Restore trailing drawdown state from disk."""
        if not self._risk_state_path.exists():
            return
        try:
            state = json.loads(self._risk_state_path.read_text())
            self._eod_high_balance = state["eod_high_balance"]
            self._trailing_floor = state["trailing_floor"]
            logger.info(
                "Loaded risk state: EOD high=$%.2f, floor=$%.2f",
                self._eod_high_balance, self._trailing_floor,
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
    def trade_count_today(self) -> int:
        return self._trade_count_today

    @property
    def eod_high_balance(self) -> float:
        return self._eod_high_balance

    @property
    def trailing_floor(self) -> float:
        return self._trailing_floor
