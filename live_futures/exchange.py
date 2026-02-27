"""
IBKR + TradersPost composite exchange adapter.

Data:      IBKR TWS/Gateway historical bars + snapshots (read-only)
Execution: TradersPost webhook (market orders via HTTP POST)
Position:  Internal tracking (persisted to disk)

TradersPost connects to your FundedNext Tradovate account via their
platform — no Tradovate API keys required.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ib_insync requires an event loop at import time
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, util

from live_futures.config import LiveFuturesConfig
from live_futures.contract_utils import get_front_month, get_ib_contract

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Raised when exchange operations fail after retries."""


class FuturesExchange:
    """IBKR (data) + TradersPost (execution) composite adapter."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [1.0, 3.0, 10.0]

    def __init__(self, config: LiveFuturesConfig):
        self._config = config
        self._session = requests.Session()

        # IBKR (data only — read-only connection)
        self._ib = IB()
        self._ib_contract = None  # ib_insync Future, set in connect()

        # TradersPost
        self._webhook_url = config.traderspost_webhook_url

        # Symbol
        self._symbol = config.contract_root
        self._tp_ticker = config.contract_root  # TradersPost uses root symbol ("MNQ")
        self._contract_name = ""  # e.g. "MNQH6" (IBKR local symbol, for logging)
        self._timeframe_minutes = config.timeframe_minutes

        # Internal position tracking (TradersPost doesn't expose position)
        self._position: int = 0  # -1, 0, +1
        self._entry_price: Optional[float] = None
        self._position_state_path = Path(config.state_dir) / "position_state.json"
        self._load_position_state()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to IBKR for data, validate TradersPost webhook, resolve symbols."""
        # Resolve TradersPost contract name
        self._contract_name = get_front_month(self._config.contract_root)
        logger.info("Contract: %s", self._contract_name)

        # Connect to IBKR
        logger.info(
            "Connecting to IBKR at %s:%d (clientId=%d, readonly)...",
            self._config.ibkr_host,
            self._config.ibkr_port,
            self._config.ibkr_client_id,
        )
        self._ib.connect(
            self._config.ibkr_host,
            self._config.ibkr_port,
            clientId=self._config.ibkr_client_id,
            readonly=True,
        )
        logger.info("Connected to IBKR")

        # Qualify futures contract
        self._ib_contract = get_ib_contract(self._config.contract_root)
        qualified = self._ib.qualifyContracts(self._ib_contract)
        if not qualified:
            raise ExchangeError(
                f"Failed to qualify futures contract: {self._ib_contract}"
            )
        logger.info("Qualified IB contract: %s", self._ib_contract)

        # Validate TradersPost webhook URL
        if not self._webhook_url:
            logger.warning(
                "No TradersPost webhook URL configured — "
                "orders will be logged but NOT executed"
            )
        else:
            logger.info("TradersPost webhook configured")

        logger.info(
            "Internal position state: %+d (entry=%.2f)",
            self._position,
            self._entry_price or 0.0,
        )

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _ensure_connected(self) -> None:
        """Reconnect to IBKR if connection was lost."""
        if not self._ib.isConnected():
            logger.warning("IBKR connection lost, reconnecting...")
            self.connect()

    # ------------------------------------------------------------------
    # OHLCV Data (IBKR)
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, limit: int = 3) -> pd.DataFrame:
        """Fetch recent OHLCV bars from IBKR.

        Returns DataFrame with DatetimeIndex UTC, columns [open, high, low, close, volume].
        """
        self._ensure_connected()

        # Enough seconds to cover `limit` bars plus 1-hour buffer
        duration_seconds = limit * self._timeframe_minutes * 60 + 3600

        try:
            bars = self._ib.reqHistoricalData(
                self._ib_contract,
                endDateTime="",
                durationStr=f"{duration_seconds} S",
                barSizeSetting=f"{self._timeframe_minutes} mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
        except Exception as e:
            logger.error("Failed to fetch OHLCV: %s", e)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = util.df(bars)
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        df = df[~df.index.duplicated(keep="last")]

        if len(df) > limit:
            df = df.iloc[-limit:]

        return df

    def fetch_ohlcv_history(self, num_bars: int) -> pd.DataFrame:
        """Fetch a large history by paginating in 5-day chunks.

        Returns DataFrame with DatetimeIndex UTC, columns [open, high, low, close, volume].
        """
        self._ensure_connected()

        all_bars = []
        end = ""  # empty = now

        # CME: ~92 bars/day at 15-min. Each 5-day chunk gives ~460 bars.
        chunks_needed = max(1, num_bars // 400 + 2)

        for i in range(chunks_needed):
            try:
                bars = self._ib.reqHistoricalData(
                    self._ib_contract,
                    endDateTime=end,
                    durationStr="5 D",
                    barSizeSetting=f"{self._timeframe_minutes} mins",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                )
            except Exception as e:
                logger.error("History fetch error (chunk %d): %s", i + 1, e)
                break

            if not bars:
                break

            all_bars = bars + all_bars  # prepend older bars

            # Move end pointer backwards
            end_dt = bars[0].date
            if hasattr(end_dt, "strftime"):
                end = end_dt.strftime("%Y%m%d-%H:%M:%S")
            else:
                end = str(end_dt)

            logger.debug(
                "  Chunk %d: %d bars total (back to %s)", i + 1, len(all_bars), end_dt,
            )

            if len(all_bars) >= num_bars:
                break

            # IB pacing: ~2s between historical data requests
            time.sleep(2)

        if not all_bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = util.df(all_bars)
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if len(df) > num_bars:
            df = df.iloc[-num_bars:]

        logger.info(
            "Fetched %d historical bars (%s to %s)", len(df), df.index[0], df.index[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Quick price check (1-min bar — works without live data subscription)
    # ------------------------------------------------------------------

    def fetch_quick_price(self) -> float:
        """Fetch approximate current price using the latest 1-min bar.

        Uses historical data (not snapshot) so it works even without
        a live market data subscription. Returns 0.0 on failure.
        """
        self._ensure_connected()
        try:
            bars = self._ib.reqHistoricalData(
                self._ib_contract,
                endDateTime="",
                durationStr="120 S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            if bars:
                return float(bars[-1].close)
        except Exception as e:
            logger.debug("Quick price check failed: %s", e)
        return 0.0

    # ------------------------------------------------------------------
    # Quote (IBKR snapshot)
    # ------------------------------------------------------------------

    def fetch_best_bid_ask(self) -> dict:
        """Fetch current bid/ask from IBKR market data snapshot."""
        self._ensure_connected()
        try:
            ticker = self._ib.reqMktData(self._ib_contract, "", True, False)
            self._ib.sleep(2)
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            self._ib.cancelMktData(self._ib_contract)
            return {"bid": bid, "ask": ask}
        except Exception as e:
            logger.debug("Bid/ask fetch failed: %s", e)
            return {"bid": 0.0, "ask": 0.0}

    def fetch_last_price(self) -> float:
        """Get the last trade price from IBKR market data snapshot."""
        self._ensure_connected()
        try:
            ticker = self._ib.reqMktData(self._ib_contract, "", True, False)
            self._ib.sleep(2)
            price = ticker.last if ticker.last and ticker.last > 0 else 0.0
            self._ib.cancelMktData(self._ib_contract)
            return price
        except Exception as e:
            logger.debug("Last price fetch failed: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # Position Management (Internal tracking)
    # ------------------------------------------------------------------

    def get_position(self) -> dict:
        """Get current position from internal state."""
        if self._position > 0:
            side = "long"
        elif self._position < 0:
            side = "short"
        else:
            side = "flat"

        return {
            "side": side,
            "size": abs(self._position) * self._config.trade_size,
            "notional": 0.0,
            "entry_price": self._entry_price,
            "unrealized_pnl": 0.0,
        }

    def get_position_sign(self) -> int:
        """Get current position as {-1, 0, +1}."""
        return self._position

    # ------------------------------------------------------------------
    # Order Execution (TradersPost webhook)
    # ------------------------------------------------------------------

    def place_market_order(self, side: str, quantity: float) -> dict:
        """Send a market order via TradersPost webhook.

        side: 'buy' or 'sell'
        """
        qty = int(quantity)
        logger.info(
            "Placing %s market order: %d %s via TradersPost",
            side, qty, self._tp_ticker,
        )

        payload = {
            "ticker": self._tp_ticker,
            "action": side,  # "buy" or "sell"
            "orderType": "market",
            "quantity": qty,
        }

        success = self._send_webhook(payload)

        # Update internal position tracking
        last_price = self.fetch_last_price()
        if side == "buy":
            self._position = 1
            self._entry_price = last_price or self._entry_price
        else:
            self._position = -1
            self._entry_price = last_price or self._entry_price

        self._save_position_state()

        return {
            "id": f"tp_{int(time.time())}",
            "average": last_price if last_price > 0 else None,
            "status": "sent" if success else "failed",
            "cost": None,
        }

    def close_position(self) -> Optional[dict]:
        """Close the current position via TradersPost exit webhook."""
        if self._position == 0:
            logger.info("Already flat, nothing to close")
            return None

        logger.info("Closing position (%+d) via TradersPost", self._position)

        payload = {
            "ticker": self._tp_ticker,
            "action": "exit",
        }

        success = self._send_webhook(payload)

        last_price = self.fetch_last_price()

        result = {
            "id": f"tp_{int(time.time())}",
            "average": last_price if last_price > 0 else None,
            "status": "sent" if success else "failed",
            "cost": None,
        }

        # Update internal state
        self._position = 0
        self._entry_price = None
        self._save_position_state()

        return result

    def set_target_position(self, target: int) -> Optional[dict]:
        """Set position to target (-1, 0, +1) using TradersPost.

        For reversals (long→short or short→long), sends two webhooks:
          1. exit  — close current position
          2. buy/sell — open new position
        This avoids TradersPost treating a reversal as just an exit.
        """
        if target == self._position:
            return None

        # Reversal: exit first, then enter new direction
        is_reversal = self._position != 0 and target != 0
        if is_reversal:
            logger.info(
                "TradersPost: REVERSAL %+d → %+d (exit then %s)",
                self._position, target, "buy" if target > 0 else "sell",
            )
            exit_payload = {"ticker": self._tp_ticker, "action": "exit"}
            self._send_webhook(exit_payload)
            time.sleep(1)  # Brief pause between exit and new entry

        # Send the target action
        action_map = {1: "buy", -1: "sell", 0: "exit"}
        payload = {
            "ticker": self._tp_ticker,
            "action": action_map[target],
        }
        if target != 0:
            payload["sentiment"] = "bullish" if target > 0 else "bearish"
            payload["quantity"] = self._config.trade_size

        logger.info(
            "TradersPost: %s (sentiment=%s, qty=%s)",
            payload["action"],
            payload.get("sentiment", "N/A"),
            payload.get("quantity", "N/A"),
        )

        success = self._send_webhook(payload)
        last_price = self.fetch_last_price()

        # Update internal state
        self._position = target
        self._entry_price = last_price if target != 0 else None
        self._save_position_state()

        return {
            "id": f"tp_{int(time.time())}",
            "average": last_price if last_price > 0 else None,
            "status": "sent" if success else "failed",
            "cost": None,
        }

    # ------------------------------------------------------------------
    # TradersPost webhook
    # ------------------------------------------------------------------

    def _send_webhook(self, payload: dict) -> bool:
        """POST JSON to TradersPost webhook URL. Returns True on success."""
        if not self._webhook_url:
            logger.warning(
                "No webhook URL — order NOT sent: %s", json.dumps(payload),
            )
            return False

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=15,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code in (200, 201, 202):
                    logger.info(
                        "Webhook sent: %s → %d %s",
                        json.dumps(payload), resp.status_code, resp.text[:200],
                    )
                    return True
                else:
                    logger.warning(
                        "Webhook returned %d: %s",
                        resp.status_code, resp.text[:500],
                    )
                    last_error = f"HTTP {resp.status_code}"

            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Webhook connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
            except Exception as e:
                last_error = e
                logger.error("Webhook failed: %s", e)
                break

        logger.error(
            "Webhook failed after %d retries: %s", self.MAX_RETRIES, last_error,
        )
        return False

    # ------------------------------------------------------------------
    # Balance (estimated from internal tracking)
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """Estimated balance — not available via TradersPost.

        Returns starting balance from config. Actual balance must be
        checked manually in the FundedNext/Tradovate dashboard.
        """
        return {
            "total": self._config.fn_account_size,
            "free": self._config.fn_account_size,
            "used": 0.0,
        }

    # ------------------------------------------------------------------
    # Internal position state persistence
    # ------------------------------------------------------------------

    def _save_position_state(self) -> None:
        """Persist position state to disk."""
        state = {
            "position": self._position,
            "entry_price": self._entry_price,
            "contract_name": self._contract_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._position_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._position_state_path.write_text(json.dumps(state, indent=2))

    def _load_position_state(self) -> None:
        """Restore position state from disk."""
        if not self._position_state_path.exists():
            return
        try:
            state = json.loads(self._position_state_path.read_text())
            self._position = state.get("position", 0)
            self._entry_price = state.get("entry_price")
            logger.info(
                "Loaded position state: %+d (entry=%.2f)",
                self._position,
                self._entry_price or 0.0,
            )
        except Exception as e:
            logger.warning("Could not load position state: %s", e)
