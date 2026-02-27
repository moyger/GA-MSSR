"""IBKR forex exchange adapter using ib_insync.

Data:      IBKR TWS/Gateway historical bars + streaming
Execution: IBKR market orders on IDEALPRO
Position:  Read from IBKR portfolio + internal state backup
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# ib_insync requires an event loop at import time (Python 3.14+)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Forex, LimitOrder, MarketOrder, util

from live_forex.config import LiveForexConfig

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Raised when exchange operations fail."""


@dataclass
class FillStats:
    """Statistics about a limit order fill attempt."""
    filled: bool = False
    attempts: int = 0
    original_price: float = 0.0
    final_price: Optional[float] = None
    fill_price: Optional[float] = None
    elapsed_sec: float = 0.0
    prices_tried: list[float] = field(default_factory=list)

    @property
    def slippage(self) -> Optional[float]:
        if self.fill_price and self.original_price:
            return self.fill_price - self.original_price
        return None


class IBKRForexExchange:
    """IBKR forex exchange adapter — data + execution via ib_insync."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [1.0, 3.0, 10.0]

    def __init__(self, config: LiveForexConfig):
        self._config = config
        self._ib = IB()
        self._contract = None
        self._pair = config.forex_pair.upper()
        self._units = config.order_units

        # Internal position backup (survives reconnections)
        self._position: int = 0  # -1, 0, +1
        self._entry_price: Optional[float] = None
        self._position_state_path = Path(config.state_dir) / "position_state.json"
        self._load_position_state()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS/Gateway and qualify the forex contract."""
        logger.info(
            "Connecting to IBKR at %s:%d (clientId=%d)...",
            self._config.ibkr_host,
            self._config.ibkr_port,
            self._config.ibkr_client_id,
        )

        self._ib.connect(
            self._config.ibkr_host,
            self._config.ibkr_port,
            clientId=self._config.ibkr_client_id,
            readonly=False,
        )
        logger.info("Connected to IBKR")

        # Create and qualify forex contract
        self._contract = Forex(self._pair)
        qualified = self._ib.qualifyContracts(self._contract)
        if not qualified:
            raise ExchangeError(
                f"Failed to qualify forex contract: {self._pair}"
            )
        logger.info("Qualified contract: %s", self._contract)

        # Check current position from IBKR
        ibkr_pos = self._get_ibkr_position()
        logger.info("IBKR position: %+d units", ibkr_pos)

        # Reconcile with internal state
        if ibkr_pos != 0 and self._position == 0:
            self._position = 1 if ibkr_pos > 0 else -1
            logger.info("Reconciled: IBKR has position, updating internal to %+d", self._position)
        elif ibkr_pos == 0 and self._position != 0:
            logger.warning(
                "Position mismatch: IBKR=flat, internal=%+d. Resetting to flat.",
                self._position,
            )
            self._position = 0
            self._entry_price = None
            self._save_position_state()

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self._ib.isConnected()

    def _ensure_connected(self) -> None:
        """Reconnect if connection was lost."""
        if not self._ib.isConnected():
            logger.warning("IBKR connection lost, reconnecting...")
            self.connect()

    # ------------------------------------------------------------------
    # OHLCV Data
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, limit: int = 3) -> pd.DataFrame:
        """Fetch recent 15-min OHLCV bars from IBKR."""
        self._ensure_connected()

        try:
            bars = self._ib.reqHistoricalData(
                self._contract,
                endDateTime="",
                durationStr=f"{limit * 15 + 60} S",  # extra buffer
                barSizeSetting="15 mins",
                whatToShow="MIDPOINT",
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
        """Fetch a large history by paginating in 5-day chunks."""
        self._ensure_connected()

        all_bars = []
        end = ""  # empty = now

        # Each 5-day chunk gives ~480 bars (96 bars/day × 5)
        chunks_needed = max(1, num_bars // 400 + 2)

        for i in range(chunks_needed):
            try:
                bars = self._ib.reqHistoricalData(
                    self._contract,
                    endDateTime=end,
                    durationStr="5 D",
                    barSizeSetting="15 mins",
                    whatToShow="MIDPOINT",
                    useRTH=False,
                    formatDate=1,
                )
            except Exception as e:
                logger.error("History fetch error (chunk %d): %s", i + 1, e)
                break

            if not bars:
                break

            all_bars = bars + all_bars

            # Move end pointer backwards
            end_dt = bars[0].date
            if hasattr(end_dt, "strftime"):
                end = end_dt.strftime("%Y%m%d-%H:%M:%S")
            else:
                end = str(end_dt)

            logger.debug("  Chunk %d: %d bars total (back to %s)", i + 1, len(all_bars), end_dt)

            if len(all_bars) >= num_bars:
                break

            # IB rate limiting
            time.sleep(2)

        if not all_bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = util.df(all_bars)
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Trim to requested bars
        if len(df) > num_bars:
            df = df.iloc[-num_bars:]

        logger.info("Fetched %d historical bars (%s to %s)", len(df), df.index[0], df.index[-1])
        return df

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def _get_ibkr_position(self) -> int:
        """Get actual position from IBKR in units (positive=long, negative=short)."""
        positions = self._ib.positions()
        for pos in positions:
            if (pos.contract.secType == "CASH" and
                pos.contract.symbol == self._pair[:3] and
                pos.contract.currency == self._pair[3:]):
                return int(pos.position)
        return 0

    def get_position(self) -> dict:
        """Get current position."""
        ibkr_pos = self._get_ibkr_position()
        if ibkr_pos > 0:
            side = "long"
        elif ibkr_pos < 0:
            side = "short"
        else:
            side = "flat"

        return {
            "side": side,
            "size": abs(ibkr_pos),
            "entry_price": self._entry_price,
            "unrealized_pnl": 0.0,
        }

    def get_position_sign(self) -> int:
        """Get current position as {-1, 0, +1}."""
        ibkr_pos = self._get_ibkr_position()
        if ibkr_pos > 0:
            return 1
        elif ibkr_pos < 0:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Order Execution
    # ------------------------------------------------------------------

    def place_market_order(self, side: str, units: int) -> dict:
        """Place a market order. side: 'BUY' or 'SELL'."""
        self._ensure_connected()

        logger.info("Placing %s market order: %d %s", side, units, self._pair)

        order = MarketOrder(side, units)
        order.tif = "GTC"
        if self._config.ibkr_account:
            order.account = self._config.ibkr_account

        trade = self._ib.placeOrder(self._contract, order)

        # Wait for fill (up to 30 seconds)
        timeout = 30
        start_time = time.time()
        while not trade.isDone() and time.time() - start_time < timeout:
            self._ib.sleep(0.5)

        if trade.isDone():
            fill_price = trade.orderStatus.avgFillPrice
            status = trade.orderStatus.status
            logger.info(
                "Order filled: %s %d @ %.5f (status=%s)",
                side, units, fill_price, status,
            )
        else:
            fill_price = 0.0
            # Check if IBKR rejected/cancelled the order
            order_status = trade.orderStatus.status
            if order_status in ("Cancelled", "Inactive"):
                status = "rejected"
                error_msg = trade.log[-1].message if trade.log else "unknown"
                logger.error(
                    "Order REJECTED by IBKR (status=%s): %s",
                    order_status, error_msg,
                )
            else:
                status = "timeout"
                logger.warning("Order did not fill within %ds (status=%s)",
                               timeout, order_status)

        # Only update internal state on successful fill
        if fill_price > 0:
            if side == "BUY":
                self._position = 1
            else:
                self._position = -1
            self._entry_price = fill_price
            self._save_position_state()

        return {
            "id": str(trade.order.orderId),
            "average": fill_price if fill_price > 0 else None,
            "status": status,
        }

    def close_position(self) -> Optional[dict]:
        """Close the current position."""
        ibkr_pos = self._get_ibkr_position()
        if ibkr_pos == 0:
            logger.info("Already flat, nothing to close")
            self._position = 0
            self._entry_price = None
            self._save_position_state()
            return None

        side = "SELL" if ibkr_pos > 0 else "BUY"
        units = abs(ibkr_pos)

        logger.info("Closing position: %s %d %s", side, units, self._pair)
        result = self.place_market_order(side, units)

        self._position = 0
        self._entry_price = None
        self._save_position_state()

        return result

    def set_target_position(self, target: int) -> Optional[dict]:
        """Set position to target (-1, 0, +1).

        Handles close + reversal in one flow.
        """
        current = self.get_position_sign()
        if target == current:
            return None

        # Close current position first
        if current != 0:
            self.close_position()
            self._ib.sleep(0.5)  # brief pause between close and open

        # Open new position
        if target != 0:
            side = "BUY" if target > 0 else "SELL"
            return self.place_market_order(side, self._units)

        return {"id": "flat", "average": None, "status": "closed"}

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    def fetch_last_price(self) -> float:
        """Get the current mid price."""
        self._ensure_connected()
        try:
            ticker = self._ib.reqMktData(self._contract, "", True, False)
            self._ib.sleep(2)  # wait for snapshot
            mid = (ticker.bid + ticker.ask) / 2 if ticker.bid > 0 and ticker.ask > 0 else 0.0
            self._ib.cancelMktData(self._contract)
            return mid
        except Exception as e:
            logger.debug("Last price fetch failed: %s", e)
            return 0.0

    def fetch_best_bid_ask(self) -> dict:
        """Fetch current best bid and ask prices.

        Returns dict with 'bid' and 'ask' floats.
        """
        self._ensure_connected()
        try:
            ticker = self._ib.reqMktData(self._contract, "", True, False)
            self._ib.sleep(2)  # wait for snapshot
            bid = float(ticker.bid) if ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask > 0 else 0.0
            self._ib.cancelMktData(self._contract)
            return {"bid": bid, "ask": ask}
        except Exception as e:
            logger.debug("Bid/ask fetch failed: %s", e)
            return {"bid": 0.0, "ask": 0.0}

    def place_limit_order_persistent(
        self,
        side: str,
        units: int,
        price: float,
        max_attempts: int = 6,
        timeout_per_attempt: float = 10.0,
        poll_interval: float = 0.5,
    ) -> tuple[Optional[dict], FillStats]:
        """Place a limit order with re-pricing on timeout. No market fallback.

        Posts at *price*, polls for fill. If not filled within
        *timeout_per_attempt*, cancels and re-prices at the current
        best bid/ask. Repeats up to *max_attempts* times.

        Returns (order_dict, fill_stats). order_dict is None if not filled.
        """
        self._ensure_connected()
        stats = FillStats(original_price=price)
        t_start = time.time()

        for attempt in range(1, max_attempts + 1):
            stats.attempts = attempt
            stats.prices_tried.append(price)

            logger.info(
                "[Reprice %d/%d] %s limit %d %s @ %.5f",
                attempt, max_attempts, side, units, self._pair, price,
            )

            order = LimitOrder(side, units, price)
            order.tif = "GTC"
            if self._config.ibkr_account:
                order.account = self._config.ibkr_account

            trade = self._ib.placeOrder(self._contract, order)

            # Poll for fill
            deadline = time.time() + timeout_per_attempt
            while time.time() < deadline:
                self._ib.sleep(poll_interval)

                if trade.isDone():
                    fill_price = trade.orderStatus.avgFillPrice
                    order_status = trade.orderStatus.status

                    if fill_price > 0 and order_status in ("Filled",):
                        stats.filled = True
                        stats.fill_price = fill_price
                        stats.final_price = price
                        stats.elapsed_sec = time.time() - t_start
                        logger.info(
                            "[Reprice %d/%d] FILLED: avg=%.5f elapsed=%.1fs",
                            attempt, max_attempts, fill_price, stats.elapsed_sec,
                        )

                        # Update internal state
                        if side == "BUY":
                            self._position = 1
                        else:
                            self._position = -1
                        self._entry_price = fill_price
                        self._save_position_state()

                        return {
                            "id": str(trade.order.orderId),
                            "average": fill_price,
                            "status": "filled",
                        }, stats

                    if order_status in ("Cancelled", "Inactive"):
                        error_msg = trade.log[-1].message if trade.log else "unknown"
                        logger.warning(
                            "[Reprice %d/%d] Order rejected/cancelled: %s",
                            attempt, max_attempts, error_msg,
                        )
                        break

            # Timeout — cancel the order
            if not trade.isDone():
                try:
                    self._ib.cancelOrder(trade.order)
                    self._ib.sleep(1)  # wait for cancel to process
                    logger.info("[Reprice %d/%d] Cancelled (timeout)", attempt, max_attempts)
                except Exception as e:
                    logger.warning("[Reprice %d/%d] Cancel failed: %s", attempt, max_attempts, e)

                # Race condition guard: check if filled during cancel
                if trade.orderStatus.status == "Filled" and trade.orderStatus.avgFillPrice > 0:
                    fill_price = trade.orderStatus.avgFillPrice
                    stats.filled = True
                    stats.fill_price = fill_price
                    stats.final_price = price
                    stats.elapsed_sec = time.time() - t_start
                    logger.info(
                        "[Reprice %d/%d] Filled during cancel: avg=%.5f",
                        attempt, max_attempts, fill_price,
                    )
                    if side == "BUY":
                        self._position = 1
                    else:
                        self._position = -1
                    self._entry_price = fill_price
                    self._save_position_state()
                    return {
                        "id": str(trade.order.orderId),
                        "average": fill_price,
                        "status": "filled",
                    }, stats

            # Re-fetch bid/ask for next attempt
            if attempt < max_attempts:
                try:
                    bba = self.fetch_best_bid_ask()
                    new_price = bba["bid"] if side == "BUY" else bba["ask"]
                    if new_price and new_price > 0:
                        price = new_price
                        logger.info("[Reprice %d/%d] New price: %.5f", attempt, max_attempts, price)
                    else:
                        logger.warning("[Reprice %d/%d] Bad bid/ask, retrying same price", attempt, max_attempts)
                except Exception as e:
                    logger.warning("[Reprice %d/%d] Bid/ask fetch failed: %s", attempt, max_attempts, e)

        # Exhausted all attempts
        stats.elapsed_sec = time.time() - t_start
        logger.warning(
            "Limit order NOT filled after %d attempts over %.1fs. Prices: %s",
            max_attempts, stats.elapsed_sec, [round(p, 5) for p in stats.prices_tried],
        )
        return None, stats

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """Get account balance from IBKR."""
        self._ensure_connected()
        try:
            summary = self._ib.accountSummary()
            nav = 0.0
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    nav = float(item.value)
                    break
            return {"total": nav, "free": nav, "used": 0.0}
        except Exception as e:
            logger.debug("Balance fetch failed: %s", e)
            return {
                "total": self._config.ftmo_account_size,
                "free": self._config.ftmo_account_size,
                "used": 0.0,
            }

    # ------------------------------------------------------------------
    # Internal position state persistence
    # ------------------------------------------------------------------

    def _save_position_state(self) -> None:
        state = {
            "position": self._position,
            "entry_price": self._entry_price,
            "pair": self._pair,
            "units": self._units,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._position_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._position_state_path.write_text(json.dumps(state, indent=2))

    def _load_position_state(self) -> None:
        if not self._position_state_path.exists():
            return
        try:
            state = json.loads(self._position_state_path.read_text())
            self._position = state.get("position", 0)
            self._entry_price = state.get("entry_price")
            logger.info(
                "Loaded position state: %+d (entry=%.5f)",
                self._position,
                self._entry_price or 0.0,
            )
        except Exception as e:
            logger.warning("Could not load position state: %s", e)
