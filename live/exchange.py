"""
Bybit exchange wrapper using ccxt.

Handles testnet/mainnet switching, retry with exponential backoff,
OHLCV pagination, position queries, and market order execution.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from live.config import LiveConfig

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Raised when exchange operations fail after retries."""


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


class BybitExchange:
    """Thin wrapper around ccxt.bybit for the GA-MSSR live bot."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [1.0, 3.0, 10.0]
    MAX_CANDLES_PER_REQUEST = 200

    def __init__(self, config: LiveConfig):
        self._config = config
        self._exchange = ccxt.bybit({
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })
        if config.testnet:
            self._exchange.set_sandbox_mode(True)
        self._symbol = config.symbol

    def connect(self) -> None:
        """Verify connectivity: load markets, set leverage, set one-way mode."""
        self._exchange.load_markets()
        logger.info(
            "Connected to Bybit %s | Symbol: %s",
            "TESTNET" if self._config.testnet else "MAINNET",
            self._symbol,
        )
        # Switch to one-way position mode (required for our long/short/flat logic)
        try:
            self._exchange.set_position_mode(False, self._symbol)  # False = one-way
            logger.info("Position mode set to one-way")
        except Exception as e:
            logger.warning("Could not set position mode (may already be one-way): %s", e)
        try:
            self._exchange.set_leverage(self._config.leverage, self._symbol)
            logger.info("Leverage set to %dx", self._config.leverage)
        except Exception as e:
            logger.warning("Could not set leverage (may already be set): %s", e)

    def fetch_ohlcv(self, since_ms: Optional[int] = None, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV candles as DataFrame with DatetimeIndex UTC."""
        raw = self._retry(
            self._exchange.fetch_ohlcv,
            self._symbol,
            self._config.timeframe,
            since=since_ms,
            limit=limit,
        )
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype("float64")
        return df

    def fetch_ohlcv_history(self, num_bars: int) -> pd.DataFrame:
        """Fetch a large history of candles by paginating forward."""
        all_dfs = []
        fetched = 0
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms = now_ms - (num_bars * self._config.timeframe_minutes * 60 * 1000)

        while fetched < num_bars:
            batch_limit = min(self.MAX_CANDLES_PER_REQUEST, num_bars - fetched)
            df = self.fetch_ohlcv(since_ms=since_ms, limit=batch_limit)
            if df.empty:
                break
            all_dfs.append(df)
            fetched += len(df)
            last_ts_ms = int(df.index[-1].timestamp() * 1000)
            since_ms = last_ts_ms + (self._config.timeframe_minutes * 60 * 1000)
            if len(df) < batch_limit:
                break

        if not all_dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        return result

    def get_position(self) -> dict:
        """Get current position for the symbol."""
        positions = self._retry(self._exchange.fetch_positions, [self._symbol])

        for pos in positions:
            if pos["symbol"] == self._symbol:
                size = float(pos.get("contracts", 0) or 0)
                side_str = pos.get("side", "")
                if size > 0 and side_str:
                    return {
                        "side": side_str,
                        "size": size,
                        "notional": float(pos.get("notional", 0) or 0),
                        "entry_price": float(pos.get("entryPrice", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                    }

        return {
            "side": "flat",
            "size": 0.0,
            "notional": 0.0,
            "entry_price": None,
            "unrealized_pnl": 0.0,
        }

    def get_position_sign(self) -> int:
        """Get current position as {-1, 0, +1}."""
        pos = self.get_position()
        if pos["side"] == "long":
            return 1
        elif pos["side"] == "short":
            return -1
        return 0

    def fetch_best_bid_ask(self) -> dict:
        """Fetch the current best bid and ask prices.

        Returns dict with 'bid' and 'ask' floats.
        """
        ticker = self._retry(self._exchange.fetch_ticker, self._symbol)
        return {
            "bid": float(ticker.get("bid", 0) or 0),
            "ask": float(ticker.get("ask", 0) or 0),
        }

    def place_market_order(self, side: str, quantity: float) -> dict:
        """Place a market order. side: 'buy' or 'sell'."""
        logger.info("Placing %s market order: %.4f %s", side, quantity, self._symbol)
        order = self._retry(
            self._exchange.create_market_order,
            self._symbol,
            side,
            quantity,
        )
        order_id = order.get("id")
        # Bybit doesn't return fill details in the initial response;
        # fetch the order to get avg_price / status / cost.
        if order_id and not order.get("average"):
            time.sleep(0.3)
            try:
                filled = self._retry(
                    self._exchange.fetch_order, order_id, self._symbol,
                    params={"acknowledged": True},
                )
                order = filled
            except Exception as e:
                logger.debug("Could not fetch market order details: %s", e)
        logger.info(
            "Order filled: id=%s avg_price=%s",
            order.get("id"),
            order.get("average"),
        )
        return order

    def place_limit_order(self, side: str, quantity: float,
                          price: float, timeout_sec: float = 10.0) -> dict:
        """Place a limit order at the given price with a fill-or-cancel timeout.

        Posts at the best bid (for buys) or best ask (for sells) to capture
        maker fees. If not filled within *timeout_sec*, cancels the limit
        order and falls back to a market order to guarantee execution.

        Returns the filled order dict.
        """
        logger.info(
            "Placing %s limit order: %.4f %s @ %.2f",
            side, quantity, self._symbol, price,
        )
        order = self._retry(
            self._exchange.create_limit_order,
            self._symbol,
            side,
            quantity,
            price,
        )
        order_id = order.get("id")

        # Poll for fill
        _ack = {"acknowledged": True}
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            time.sleep(0.5)
            status = self._retry(
                self._exchange.fetch_order, order_id, self._symbol, params=_ack,
            )
            if status.get("status") in ("closed", "filled"):
                logger.info(
                    "Limit order filled: id=%s avg_price=%s",
                    order_id, status.get("average"),
                )
                return status
            if status.get("status") in ("canceled", "cancelled", "expired"):
                break

        # Not filled — cancel and fall back to market
        try:
            self._retry(self._exchange.cancel_order, order_id, self._symbol)
            logger.info("Limit order %s cancelled (timeout)", order_id)
        except Exception as e:
            logger.warning("Cancel failed (may already be filled): %s", e)

        # Always re-check fill status after cancel attempt — the order may
        # have filled between the last poll and the cancel call
        status = self._retry(
            self._exchange.fetch_order, order_id, self._symbol, params=_ack,
        )
        if status.get("status") in ("closed", "filled"):
            logger.info("Limit order %s was already filled, skipping market fallback", order_id)
            return status

        logger.info("Falling back to market order for %s %.4f %s", side, quantity, self._symbol)
        return self.place_market_order(side, quantity)

    def place_limit_order_persistent(
        self,
        side: str,
        quantity: float,
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
        stats = FillStats(original_price=price)
        t_start = time.time()
        _ack = {"acknowledged": True}

        for attempt in range(1, max_attempts + 1):
            stats.attempts = attempt
            stats.prices_tried.append(price)

            logger.info(
                "[Reprice %d/%d] %s limit %.4f %s @ %.2f",
                attempt, max_attempts, side, quantity, self._symbol, price,
            )

            try:
                order = self._retry(
                    self._exchange.create_limit_order,
                    self._symbol, side, quantity, price,
                )
            except ExchangeError as e:
                logger.error("Failed to place limit order: %s", e)
                stats.elapsed_sec = time.time() - t_start
                return None, stats

            order_id = order.get("id")

            # Poll for fill
            deadline = time.time() + timeout_per_attempt
            while time.time() < deadline:
                time.sleep(poll_interval)
                try:
                    status = self._retry(
                        self._exchange.fetch_order, order_id, self._symbol,
                        params=_ack,
                    )
                except ExchangeError:
                    continue

                if status.get("status") in ("closed", "filled"):
                    stats.filled = True
                    stats.fill_price = status.get("average")
                    stats.final_price = price
                    stats.elapsed_sec = time.time() - t_start
                    logger.info(
                        "[Reprice %d/%d] FILLED: id=%s avg=%.2f elapsed=%.1fs",
                        attempt, max_attempts, order_id,
                        stats.fill_price or 0, stats.elapsed_sec,
                    )
                    return status, stats

                if status.get("status") in ("canceled", "cancelled", "expired"):
                    break

            # Timeout — cancel the order
            try:
                self._retry(self._exchange.cancel_order, order_id, self._symbol)
                logger.info("[Reprice %d/%d] Cancelled (timeout)", attempt, max_attempts)
            except Exception as e:
                logger.warning("[Reprice %d/%d] Cancel failed: %s", attempt, max_attempts, e)

            # Race condition guard: check if filled during cancel
            try:
                status = self._retry(
                    self._exchange.fetch_order, order_id, self._symbol,
                    params=_ack,
                )
                if status.get("status") in ("closed", "filled"):
                    stats.filled = True
                    stats.fill_price = status.get("average")
                    stats.final_price = price
                    stats.elapsed_sec = time.time() - t_start
                    logger.info(
                        "[Reprice %d/%d] Filled during cancel: avg=%.2f",
                        attempt, max_attempts, stats.fill_price or 0,
                    )
                    return status, stats
            except ExchangeError:
                pass

            # Re-fetch bid/ask for next attempt
            if attempt < max_attempts:
                try:
                    bba = self.fetch_best_bid_ask()
                    new_price = bba["bid"] if side == "buy" else bba["ask"]
                    if new_price and new_price > 0:
                        price = new_price
                        logger.info("[Reprice %d/%d] New price: %.2f", attempt, max_attempts, price)
                    else:
                        logger.warning("[Reprice %d/%d] Bad bid/ask, retrying same price", attempt, max_attempts)
                except Exception as e:
                    logger.warning("[Reprice %d/%d] Bid/ask fetch failed, retrying same price: %s", attempt, max_attempts, e)

        # Exhausted all attempts
        stats.elapsed_sec = time.time() - t_start
        logger.warning(
            "Limit order NOT filled after %d attempts over %.1fs. Prices tried: %s",
            max_attempts, stats.elapsed_sec, [round(p, 2) for p in stats.prices_tried],
        )
        return None, stats

    def close_position(self) -> Optional[dict]:
        """Close the current position entirely. Returns None if already flat."""
        pos = self.get_position()
        if pos["side"] == "flat":
            logger.info("Already flat, nothing to close")
            return None
        close_side = "sell" if pos["side"] == "long" else "buy"
        return self.place_market_order(close_side, pos["size"])

    def get_balance(self) -> dict:
        """Get USDT balance: total, free, used."""
        balance = self._retry(self._exchange.fetch_balance)
        usdt = balance.get("USDT", {})
        return {
            "total": float(usdt.get("total", 0) or 0),
            "free": float(usdt.get("free", 0) or 0),
            "used": float(usdt.get("used", 0) or 0),
        }

    def _retry(self, func, *args, **kwargs):
        """Execute a ccxt call with exponential backoff retry."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RateLimitExceeded,
            ) as e:
                last_error = e
                wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Exchange error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
            except (
                ccxt.AuthenticationError,
                ccxt.InsufficientFunds,
                ccxt.InvalidOrder,
            ) as e:
                logger.error("Non-retryable exchange error: %s", e)
                raise ExchangeError(str(e)) from e

        raise ExchangeError(
            f"Failed after {self.MAX_RETRIES} retries: {last_error}"
        ) from last_error
