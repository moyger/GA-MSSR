"""Tests for BybitExchange persistent limit order re-pricing."""
from unittest.mock import MagicMock, patch
import pytest

from live.exchange import FillStats


class TestFillStats:
    def test_slippage_calculated(self):
        stats = FillStats(original_price=100.0, fill_price=100.05)
        assert stats.slippage == pytest.approx(0.05)

    def test_slippage_none_when_no_fill(self):
        stats = FillStats(original_price=100.0)
        assert stats.slippage is None

    def test_defaults(self):
        stats = FillStats()
        assert stats.filled is False
        assert stats.attempts == 0
        assert stats.prices_tried == []


class TestPlaceLimitOrderPersistent:
    """Test the re-pricing loop using a mocked ccxt exchange."""

    @pytest.fixture
    def exchange(self):
        """BybitExchange with mocked ccxt internals."""
        with patch("live.exchange.ccxt") as mock_ccxt:
            mock_ccxt.bybit.return_value = MagicMock()
            # Avoid real __init__ side effects by patching
            from live.exchange import BybitExchange
            config = MagicMock()
            config.api_key = "test"
            config.api_secret = "test"
            config.testnet = True
            config.symbol = "ETH/USDT:USDT"
            ex = BybitExchange(config)
            return ex

    def test_fills_on_first_attempt(self, exchange):
        """Order fills during the first poll cycle."""
        exchange._exchange.create_limit_order.return_value = {"id": "ord1"}
        exchange._exchange.fetch_order.return_value = {
            "id": "ord1", "status": "closed", "average": 3000.50,
        }

        order, stats = exchange.place_limit_order_persistent(
            "buy", 0.01, 3000.0,
            max_attempts=3, timeout_per_attempt=1.0, poll_interval=0.1,
        )

        assert order is not None
        assert order["status"] == "closed"
        assert stats.filled is True
        assert stats.attempts == 1
        assert stats.fill_price == 3000.50
        assert stats.original_price == 3000.0

    def test_fills_after_reprice(self, exchange):
        """First attempt times out, second attempt fills."""
        exchange._exchange.create_limit_order.return_value = {"id": "ord1"}
        exchange._exchange.cancel_order.return_value = {}
        exchange._exchange.fetch_ticker.return_value = {"bid": 3001.0, "ask": 3001.5}

        call_count = [0]

        def mock_fetch_order(order_id, symbol, params=None):
            call_count[0] += 1
            # First attempt: ~5 polls + 1 post-cancel check = ~6 calls, all "open"
            # Then re-price triggers attempt 2, which fills immediately
            if call_count[0] <= 7:
                return {"id": "ord1", "status": "open"}
            return {"id": "ord1", "status": "closed", "average": 3001.0}

        exchange._exchange.fetch_order.side_effect = mock_fetch_order

        order, stats = exchange.place_limit_order_persistent(
            "buy", 0.01, 3000.0,
            max_attempts=3, timeout_per_attempt=0.5, poll_interval=0.1,
        )

        assert order is not None
        assert stats.filled is True
        assert stats.attempts == 2
        assert len(stats.prices_tried) == 2

    def test_gives_up_after_max_attempts(self, exchange):
        """All attempts fail — returns None, no market fallback."""
        exchange._exchange.create_limit_order.return_value = {"id": "ord1"}
        exchange._exchange.fetch_order.return_value = {"id": "ord1", "status": "open"}
        exchange._exchange.cancel_order.return_value = {}
        exchange._exchange.fetch_ticker.return_value = {"bid": 3000.0, "ask": 3000.5}

        order, stats = exchange.place_limit_order_persistent(
            "buy", 0.01, 3000.0,
            max_attempts=2, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is None
        assert stats.filled is False
        assert stats.attempts == 2

    def test_race_condition_fill_during_cancel(self, exchange):
        """Order fills between last poll and cancel — detected by post-cancel check."""
        exchange._exchange.create_limit_order.return_value = {"id": "ord1"}
        exchange._exchange.cancel_order.side_effect = Exception("already filled")

        call_count = [0]

        def mock_fetch_order(order_id, symbol, params=None):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"id": "ord1", "status": "open"}
            return {"id": "ord1", "status": "filled", "average": 3000.25}

        exchange._exchange.fetch_order.side_effect = mock_fetch_order

        order, stats = exchange.place_limit_order_persistent(
            "buy", 0.01, 3000.0,
            max_attempts=3, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is not None
        assert stats.filled is True
        assert stats.fill_price == 3000.25

    def test_exchange_error_returns_none(self, exchange):
        """Non-retryable error on order creation returns None."""
        from live.exchange import ExchangeError

        exchange._exchange.create_limit_order.side_effect = ExchangeError("Insufficient funds")

        # Override _retry to propagate the error
        def failing_retry(func, *args, **kwargs):
            raise ExchangeError("Insufficient funds")
        exchange._retry = failing_retry

        order, stats = exchange.place_limit_order_persistent(
            "buy", 0.01, 3000.0,
            max_attempts=3, timeout_per_attempt=0.3,
        )

        assert order is None
        assert stats.filled is False
        assert stats.attempts == 1
