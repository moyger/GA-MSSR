"""Tests for IBKRForexExchange persistent limit order re-pricing."""
from unittest.mock import MagicMock, patch
import pytest

from live_forex.exchange import FillStats


class TestForexFillStats:
    def test_slippage_calculated(self):
        stats = FillStats(original_price=0.63250, fill_price=0.63255)
        assert stats.slippage == pytest.approx(0.00005)

    def test_slippage_none_when_no_fill(self):
        stats = FillStats(original_price=0.63250)
        assert stats.slippage is None


class TestIBKRPlaceLimitOrderPersistent:
    """Test the IBKR re-pricing loop with mocked ib_insync."""

    @pytest.fixture
    def exchange(self):
        """IBKRForexExchange with mocked IB connection."""
        with patch("live_forex.exchange.IB") as MockIB, \
             patch("live_forex.exchange.Forex") as MockForex:
            mock_ib = MagicMock()
            MockIB.return_value = mock_ib
            mock_ib.isConnected.return_value = True

            from live_forex.exchange import IBKRForexExchange
            config = MagicMock()
            config.forex_pair = "AUDUSD"
            config.order_units = 5000
            config.ibkr_host = "127.0.0.1"
            config.ibkr_port = 7497
            config.ibkr_client_id = 2
            config.ibkr_account = ""
            config.state_dir = "/tmp/test_forex_state"

            ex = IBKRForexExchange(config)
            ex._contract = MagicMock()
            ex._ib = mock_ib
            return ex

    def test_fills_on_first_attempt(self, exchange):
        """Limit order fills during first poll cycle."""
        mock_trade = MagicMock()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 0.63250
        mock_trade.order.orderId = 12345
        exchange._ib.placeOrder.return_value = mock_trade

        order, stats = exchange.place_limit_order_persistent(
            "BUY", 5000, 0.63250,
            max_attempts=3, timeout_per_attempt=1.0, poll_interval=0.1,
        )

        assert order is not None
        assert order["average"] == 0.63250
        assert stats.filled is True
        assert stats.attempts == 1

    def test_fills_after_reprice(self, exchange):
        """First attempt times out, second fills at new price."""
        call_count = [0]

        def mock_is_done():
            call_count[0] += 1
            # First attempt: not done, then done after cancel and new order
            return call_count[0] > 5

        mock_trade1 = MagicMock()
        mock_trade1.isDone.return_value = False
        mock_trade1.orderStatus.status = "PreSubmitted"
        mock_trade1.orderStatus.avgFillPrice = 0
        mock_trade1.order.orderId = 1

        mock_trade2 = MagicMock()
        mock_trade2.isDone.return_value = True
        mock_trade2.orderStatus.status = "Filled"
        mock_trade2.orderStatus.avgFillPrice = 0.63255
        mock_trade2.order.orderId = 2

        exchange._ib.placeOrder.side_effect = [mock_trade1, mock_trade2]

        # Bid/ask for re-pricing
        mock_ticker = MagicMock()
        mock_ticker.bid = 0.63255
        mock_ticker.ask = 0.63260
        exchange._ib.reqMktData.return_value = mock_ticker

        order, stats = exchange.place_limit_order_persistent(
            "BUY", 5000, 0.63250,
            max_attempts=3, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is not None
        assert stats.filled is True
        assert stats.attempts == 2
        assert len(stats.prices_tried) == 2

    def test_gives_up_after_max_attempts(self, exchange):
        """All attempts time out — returns None."""
        mock_trade = MagicMock()
        mock_trade.isDone.return_value = False
        mock_trade.orderStatus.status = "PreSubmitted"
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.order.orderId = 1
        exchange._ib.placeOrder.return_value = mock_trade

        mock_ticker = MagicMock()
        mock_ticker.bid = 0.63250
        mock_ticker.ask = 0.63255
        exchange._ib.reqMktData.return_value = mock_ticker

        order, stats = exchange.place_limit_order_persistent(
            "BUY", 5000, 0.63250,
            max_attempts=2, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is None
        assert stats.filled is False
        assert stats.attempts == 2

    def test_race_condition_fill_during_cancel(self, exchange):
        """Order fills after cancel is sent."""
        mock_trade = MagicMock()
        mock_trade.isDone.return_value = False
        mock_trade.orderStatus.status = "PreSubmitted"
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.order.orderId = 1

        exchange._ib.placeOrder.return_value = mock_trade

        # After cancel, status changes to Filled
        def cancel_side_effect(order):
            mock_trade.orderStatus.status = "Filled"
            mock_trade.orderStatus.avgFillPrice = 0.63248

        exchange._ib.cancelOrder.side_effect = cancel_side_effect

        order, stats = exchange.place_limit_order_persistent(
            "BUY", 5000, 0.63250,
            max_attempts=3, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is not None
        assert stats.filled is True
        assert stats.fill_price == 0.63248

    def test_rejected_order_moves_to_next_attempt(self, exchange):
        """IBKR rejects the order — tries next attempt with re-price."""
        mock_trade_rejected = MagicMock()
        mock_trade_rejected.isDone.return_value = True
        mock_trade_rejected.orderStatus.status = "Cancelled"
        mock_trade_rejected.orderStatus.avgFillPrice = 0
        mock_trade_rejected.order.orderId = 1
        mock_trade_rejected.log = [MagicMock(message="Insufficient margin")]

        mock_trade_filled = MagicMock()
        mock_trade_filled.isDone.return_value = True
        mock_trade_filled.orderStatus.status = "Filled"
        mock_trade_filled.orderStatus.avgFillPrice = 0.63260
        mock_trade_filled.order.orderId = 2

        exchange._ib.placeOrder.side_effect = [mock_trade_rejected, mock_trade_filled]

        mock_ticker = MagicMock()
        mock_ticker.bid = 0.63260
        mock_ticker.ask = 0.63265
        exchange._ib.reqMktData.return_value = mock_ticker

        order, stats = exchange.place_limit_order_persistent(
            "BUY", 5000, 0.63250,
            max_attempts=3, timeout_per_attempt=0.3, poll_interval=0.1,
        )

        assert order is not None
        assert stats.filled is True
        assert stats.attempts == 2
