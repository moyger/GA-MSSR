"""
Main live trading bot for GA-MSSR on CME futures.

Data:      IBKR TWS/Gateway (OHLCV bars, read-only connection)
Execution: TradersPost webhooks (connected to FundedNext Tradovate account)

Lifecycle:
1. Connect to IBKR (data) + validate TradersPost webhook
2. Train model or load from disk
3. Warm up signal engine with recent candles
4. Reconcile current position (from internal state)
5. Schedule: on_bar every 15 min, retrain daily
6. Run until SIGTERM/SIGINT

Key differences from Bybit bot (live/bot.py):
- CME trading hours guard (skip during break/weekend)
- Weekend auto-flatten before Friday close
- FundedNext risk checks (trailing drawdown, consistency, daily limit)
- CME bars_per_day = 92 (23h trading, not 24/7)
- TradersPost webhook execution (no direct exchange API)
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd

from strategies.khushi_strategy import KhushiSignalEngine
from live_futures.config import LiveFuturesConfig
from live_futures.exchange import FuturesExchange, ExchangeError
from live_futures.trainer import FuturesTrainer, ModelState
from live_futures.risk import FuturesRiskManager
from live_futures.notifier import SlackNotifier
from live_futures.trading_hours import CMETradingHours
from monitoring.heartbeat import HeartbeatMonitor
from monitoring.pidlock import PidLock

logger = logging.getLogger("ga_mssr_futures")


class LiveFuturesBot:
    """GA-MSSR live futures trading bot orchestrator."""

    def __init__(self, config: LiveFuturesConfig):
        self._config = config
        self._exchange = FuturesExchange(config)
        self._trainer = FuturesTrainer(config)
        self._risk = FuturesRiskManager(config)
        self._slack = SlackNotifier(config.slack_webhook_url)
        self._trading_hours = CMETradingHours()
        self._engine: Optional[KhushiSignalEngine] = None
        self._model: Optional[ModelState] = None
        self._running = False
        self._last_signal: Optional[int] = None

        # Stop-loss tracking
        self._entry_price: Optional[float] = None
        self._stop_price: Optional[float] = None

        self._state_dir = Path(config.state_dir)
        self._model_path = self._state_dir / "model_state.json"
        self._trade_log_path = self._state_dir / "trade_log.jsonl"

        self._heartbeat = HeartbeatMonitor(
            heartbeat_path=self._state_dir / "heartbeat.json",
            bot_name="Futures (MNQ)",
            slack_notifier=self._slack,
            interval_sec=3600,
            write_interval_sec=60,
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Full startup sequence."""
        self._setup_logging()

        # Prevent duplicate instances
        self._pidlock = PidLock("/tmp/ga_mssr_futures.pid")
        self._pidlock.acquire()

        logger.info("=" * 60)
        logger.info("GA-MSSR Futures Bot starting")
        logger.info("  Contract:   %s", self._config.contract_root)
        logger.info("  Execution:  TradersPost webhook")
        logger.info("  Timeframe:  %s", self._config.timeframe)
        logger.info("  Trade size: %d contracts", self._config.trade_size)
        logger.info("  FN account: $%.0f", self._config.fn_account_size)
        logger.info("  FN daily:   $%.0f", self._config.fn_daily_loss_limit)
        logger.info("  FN max DD:  $%.0f", self._config.fn_trailing_max_loss)
        logger.info("  Stop-loss:  %.0f pts ($%.0f)",
                     self._config.stop_loss_points,
                     self._config.stop_loss_points * 2)
        logger.info("=" * 60)

        # Connect
        self._exchange.connect()

        # Train or load model
        self._initialize_model()

        # Warm up signal engine
        self._warmup()

        # Reconcile position (from internal state)
        current_pos = self._exchange.get_position_sign()
        logger.info("Current position (internal): %+d", current_pos)

        self._slack.notify_startup(
            symbol=self._config.contract_root,
            trade_size=self._config.trade_size,
            leverage=1,
            testnet=False,
            balance=self._config.fn_account_size,
            position=current_pos,
        )

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Start scheduler
        self._running = True
        self._start_scheduler()

    def _initialize_model(self) -> None:
        """Train a new model or load from disk."""
        if self._model_path.exists():
            try:
                self._model = ModelState.load(self._model_path)
                logger.info(
                    "Loaded saved model (trained %s, SSR=%.4f)",
                    self._model.training_timestamp,
                    self._model.best_fitness,
                )
                self._create_engine_from_model()
                return
            except Exception as e:
                logger.warning("Failed to load model: %s", e)

        logger.info("No saved model found. Training new model...")
        self._retrain()

    def _retrain(self) -> None:
        """Fetch training data and train a new model."""
        bars_needed = self._config.train_days * self._config.bars_per_day
        logger.info(
            "Fetching %d bars for training (%d days × %d bars/day)...",
            bars_needed, self._config.train_days, self._config.bars_per_day,
        )

        df = self._exchange.fetch_ohlcv_history(bars_needed)
        if len(df) < bars_needed * 0.8:
            logger.error(
                "Insufficient training data: got %d bars, need ~%d",
                len(df), bars_needed,
            )
            if self._model is not None:
                logger.warning("Keeping existing model")
                return
            raise RuntimeError("Cannot train: insufficient data")

        logger.info(
            "Fetched %d bars (%s to %s)",
            len(df), df.index[0], df.index[-1],
        )

        try:
            self._model = self._trainer.train(df)
            self._model.save(self._model_path)
            self._create_engine_from_model()
            logger.info("Model trained and saved successfully")
        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            if self._engine is None:
                raise

    def _create_engine_from_model(self) -> None:
        """Create a fresh KhushiSignalEngine from the current ModelState."""
        self._engine = KhushiSignalEngine(
            ga_weights=self._model.ga_weights,
            rule_params=self._model.rule_params,
            warmup_bars=self._config.warmup_bars,
            denoise=self._config.denoise,
            wavelet=self._config.wavelet,
            er_period=self._config.er_period,
            er_threshold=self._config.er_threshold,
            er_enter_threshold=self._config.er_enter_threshold,
            er_exit_threshold=self._config.er_exit_threshold,
            min_hold_bars=self._config.min_hold_bars,
        )

    def _warmup(self) -> None:
        """Fetch recent candles and push into signal engine to fill the buffer."""
        warmup_count = self._config.warmup_bars + 10
        logger.info("Warming up signal engine with %d bars...", warmup_count)

        df = self._exchange.fetch_ohlcv_history(warmup_count)
        logger.info("Fetched %d warmup bars", len(df))

        for ts, row in df.iterrows():
            self._engine.push_bar(
                timestamp=ts,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

        logger.info(
            "Signal engine: %d bars pushed, warmed_up=%s",
            self._engine.bar_count,
            self._engine.is_warmed_up,
        )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _start_scheduler(self) -> None:
        """Main trading loop — runs on the main thread for ib_insync compatibility.

        ib_insync requires all IB API calls on the same thread that called
        connect(), so we use a simple polling loop instead of APScheduler's
        ThreadPoolExecutor.
        """
        logger.info("Scheduler started. Waiting for next 15-min candle...")

        # Track last-fired timestamps to avoid double-firing
        last_bar_key = None  # (date, hour, quarter)
        last_retrain_date = None
        last_stop_check = 0.0  # monotonic time of last stop-loss price check

        bar_trigger_minutes = {1, 16, 31, 46}
        stop_check_interval = 30  # seconds between stop-loss price checks

        try:
            while self._running:
                now = datetime.now(timezone.utc)

                # ── 15-min candle handler (fire at :01, :16, :31, :46) ──
                if now.minute in bar_trigger_minutes:
                    bar_key = (now.date(), now.hour, now.minute // 15)
                    if bar_key != last_bar_key:
                        last_bar_key = bar_key
                        self._on_bar_safe()

                # ── Stop-loss check (every 30s when in a position) ──
                if (
                    self._stop_price is not None
                    and time.monotonic() - last_stop_check >= stop_check_interval
                ):
                    last_stop_check = time.monotonic()
                    self._check_stop_loss()

                # ── Daily retrain ──
                if (
                    now.hour == self._config.retrain_hour_utc
                    and now.minute >= 5
                    and now.date() != last_retrain_date
                ):
                    last_retrain_date = now.date()
                    self._retrain_safe()

                self._heartbeat.tick(
                    position=self._exchange.get_position_sign(),
                    daily_pnl=self._risk.daily_pnl,
                )

                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()

    # ------------------------------------------------------------------
    # Core trading logic
    # ------------------------------------------------------------------

    def _on_bar_safe(self) -> None:
        """Wrapper with exception handling."""
        try:
            self._on_bar()
        except ExchangeError as e:
            logger.error("Exchange error in on_bar: %s", e)
            self._slack.notify_error(f"Exchange error: {e}")
        except Exception as e:
            logger.error("Unexpected error in on_bar: %s", e, exc_info=True)
            self._slack.notify_error(f"Unexpected error: {e}")

    def _on_bar(self) -> None:
        """Called every 15 minutes after candle close."""
        # CME trading hours guard
        if not self._trading_hours.is_market_open():
            mins = self._trading_hours.minutes_to_next_open()
            if mins > 0:
                logger.debug("Market closed, next open in ~%d min", mins)
            return

        # Weekend flatten
        if self._trading_hours.should_flatten_for_weekend(
            minutes_before=self._config.flatten_before_weekend_min,
        ):
            logger.info("Approaching weekend close — flattening position")
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                self._exchange.close_position()
                self._slack.send(
                    "*GA-MSSR Futures | Weekend Flatten*\n"
                    "Closed position before Friday close"
                )
            return

        # Pass estimated balance for EOD high-water mark on day roll
        estimated_balance = self._config.fn_account_size + self._risk.daily_pnl
        self._risk.check_new_day(current_balance=estimated_balance)

        # Check trailing drawdown (FundedNext)
        if self._risk.check_trailing_drawdown(estimated_balance):
            self._slack.send(
                "*GA-MSSR Futures | TRAILING DD BREACHED*\n"
                f"Balance: ${estimated_balance:,.2f}\n"
                f"Floor: ${self._risk.trailing_floor:,.2f}\n"
                "Challenge FAILED. Halting trading."
            )
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                self._exchange.close_position()
            return

        # Fetch latest candles
        df = self._exchange.fetch_ohlcv(limit=3)
        if df.empty or len(df) < 2:
            logger.warning("Insufficient candles returned")
            return

        # Use the most recently closed candle
        now = pd.Timestamp.now(tz="UTC")
        latest = df.iloc[-1]
        ts = df.index[-1]
        expected_close = ts + pd.Timedelta(minutes=self._config.timeframe_minutes)
        if expected_close > now:
            latest = df.iloc[-2]
            ts = df.index[-2]

        # Push to signal engine
        self._engine.push_bar(
            timestamp=ts,
            open_=float(latest["open"]),
            high=float(latest["high"]),
            low=float(latest["low"]),
            close=float(latest["close"]),
            volume=float(latest["volume"]),
        )

        # Compute target position
        target = self._engine.compute_position(self._config.position_threshold)
        if target is None:
            logger.info(
                "Signal engine warming up (%d/%d bars)",
                self._engine.bar_count, self._config.warmup_bars,
            )
            return

        self._last_signal = target

        # --- Risk checks ---

        # Daily loss limit (FundedNext)
        if self._risk.check_daily_limit():
            self._slack.notify_daily_limit(
                self._risk.daily_pnl, self._config.fn_daily_loss_limit,
            )
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning("Daily limit hit — closing position")
                self._exchange.close_position()
            return

        # Consistency rule (FundedNext)
        if self._risk.check_consistency_rule():
            logger.info("Consistency cap reached — holding for the day")
            return

        # Get current position (internal tracking)
        current = self._exchange.get_position_sign()

        logger.info(
            "Bar %s | close=%.2f | target=%+d | current=%+d",
            ts, float(latest["close"]), target, current,
        )
        self._heartbeat.record_bar(str(ts))

        # Execute if position change needed
        if target != current:
            self._execute_position_change(
                current, target, bar_close=float(latest["close"]),
            )

    def _execute_position_change(
        self, current: int, target: int, bar_close: float = 0.0,
    ) -> None:
        """Adjust position from current to target via TradersPost."""
        qty = self._config.trade_size

        # Position size check
        if target != 0 and not self._risk.check_position_size(qty):
            logger.warning("Position size check failed, skipping")
            return

        # Use set_target_position for a single webhook call
        # TradersPost handles close + open for reversals
        order = self._exchange.set_target_position(target)
        if order:
            # Record PnL when closing a position (exit or reversal)
            if current != 0 and self._entry_price and bar_close > 0:
                self._record_pnl(current, bar_close)

            # Set stop-loss price for the new position
            if target != 0 and bar_close > 0:
                sl_pts = self._config.stop_loss_points
                if target > 0:  # long
                    self._stop_price = bar_close - sl_pts
                else:  # short
                    self._stop_price = bar_close + sl_pts
                self._entry_price = bar_close
                logger.info(
                    "Stop-loss set: entry=%.2f stop=%.2f (%.0f pts)",
                    bar_close, self._stop_price, sl_pts,
                )
            else:
                # Exiting — clear stop
                self._stop_price = None
                self._entry_price = None

            self._log_trade(current, target, order)

            self._slack.notify_trade(
                side="buy" if target > 0 else ("sell" if target < 0 else "exit"),
                qty=qty,
                symbol=self._config.contract_root,
                price=bar_close if bar_close > 0 else order.get("average"),
                current=current,
                target=target,
            )

    def _check_stop_loss(self) -> None:
        """Check current price against stop-loss level. Exits immediately if hit."""
        if self._stop_price is None:
            return

        try:
            price = self._exchange.fetch_quick_price()
        except Exception as e:
            logger.debug("Stop-loss price check failed: %s", e)
            return

        if price <= 0:
            return

        position = self._exchange.get_position_sign()
        triggered = False

        if position > 0 and price <= self._stop_price:  # long stop
            triggered = True
        elif position < 0 and price >= self._stop_price:  # short stop
            triggered = True

        if triggered:
            logger.warning(
                "STOP-LOSS HIT: price=%.2f stop=%.2f entry=%.2f | Closing position",
                price, self._stop_price, self._entry_price or 0.0,
            )
            self._exchange.close_position()

            # Record PnL from stop-loss
            if self._entry_price:
                self._record_pnl(position, price)

            loss_pts = abs(price - (self._entry_price or 0.0))
            self._slack.send(
                f"*GA-MSSR Futures | STOP-LOSS*\n"
                f"Price {price:.2f} hit stop {self._stop_price:.2f}\n"
                f"Loss: ~{loss_pts:.0f} pts (${loss_pts * 2:.0f})\n"
                f"Daily PnL: ${self._risk.daily_pnl:.2f}"
            )

            self._stop_price = None
            self._entry_price = None

    def _record_pnl(self, position: int, exit_price: float) -> None:
        """Estimate realized PnL and record it in the risk manager.

        MNQ: $2 per point per contract.
        """
        if not self._entry_price or exit_price <= 0:
            return

        pts = exit_price - self._entry_price
        if position < 0:  # short: profit when price drops
            pts = -pts

        dollar_per_point = 2.0  # MNQ multiplier
        pnl = pts * dollar_per_point * self._config.trade_size
        self._risk.record_trade(pnl)
        logger.info(
            "PnL recorded: %.2f pts × $%d × %d = $%.2f | Daily: $%.2f",
            pts, int(dollar_per_point), self._config.trade_size,
            pnl, self._risk.daily_pnl,
        )

    def _log_trade(self, current: int, target: int, order: dict) -> None:
        """Append trade to the trade log (JSONL)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_position": current,
            "to_position": target,
            "symbol": self._config.contract_root,
            "order_id": order.get("id"),
            "avg_price": order.get("average"),
            "status": order.get("status"),
        }
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._trade_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Retrain
    # ------------------------------------------------------------------

    def _retrain_safe(self) -> None:
        """Wrapper with exception handling."""
        try:
            logger.info("Starting scheduled retrain...")
            self._retrain()
            self._warmup()
            logger.info("Retrain complete. Signal engine re-warmed.")
            if self._model:
                self._slack.notify_retrain(
                    self._model.best_fitness,
                    str(self._model.training_timestamp),
                )
        except Exception as e:
            logger.error("Retrain failed: %s", e, exc_info=True)
            self._slack.notify_error(f"Retrain failed: {e}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_shutdown(self, signum, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down...", sig_name)
        self._shutdown()

    def _shutdown(self) -> None:
        """Graceful shutdown: stop loop, disconnect IBKR, log position."""
        self._running = False

        try:
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning(
                    "Shutting down with open %s position (size=%s). "
                    "Position will remain open on Tradovate.",
                    pos["side"], pos["size"],
                )
            self._slack.notify_shutdown(pos["side"], pos.get("size", 0.0))
        except Exception as e:
            logger.error("Could not check position on shutdown: %s", e)
            self._slack.notify_shutdown("unknown", 0.0)

        try:
            self._exchange.disconnect()
        except Exception:
            pass

        logger.info("Futures bot shutdown complete")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        log_dir = Path(self._config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # Rotating file: 10 MB, keep 5
        fh = RotatingFileHandler(
            log_dir / "ga_mssr_futures.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(ch)
