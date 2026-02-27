"""
Live forex trading bot for GA-MSSR on IBKR.

Data:      IBKR TWS/Gateway (15-min bars via ib_insync)
Execution: IBKR market orders on IDEALPRO
Risk:      FTMO-style rules ($500 daily limit, 10% max total loss)

Lifecycle:
1. Connect to TWS/Gateway
2. Train model or load from disk
3. Warm up signal engine with recent bars
4. Reconcile current position from IBKR
5. Main loop: on_bar every 15 min, retrain daily
6. Run until SIGTERM/SIGINT
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
from live_forex.config import LiveForexConfig
from live_forex.exchange import IBKRForexExchange, ExchangeError, FillStats
from live_forex.risk import ForexRiskManager
from live_forex.trading_hours import ForexTradingHours
from live_futures.trainer import FuturesTrainer, ModelState
from live.notifier import SlackNotifier
from monitoring.heartbeat import HeartbeatMonitor
from monitoring.pidlock import PidLock

logger = logging.getLogger("ga_mssr_forex")

# Bar schedule: fire at minutes 1, 16, 31, 46 (1 min after candle close)
BAR_MINUTES = {1, 16, 31, 46}


class LiveForexBot:
    """GA-MSSR live forex trading bot on IBKR."""

    def __init__(self, config: LiveForexConfig):
        self._config = config
        self._exchange = IBKRForexExchange(config)
        self._trainer = FuturesTrainer(config)
        self._risk = ForexRiskManager(config)
        self._slack = SlackNotifier(config.slack_webhook_url)
        self._trading_hours = ForexTradingHours()
        self._engine: Optional[KhushiSignalEngine] = None
        self._model: Optional[ModelState] = None
        self._running = False
        self._last_signal: Optional[int] = None
        self._last_bar_minute: Optional[int] = None
        self._last_retrain_date: Optional[str] = None

        self._state_dir = Path(config.state_dir)
        self._model_path = self._state_dir / "model_state.json"
        self._trade_log_path = self._state_dir / "trade_log.jsonl"

        self._heartbeat = HeartbeatMonitor(
            heartbeat_path=self._state_dir / "heartbeat.json",
            bot_name="Forex (AUDUSD)",
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
        self._pidlock = PidLock("/tmp/ga_mssr_forex.pid")
        self._pidlock.acquire()

        pair = self._config.forex_pair
        units = self._config.order_units
        mini_lots = units // 10_000
        std_lots = units // 100_000

        logger.info("=" * 60)
        logger.info("GA-MSSR Forex Bot starting")
        logger.info("  Pair:       %s", pair)
        logger.info("  Execution:  IBKR (ib_insync)")
        logger.info("  Timeframe:  %s", self._config.timeframe)
        logger.info("  Size:       %d units (%d mini / %d std lots)",
                     units, mini_lots, std_lots)
        logger.info("  Threshold:  %.2f", self._config.position_threshold)
        logger.info("  FTMO acct:  $%.0f", self._config.ftmo_account_size)
        logger.info("  Daily stop: $%.0f", self._config.ftmo_daily_loss_limit)
        logger.info("  Max loss:   $%.0f", self._config.ftmo_max_total_loss)
        logger.info("  Target:     $%.0f", self._config.ftmo_profit_target)
        logger.info("=" * 60)

        # Connect to IBKR
        self._exchange.connect()

        # Train or load model
        self._initialize_model()

        # Warm up signal engine
        self._warmup()

        # Reconcile position
        current_pos = self._exchange.get_position_sign()
        logger.info("Current position: %+d", current_pos)

        self._slack.notify_startup(
            symbol=pair,
            trade_size=mini_lots,
            leverage=1,
            testnet=True,  # paper account
            balance=self._config.ftmo_account_size,
            position=current_pos,
        )

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Start main loop
        self._running = True
        self._run_loop()

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
        """Fetch training data from IBKR and train a new model."""
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

        logger.info("Fetched %d bars (%s to %s)", len(df), df.index[0], df.index[-1])

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
        )

    def _warmup(self) -> None:
        """Fetch recent bars and push into signal engine to fill the buffer."""
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
    # Main loop (runs on ib_insync event loop)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main loop using ib_insync's event loop.

        Polls every 10 seconds. Fires on_bar at minutes 1, 16, 31, 46
        and retrain at the configured hour.
        """
        logger.info("Main loop started. Waiting for next 15-min candle...")

        while self._running:
            try:
                # ib_insync sleep keeps the event loop alive
                self._exchange._ib.sleep(10)

                now = datetime.now(timezone.utc)
                minute = now.minute

                # Check for bar trigger (fire once per scheduled minute)
                if minute in BAR_MINUTES and self._last_bar_minute != minute:
                    self._last_bar_minute = minute
                    self._on_bar_safe()

                # Reset bar minute tracker when we leave the trigger window
                if minute not in BAR_MINUTES:
                    self._last_bar_minute = None

                # Check for daily retrain
                today = now.strftime("%Y-%m-%d")
                if (now.hour == self._config.retrain_hour_utc and
                        now.minute >= 5 and now.minute < 15 and
                        self._last_retrain_date != today):
                    self._last_retrain_date = today
                    self._retrain_safe()

                # Heartbeat (internally throttled)
                try:
                    self._heartbeat.tick(
                        position=self._last_signal or 0,
                        daily_pnl=self._risk.daily_pnl,
                    )
                except Exception:
                    pass

            except (KeyboardInterrupt, SystemExit):
                self._shutdown()
                break
            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)
                time.sleep(5)

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
        # Forex trading hours guard
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
                    f"*GA-MSSR Forex | Weekend Flatten*\n"
                    f"Closed {self._config.forex_pair} position before Friday close"
                )
            return

        self._risk.check_new_day()

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

        # Daily loss limit
        if self._risk.check_daily_limit():
            self._slack.notify_daily_limit(
                self._risk.daily_pnl, self._config.ftmo_daily_loss_limit,
            )
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning("Daily limit hit — closing position")
                self._exchange.close_position()
            return

        # Get current position
        current = self._exchange.get_position_sign()

        logger.info(
            "Bar %s | close=%.5f | target=%+d | current=%+d",
            ts, float(latest["close"]), target, current,
        )
        self._heartbeat.record_bar(str(ts))

        # Execute if position change needed
        if target != current:
            self._execute_position_change(current, target)

    def _execute_position_change(self, current: int, target: int) -> None:
        """Adjust position from current to target."""
        units = self._config.order_units

        # Close existing position first (market order — speed for exits)
        if current != 0:
            logger.info("Closing current position (%+d)", current)
            self._exchange.close_position()
            self._exchange._ib.sleep(0.5)

        # Open new position with persistent limit orders
        if target != 0:
            side = "BUY" if target > 0 else "SELL"

            # Fetch bid/ask for limit price
            try:
                bba = self._exchange.fetch_best_bid_ask()
                price = bba["bid"] if side == "BUY" else bba["ask"]
                if not price or price <= 0:
                    logger.warning("Invalid bid/ask price, skipping entry")
                    return
            except Exception as e:
                logger.warning("Could not fetch bid/ask, skipping entry: %s", e)
                return

            # Persistent limit order — no market fallback
            order, stats = self._exchange.place_limit_order_persistent(
                side=side,
                units=units,
                price=price,
                max_attempts=self._config.reprice_max_attempts,
                timeout_per_attempt=self._config.reprice_timeout_sec,
                poll_interval=self._config.reprice_poll_interval_sec,
            )

            self._log_fill_stats(side, stats)

            if order is None:
                logger.warning(
                    "Entry %s limit not filled after %d attempts (%.1fs). "
                    "Skipping, will re-evaluate next bar.",
                    side, stats.attempts, stats.elapsed_sec,
                )
                self._slack.send(
                    f"*GA-MSSR Forex | Entry not filled*\n"
                    f"{side} limit not filled after {stats.attempts} attempts "
                    f"({stats.elapsed_sec:.1f}s)"
                )
                return

            self._log_trade(current, target, order)

            mini_lots = units // 10_000
            self._slack.notify_trade(
                side=side.lower(),
                qty=mini_lots,
                symbol=self._config.forex_pair,
                price=order.get("average"),
                current=current,
                target=target,
                balance=self._config.ftmo_account_size,
            )

    def _log_fill_stats(self, side: str, stats: FillStats) -> None:
        """Log fill statistics for analysis."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "fill_stats",
            "side": side,
            "filled": stats.filled,
            "attempts": stats.attempts,
            "original_price": stats.original_price,
            "final_price": stats.final_price,
            "fill_price": stats.fill_price,
            "slippage": stats.slippage,
            "elapsed_sec": round(stats.elapsed_sec, 2),
            "prices_tried": stats.prices_tried,
        }
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._trade_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_trade(self, current: int, target: int, order: dict) -> None:
        """Append trade to the trade log (JSONL)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": self._config.forex_pair,
            "from_position": current,
            "to_position": target,
            "units": self._config.order_units,
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
        """Graceful shutdown: disconnect IBKR."""
        self._running = False

        try:
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning(
                    "Shutting down with open %s position (%s units). "
                    "Position remains on IBKR.",
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

        logger.info("Forex bot shutdown complete")
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
            log_dir / "ga_mssr_forex.log",
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
