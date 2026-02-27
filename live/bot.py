"""
Main live trading bot for GA-MSSR on Bybit.

Lifecycle:
1. Connect to exchange, verify balance
2. Train model or load from disk
3. Warm up signal engine with recent candles
4. Reconcile current exchange position
5. Schedule: on_bar every 15 min, retrain daily
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
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from strategies.khushi_strategy import KhushiSignalEngine
from live.config import LiveConfig
from live.exchange import BybitExchange, ExchangeError, FillStats
from live.trainer import Trainer, ModelState
from live.risk import RiskManager
from live.notifier import SlackNotifier
from monitoring.heartbeat import HeartbeatMonitor
from monitoring.pidlock import PidLock

logger = logging.getLogger("ga_mssr_bot")


class LiveBot:
    """GA-MSSR live trading bot orchestrator."""

    def __init__(self, config: LiveConfig):
        self._config = config
        self._exchange = BybitExchange(config)
        self._trainer = Trainer(config)
        self._risk = RiskManager(config)
        self._slack = SlackNotifier(config.slack_webhook_url)
        self._engine: Optional[KhushiSignalEngine] = None
        self._model: Optional[ModelState] = None
        self._scheduler: Optional[BlockingScheduler] = None
        self._running = False
        self._last_signal: Optional[int] = None
        self._entry_price: Optional[float] = None

        self._state_dir = Path(config.state_dir)
        self._model_path = self._state_dir / "model_state.json"
        self._trade_log_path = self._state_dir / "trade_log.jsonl"

        self._heartbeat = HeartbeatMonitor(
            heartbeat_path=self._state_dir / "heartbeat.json",
            bot_name="Bybit (ETH/USDT)",
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
        self._pidlock = PidLock("/tmp/ga_mssr_live.pid")
        self._pidlock.acquire()

        logger.info("=" * 60)
        logger.info("GA-MSSR Live Bot starting")
        logger.info("  Symbol:     %s", self._config.symbol)
        logger.info("  Testnet:    %s", self._config.testnet)
        logger.info("  Timeframe:  %s", self._config.timeframe)
        logger.info("  Trade size: %.4f", self._config.trade_size)
        logger.info("  Leverage:   %dx", self._config.leverage)
        logger.info("=" * 60)

        # Connect
        self._exchange.connect()
        balance = self._exchange.get_balance()
        logger.info(
            "Account balance: $%.2f (free: $%.2f)",
            balance["total"], balance["free"],
        )

        # Train or load model
        self._initialize_model()

        # Warm up signal engine
        self._warmup()

        # Reconcile position
        current_pos = self._exchange.get_position_sign()
        logger.info("Current exchange position: %+d", current_pos)

        self._slack.notify_startup(
            symbol=self._config.symbol,
            trade_size=self._config.trade_size,
            leverage=self._config.leverage,
            testnet=self._config.testnet,
            balance=balance["total"],
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
        """Fetch training data from exchange and train a new model."""
        bars_per_day = 96  # 15-min bars, 24h crypto
        bars_needed = self._config.train_days * bars_per_day
        logger.info(
            "Fetching %d bars for training (%d days)...",
            bars_needed, self._config.train_days,
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
        """Start APScheduler with on_bar (every 15 min) and retrain (daily)."""
        self._scheduler = BlockingScheduler(timezone="UTC")

        # Fire 1 minute after each 15-min candle close
        self._scheduler.add_job(
            self._on_bar_safe,
            CronTrigger(minute="1,16,31,46", timezone="UTC"),
            id="on_bar",
            name="Process 15-min candle",
            misfire_grace_time=300,
        )

        # Daily retrain
        self._scheduler.add_job(
            self._retrain_safe,
            CronTrigger(
                hour=self._config.retrain_hour_utc,
                minute=5,
                timezone="UTC",
            ),
            id="retrain",
            name="Daily model retrain",
            misfire_grace_time=3600,
        )

        # Heartbeat: write file every 60s, Slack every hour
        self._scheduler.add_job(
            self._heartbeat_tick,
            "interval",
            seconds=60,
            id="heartbeat",
            name="Heartbeat",
            misfire_grace_time=120,
        )

        logger.info("Scheduler started. Waiting for next 15-min candle...")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()

    def _heartbeat_tick(self) -> None:
        """Write heartbeat state.  Called by scheduler every 60s."""
        try:
            self._heartbeat.tick(
                position=self._last_signal or 0,
                daily_pnl=self._risk.daily_pnl,
            )
        except Exception:
            pass

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
        """
        Called every 15 minutes after candle close.

        1. Fetch latest closed candle
        2. Push to signal engine -> get target position
        3. Risk check
        4. Compare with exchange position, execute if different
        """
        self._risk.check_new_day()

        # Fetch latest candles (2 to ensure we have the closed one)
        df = self._exchange.fetch_ohlcv(limit=3)
        if df.empty or len(df) < 2:
            logger.warning("Insufficient candles returned from exchange")
            return

        # Use the second-to-last candle (most recently closed)
        # The last candle may still be forming
        now = pd.Timestamp.now(tz="UTC")
        latest = df.iloc[-1]
        ts = df.index[-1]
        expected_close = ts + pd.Timedelta(minutes=self._config.timeframe_minutes)
        if expected_close > now:
            # Last candle hasn't closed yet, use previous
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

        # Risk check
        if self._risk.check_daily_limit():
            self._slack.notify_daily_limit(
                self._risk.daily_pnl, self._config.max_daily_loss_usd,
            )
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning("Daily limit hit - closing position")
                self._exchange.close_position()
            return

        # Get current exchange position
        current = self._exchange.get_position_sign()

        logger.info(
            "Bar %s | close=%.2f | target=%+d | current=%+d",
            ts, float(latest["close"]), target, current,
        )
        self._heartbeat.record_bar(str(ts))

        # Execute if position change needed
        if target != current:
            self._execute_position_change(current, target)

    def _execute_position_change(self, current: int, target: int) -> None:
        """Adjust exchange position from current to target."""
        qty = self._config.trade_size

        # Close existing position if not flat
        if current != 0:
            logger.info("Closing current position (%+d)", current)
            # Record PnL before closing
            close_price = None
            try:
                bba = self._exchange.fetch_best_bid_ask()
                close_price = bba["bid"] if current > 0 else bba["ask"]
            except Exception:
                pass
            if close_price and self._entry_price:
                self._record_pnl(current, close_price)

            self._exchange.close_position()
            self._entry_price = None
            time.sleep(0.5)

        # Open new position if target is not flat
        if target != 0:
            side = "buy" if target > 0 else "sell"

            # Fetch best bid/ask for limit order and notional check
            try:
                bba = self._exchange.fetch_best_bid_ask()
                price = bba["bid"] if side == "buy" else bba["ask"]
                if not price or price <= 0:
                    logger.warning("Invalid bid/ask price, skipping entry")
                    return
                notional = qty * price
                if not self._risk.check_position_size(notional):
                    logger.warning("Position size check failed, skipping")
                    return
            except Exception as e:
                logger.warning("Could not fetch bid/ask, skipping entry: %s", e)
                return

            # Persistent limit order — no market fallback
            order, stats = self._exchange.place_limit_order_persistent(
                side=side,
                quantity=qty,
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
                self._slack.notify_error(
                    f"Entry {side} limit not filled after {stats.attempts} attempts "
                    f"({stats.elapsed_sec:.1f}s). Prices: "
                    f"{', '.join(f'${p:,.2f}' for p in stats.prices_tried)}"
                )
                return

            self._log_trade(side, qty, order)

            fill_price = order.get("average") or price
            self._entry_price = fill_price
            try:
                bal = self._exchange.get_balance().get("total")
            except Exception:
                bal = None
            self._slack.notify_trade(
                side=side, qty=qty, symbol=self._config.symbol,
                price=fill_price, current=current, target=target,
                balance=bal,
            )

    def _record_pnl(self, position: int, exit_price: float) -> None:
        """Estimate realized PnL and record it in the risk manager.

        PnL = (exit - entry) * qty for longs, (entry - exit) * qty for shorts.
        """
        if not self._entry_price or exit_price <= 0:
            return

        price_diff = exit_price - self._entry_price
        if position < 0:  # short: profit when price drops
            price_diff = -price_diff

        pnl = price_diff * self._config.trade_size
        self._risk.record_trade(pnl)
        logger.info(
            "PnL recorded: %.2f × %.4f = $%.2f | Daily: $%.2f",
            price_diff, self._config.trade_size,
            pnl, self._risk.daily_pnl,
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

    def _log_trade(self, side: str, qty: float, order: dict) -> None:
        """Append trade to the trade log (JSONL)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "quantity": qty,
            "symbol": self._config.symbol,
            "order_id": order.get("id"),
            "avg_price": order.get("average"),
            "status": order.get("status"),
            "cost": order.get("cost"),
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
        """Graceful shutdown: stop scheduler, log position, exit."""
        self._running = False

        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

        try:
            pos = self._exchange.get_position()
            if pos["side"] != "flat":
                logger.warning(
                    "Shutting down with open %s position (size=%.4f). "
                    "Position will remain open on the exchange.",
                    pos["side"], pos["size"],
                )
            self._slack.notify_shutdown(pos["side"], pos.get("size", 0.0))
        except Exception as e:
            logger.error("Could not check position on shutdown: %s", e)
            self._slack.notify_shutdown("unknown", 0.0)

        logger.info("Bot shutdown complete")
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
            log_dir / "ga_mssr_live.log",
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
