"""
NautilusTrader Actor for daily GA-MSSR model retraining.

Runs alongside KhushiStrategy on the same TradingNode. Fires a daily timer,
requests historical bars from IB, trains a fresh model via FuturesTrainer,
and hot-swaps the strategy's signal engine weights without restart.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from live_futures.config import LiveFuturesConfig
from live_futures.trainer import FuturesTrainer, ModelState

if TYPE_CHECKING:
    from strategies.khushi_strategy import KhushiStrategy

logger = logging.getLogger(__name__)


class RetrainActorConfig(ActorConfig, frozen=True):
    """Configuration for the retrain actor."""

    bar_type: str = ""
    retrain_hour_utc: int = 0
    train_bars: int = 1840  # 20 days × 92 bars/day
    model_save_path: str = "live_futures/state/model_state.json"


class RetrainActor(Actor):
    """
    Daily model retraining actor.

    Requests historical bars from IB, trains a new GA-MSSR model via
    FuturesTrainer, saves the model to disk, and hot-swaps the strategy's
    signal engine weights — no restart, no buffer loss.
    """

    def __init__(
        self,
        config: RetrainActorConfig,
        strategy: KhushiStrategy,
        futures_config: LiveFuturesConfig,
    ) -> None:
        super().__init__(config)
        self._strategy = strategy
        self._trainer = FuturesTrainer(futures_config)
        self._bar_type = BarType.from_str(config.bar_type)
        self._train_bars = config.train_bars
        self._model_save_path = Path(config.model_save_path)

    def on_start(self) -> None:
        """Set up the daily retrain timer."""
        self.clock.set_timer(
            name="retrain_daily",
            interval=timedelta(hours=24),
            callback=self._on_retrain_timer,
        )
        self.log.info(
            f"RetrainActor started: daily timer set, "
            f"train_bars={self._train_bars}, bar_type={self._bar_type}"
        )

    def _on_retrain_timer(self, event) -> None:
        """Timer callback: request historical bars from IB for training."""
        self.log.info("Retrain timer fired — requesting historical bars...")
        days_back = (self._train_bars // 92) + 2  # extra margin
        start = datetime.now(timezone.utc) - timedelta(days=days_back)

        self.request_bars(
            bar_type=self._bar_type,
            start=start,
            callback=self._on_bars_ready,
        )

    def _on_bars_ready(self, request_id) -> None:
        """Callback after historical bars are cached. Train and hot-swap."""
        bars = self.cache.bars(self._bar_type)
        if bars is None or len(bars) < 100:
            self.log.warning(
                f"Insufficient bars for retraining: got {len(bars) if bars else 0}, need 100+"
            )
            return

        self.log.info(f"Received {len(bars)} bars, starting training...")

        # Convert NautilusTrader Bar objects to DataFrame
        df = self._bars_to_dataframe(bars)

        # Trim to training window
        if len(df) > self._train_bars:
            df = df.iloc[-self._train_bars:]

        try:
            model = self._trainer.train(df)
            model.save(self._model_save_path)
            self._strategy.update_from_model(model.ga_weights, model.rule_params)
            self.log.info(
                f"Retrain complete: SSR={model.best_fitness:.4f}, "
                f"return={model.total_return:.4f}, saved to {self._model_save_path}"
            )
        except Exception as e:
            self.log.error(f"Retrain failed: {e}")

    def _bars_to_dataframe(self, bars: list[Bar]) -> pd.DataFrame:
        """Convert a list of NautilusTrader Bar objects to an OHLCV DataFrame."""
        records = []
        for bar in bars:
            records.append({
                "timestamp": pd.Timestamp(bar.ts_event, unit="ns", tz="UTC"),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })
        df = pd.DataFrame(records)
        df = df.set_index("timestamp").sort_index()
        return df

    def on_stop(self) -> None:
        """Cancel the retrain timer."""
        self.clock.cancel_timer("retrain_daily")
        self.log.info("RetrainActor stopped")
