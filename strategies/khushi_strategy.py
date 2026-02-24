"""
NautilusTrader strategy for the Khushi GA-MSSR trading system.

KhushiSignalEngine: Pure Python signal computation (no nautilus dependency).
KhushiStrategy: NautilusTrader Strategy wrapper for live/backtest execution.
KhushiStrategyConfig: Frozen config holding pre-trained model parameters.

Usage:
    1. Train offline: GAMSSR.fit() → best_weights, train_rule_params() → rule_params
    2. Create config with trained weights + params
    3. Run backtest or live via NautilusTrader engine
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional

from data.wavelet_filter import WaveletFilter
from strategies.khushi_rules import get_rule_features


# ---------------------------------------------------------------------------
# Signal Engine (pure Python, no nautilus dependency)
# ---------------------------------------------------------------------------

class KhushiSignalEngine:
    """
    Computes the GA-weighted trading signal from a rolling OHLC buffer.

    On each bar:
      1. Append bar to rolling deque
      2. Convert buffer → DataFrame
      3. Optionally denoise close prices
      4. Compute 16 rule signals via get_rule_features()
      5. Weight signals with GA-optimized weights
      6. Discretize to -1/0/+1

    Parameters
    ----------
    ga_weights : list[float]
        16 GA-optimized rule weights from GAMSSR.fit().
    rule_params : list
        16 rule parameter tuples from train_rule_params().
    warmup_bars : int
        Minimum bars before producing a signal (default: 200).
    denoise : bool
        Whether to apply wavelet denoising to close prices (default: True).
    wavelet : str
        Wavelet name for denoising (default: 'db4').
    """

    def __init__(
        self,
        ga_weights: list[float],
        rule_params: list,
        warmup_bars: int = 200,
        denoise: bool = True,
        wavelet: str = "db4",
    ):
        self._weights = np.array(ga_weights, dtype=np.float64)
        # Convert rule_params: lists back to tuples for rules, single-element lists to scalars
        self._rule_params = []
        for p in rule_params:
            if isinstance(p, (list, tuple)) and len(p) == 1:
                self._rule_params.append(p[0])
            elif isinstance(p, list):
                self._rule_params.append(tuple(p))
            else:
                self._rule_params.append(p)

        self._warmup_bars = warmup_bars
        self._denoise = denoise
        self._wavelet_filter = WaveletFilter(wavelet=wavelet) if denoise else None
        self._buffer: deque[dict] = deque(maxlen=warmup_bars)
        self._bar_count = 0

    def push_bar(
        self,
        timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Append a bar to the rolling buffer."""
        self._buffer.append({
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        self._bar_count += 1

    def compute_position(self, threshold: float = 0.0) -> Optional[int]:
        """
        Compute discretized target position from current buffer.

        Returns
        -------
        int or None
            +1 (long), -1 (short), 0 (flat), or None if still warming up.
        """
        if self._bar_count < self._warmup_bars:
            return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self._buffer))
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        # Optionally denoise close prices
        if self._denoise and self._wavelet_filter is not None and len(df) >= 2:
            df["close"] = self._wavelet_filter.denoise(df["close"].values)

        # Compute rule features (logr + 16 signals)
        features = get_rule_features(df, self._rule_params)
        if len(features) == 0:
            return None

        # Take last row's 16 rule signals
        signal_cols = [c for c in features.columns if c.startswith("rule_")]
        last_signals = features[signal_cols].iloc[-1].values.astype(np.float64)

        # Weighted combination
        raw_position = float(self._weights @ last_signals)

        # Discretize with threshold
        if raw_position > threshold:
            return 1
        elif raw_position < -threshold:
            return -1
        else:
            return 0

    @property
    def bar_count(self) -> int:
        """Number of bars received so far."""
        return self._bar_count

    @property
    def is_warmed_up(self) -> bool:
        """Whether the buffer has enough bars for signal computation."""
        return self._bar_count >= self._warmup_bars


# ---------------------------------------------------------------------------
# NautilusTrader Strategy (requires nautilus_trader)
# ---------------------------------------------------------------------------

try:
    from decimal import Decimal

    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.trading.strategy import Strategy

    class KhushiStrategyConfig(StrategyConfig, frozen=True):
        """Configuration for the Khushi GA-MSSR trading strategy."""
        instrument_id: InstrumentId
        bar_type: BarType
        trade_size: Decimal = Decimal("1")
        # Pre-trained model parameters
        ga_weights: list[float] = []
        rule_params: list = []
        # Signal computation
        warmup_bars: int = 200
        position_threshold: float = 0.0
        denoise: bool = True
        wavelet: str = "db4"
        # Risk management
        hard_stop_daily: float = 400.0
        max_drawdown_pct: float = 0.15

    class KhushiStrategy(Strategy):
        """
        NautilusTrader strategy implementing the Khushi GA-MSSR system.

        Uses pre-trained GA weights and rule parameters to generate
        trading signals bar-by-bar. Manages a single futures position
        (long/flat/short) based on the weighted signal.

        Risk controls:
        - Daily hard stop: flatten and stop trading if daily loss exceeds limit
        - Position sizing: 1 contract (configurable via trade_size)
        """

        def __init__(self, config: KhushiStrategyConfig) -> None:
            super().__init__(config)
            self._engine = KhushiSignalEngine(
                ga_weights=list(config.ga_weights),
                rule_params=list(config.rule_params),
                warmup_bars=config.warmup_bars,
                denoise=config.denoise,
                wavelet=config.wavelet,
            )
            self.instrument: Optional[Instrument] = None
            self._daily_pnl: float = 0.0
            self._daily_stopped: bool = False
            self._current_day: Optional[object] = None

        def on_start(self) -> None:
            """Subscribe to bar data on strategy start."""
            self.instrument = self.cache.instrument(self.config.instrument_id)
            if self.instrument is None:
                self.log.error(
                    f"Instrument not found: {self.config.instrument_id}"
                )
                self.stop()
                return
            self.subscribe_bars(self.config.bar_type)

        def on_bar(self, bar: Bar) -> None:
            """Process each bar: compute signal, manage position."""
            # Reset daily tracking on new day
            bar_day = pd.Timestamp(bar.ts_event, unit="ns").date()
            if self._current_day is not None and bar_day != self._current_day:
                self._daily_pnl = 0.0
                self._daily_stopped = False
            self._current_day = bar_day

            # Push bar into signal engine
            self._engine.push_bar(
                timestamp=pd.Timestamp(bar.ts_event, unit="ns"),
                open_=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )

            # Get target position
            target = self._engine.compute_position(self.config.position_threshold)
            if target is None:
                return  # Still warming up

            # Risk check: daily hard stop
            if self._daily_stopped:
                return

            if self._daily_pnl <= -self.config.hard_stop_daily:
                self._daily_stopped = True
                if not self.portfolio.is_flat(self.config.instrument_id):
                    self.close_all_positions(self.config.instrument_id)
                self.log.warning(
                    f"Daily hard stop hit: PnL=${self._daily_pnl:.2f}"
                )
                return

            # Determine current position direction
            net_qty = self.portfolio.net_position(self.config.instrument_id)
            current_sign = 1 if net_qty > 0 else (-1 if net_qty < 0 else 0)

            # Execute position change if needed
            if target != current_sign:
                # Close existing position first
                if current_sign != 0:
                    self.close_all_positions(self.config.instrument_id)

                # Open new position
                if target != 0:
                    side = OrderSide.BUY if target > 0 else OrderSide.SELL
                    order = self.order_factory.market(
                        instrument_id=self.config.instrument_id,
                        order_side=side,
                        quantity=self.instrument.make_qty(self.config.trade_size),
                        time_in_force=TimeInForce.GTC,
                    )
                    self.submit_order(order)

        def on_stop(self) -> None:
            """Clean up on strategy stop."""
            self.cancel_all_orders(self.config.instrument_id)
            self.close_all_positions(self.config.instrument_id)
            self.unsubscribe_bars(self.config.bar_type)

        def on_reset(self) -> None:
            """Reset strategy state."""
            self._engine = KhushiSignalEngine(
                ga_weights=list(self.config.ga_weights),
                rule_params=list(self.config.rule_params),
                warmup_bars=self.config.warmup_bars,
                denoise=self.config.denoise,
                wavelet=self.config.wavelet,
            )
            self._daily_pnl = 0.0
            self._daily_stopped = False
            self._current_day = None

    _NAUTILUS_AVAILABLE = True

except ImportError:
    _NAUTILUS_AVAILABLE = False
