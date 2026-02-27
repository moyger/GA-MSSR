"""
Model training for the futures bot.

Reuses the same training pipeline as the Bybit bot (wavelet denoise →
rule param grid search → GA weight optimization) but configured for
CME futures (92 bars/day instead of 96).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.wavelet_filter import WaveletFilter
from strategies.khushi_rules import train_rule_params, get_rule_features
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig, GAMSSRResult
from live_futures.config import LiveFuturesConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Snapshot of a trained model."""

    ga_weights: list[float]
    rule_params: list
    training_timestamp: str
    training_bars: int
    best_fitness: float
    total_return: float
    max_drawdown: float

    def to_json(self) -> str:
        d = asdict(self)
        d["rule_params"] = [
            list(p) if isinstance(p, tuple) else p
            for p in d["rule_params"]
        ]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelState":
        return cls(**json.loads(s))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info("Model state saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelState":
        return cls.from_json(path.read_text())


class FuturesTrainer:
    """Trains the GA-MSSR model from futures OHLCV data."""

    def __init__(self, config: LiveFuturesConfig):
        self._config = config
        self._wavelet_filter = WaveletFilter(wavelet=config.wavelet)

    def train(self, df: pd.DataFrame) -> ModelState:
        """Full training pipeline on an OHLCV DataFrame."""
        logger.info("Starting model training on %d bars...", len(df))
        t0 = time.time()

        df_train = df.copy()
        if self._config.denoise and len(df_train) >= 2:
            df_train["close"] = self._wavelet_filter.denoise(
                df_train["close"].values
            )
            logger.info("  Wavelet denoising applied")

        rule_params = train_rule_params(
            df_train,
            periods=list(self._config.rule_param_periods),
        )
        logger.info("  Rule parameters trained")

        features = get_rule_features(df_train, rule_params)
        if len(features) < 100:
            raise ValueError(
                f"Too few feature rows after NaN drop: {len(features)} "
                f"(need at least 100)"
            )
        logger.info("  Feature matrix: %s", features.shape)

        ga_config = GAMSSRConfig(
            sol_per_pop=self._config.ga_sol_per_pop,
            num_parents_mating=self._config.ga_sol_per_pop // 2,
            num_generations=self._config.ga_num_generations,
            random_seed=self._config.ga_random_seed,
        )
        ga = GAMSSR(ga_config)
        result: GAMSSRResult = ga.fit(features)

        elapsed = time.time() - t0
        logger.info(
            "  Training complete in %.1fs | SSR=%.4f | Return=%.4f | MaxDD=%.4f",
            elapsed,
            result.best_fitness,
            result.ssr_result.total_return,
            result.ssr_result.max_drawdown,
        )

        serializable_params = []
        for p in rule_params:
            if isinstance(p, tuple):
                serializable_params.append(list(p))
            elif isinstance(p, (int, float)):
                serializable_params.append(p)
            else:
                serializable_params.append(p)

        return ModelState(
            ga_weights=result.best_weights.tolist(),
            rule_params=serializable_params,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            training_bars=len(df),
            best_fitness=float(result.best_fitness),
            total_return=float(result.ssr_result.total_return),
            max_drawdown=float(result.ssr_result.max_drawdown),
        )
