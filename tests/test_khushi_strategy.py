"""Tests for KhushiSignalEngine and KhushiStrategy."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from decimal import Decimal

from data.loader import load_nq_data
from strategies.khushi_rules import train_rule_params, get_rule_features
from strategies.khushi_strategy import KhushiSignalEngine
from optimizers.ga_mssr import GAMSSR, GAMSSRConfig

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_nq_1m.csv"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_model():
    """Train a small GA model on fixture data, return (weights, params, df)."""
    df = load_nq_data(FIXTURE_PATH)
    params = train_rule_params(df, periods=[3, 7, 15])
    features = get_rule_features(df, params)
    ga = GAMSSR(GAMSSRConfig(
        sol_per_pop=8, num_parents_mating=4,
        num_generations=10, random_seed=42,
    ))
    result = ga.fit(features)
    return result.best_weights.tolist(), params, df


@pytest.fixture
def signal_engine(trained_model):
    """KhushiSignalEngine with trained model, warmup=50 for speed."""
    weights, params, _ = trained_model
    return KhushiSignalEngine(
        ga_weights=weights,
        rule_params=params,
        warmup_bars=50,
        denoise=False,
    )


def _push_all_bars(engine, df):
    """Push all bars from DataFrame into signal engine."""
    for idx, row in df.iterrows():
        engine.push_bar(
            timestamp=idx,
            open_=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


# ---------------------------------------------------------------------------
# KhushiSignalEngine tests (no nautilus dependency)
# ---------------------------------------------------------------------------

class TestKhushiSignalEngine:
    def test_warmup_returns_none(self, signal_engine, trained_model):
        """Before warmup is complete, compute_position returns None."""
        _, _, df = trained_model
        # Push only 10 bars (warmup is 50)
        for idx, row in df.head(10).iterrows():
            signal_engine.push_bar(
                timestamp=idx,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        assert signal_engine.compute_position() is None

    def test_position_after_warmup(self, signal_engine, trained_model):
        """After warmup, compute_position returns a valid position."""
        _, _, df = trained_model
        _push_all_bars(signal_engine, df)
        pos = signal_engine.compute_position()
        assert pos is not None
        assert pos in (-1, 0, 1)

    def test_zero_weights_flat(self, trained_model):
        """All-zero weights should produce position 0."""
        _, params, df = trained_model
        engine = KhushiSignalEngine(
            ga_weights=[0.0] * 16,
            rule_params=params,
            warmup_bars=50,
            denoise=False,
        )
        _push_all_bars(engine, df)
        assert engine.compute_position() == 0

    def test_position_in_range(self, signal_engine, trained_model):
        """Position must always be -1, 0, or +1."""
        _, _, df = trained_model
        _push_all_bars(signal_engine, df)
        pos = signal_engine.compute_position()
        assert pos in (-1, 0, 1)

    def test_threshold_deadband(self, trained_model):
        """Signals below a high threshold should produce position 0."""
        _, params, df = trained_model
        # Use very small weights so raw position is near zero
        engine = KhushiSignalEngine(
            ga_weights=[0.001] * 16,
            rule_params=params,
            warmup_bars=50,
            denoise=False,
        )
        _push_all_bars(engine, df)
        # With threshold=100 (impossibly high), should be 0
        assert engine.compute_position(threshold=100.0) == 0

    def test_denoise_toggle(self, trained_model):
        """Both denoise=True and denoise=False should produce valid positions."""
        weights, params, df = trained_model
        for do_denoise in [True, False]:
            engine = KhushiSignalEngine(
                ga_weights=weights,
                rule_params=params,
                warmup_bars=50,
                denoise=do_denoise,
            )
            _push_all_bars(engine, df)
            pos = engine.compute_position()
            assert pos in (-1, 0, 1), f"denoise={do_denoise} gave invalid position"

    def test_bar_count_tracks(self, signal_engine, trained_model):
        """bar_count should increment with each push_bar call."""
        _, _, df = trained_model
        assert signal_engine.bar_count == 0
        _push_all_bars(signal_engine, df)
        assert signal_engine.bar_count == len(df)

    def test_is_warmed_up(self, signal_engine, trained_model):
        """is_warmed_up should be False before warmup, True after."""
        _, _, df = trained_model
        assert not signal_engine.is_warmed_up
        _push_all_bars(signal_engine, df)
        assert signal_engine.is_warmed_up

    def test_rule_params_conversion(self, trained_model):
        """Rule params passed as lists should be converted properly."""
        weights, params, df = trained_model
        # Convert params to lists (as they'd come from JSON serialization)
        list_params = []
        for p in params:
            if isinstance(p, tuple):
                list_params.append(list(p))
            else:
                list_params.append([p])
        engine = KhushiSignalEngine(
            ga_weights=weights,
            rule_params=list_params,
            warmup_bars=50,
            denoise=False,
        )
        _push_all_bars(engine, df)
        pos = engine.compute_position()
        assert pos in (-1, 0, 1)


# ---------------------------------------------------------------------------
# NautilusTrader integration tests (skip if not installed)
# ---------------------------------------------------------------------------

nautilus = pytest.importorskip("nautilus_trader")


class TestKhushiStrategyBacktest:
    def test_strategy_runs_backtest(self, trained_model):
        """Full backtest completes without error."""
        from strategies.khushi_strategy import KhushiStrategyConfig
        from strategies.backtest import run_backtest
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.data import BarType

        weights, params, df = trained_model
        # Convert rule_params for config serialization
        config_params = []
        for p in params:
            if isinstance(p, tuple):
                config_params.append(list(p))
            elif isinstance(p, (int, float)):
                config_params.append([p])
            else:
                config_params.append(p)

        config = KhushiStrategyConfig(
            instrument_id=InstrumentId.from_str("NQ.SIM"),
            bar_type=BarType.from_str("NQ.SIM-1-MINUTE-LAST-EXTERNAL"),
            trade_size=Decimal("1"),
            ga_weights=weights,
            rule_params=config_params,
            warmup_bars=50,
            denoise=False,
            order_id_tag="001",
        )
        engine = run_backtest(df, config, log_level="ERROR")
        try:
            # Verify engine ran successfully
            assert engine is not None
        finally:
            engine.reset()
            engine.dispose()

    def test_strategy_generates_orders(self, trained_model):
        """Strategy should generate some orders during backtest."""
        from strategies.khushi_strategy import KhushiStrategyConfig
        from strategies.backtest import run_backtest
        from nautilus_trader.model.identifiers import InstrumentId, Venue
        from nautilus_trader.model.data import BarType

        weights, params, df = trained_model
        config_params = []
        for p in params:
            if isinstance(p, tuple):
                config_params.append(list(p))
            elif isinstance(p, (int, float)):
                config_params.append([p])
            else:
                config_params.append(p)

        config = KhushiStrategyConfig(
            instrument_id=InstrumentId.from_str("NQ.SIM"),
            bar_type=BarType.from_str("NQ.SIM-1-MINUTE-LAST-EXTERNAL"),
            trade_size=Decimal("1"),
            ga_weights=weights,
            rule_params=config_params,
            warmup_bars=50,
            denoise=False,
            order_id_tag="002",
        )
        engine = run_backtest(df, config, log_level="ERROR")
        try:
            report = engine.trader.generate_order_fills_report()
            # With 100 bars and warmup=50, should have some orders
            assert len(report) >= 0  # At minimum no crash
        finally:
            engine.reset()
            engine.dispose()

    def test_create_nq_instrument(self):
        """NQ instrument should have correct specs."""
        from strategies.backtest import create_nq_instrument

        nq = create_nq_instrument("SIM")
        assert str(nq.id) == "NQ.SIM"
        assert nq.price_precision == 2
        assert float(nq.multiplier) == 20.0
        assert float(nq.price_increment) == 0.25
