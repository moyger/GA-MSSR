#!/usr/bin/env python3
"""
Live MNQ futures trading via NautilusTrader + Interactive Brokers.

Connects to TWS/IB Gateway, loads a pre-trained GA-MSSR model, and runs
the KhushiStrategy for automated trading on MNQ (Micro E-mini Nasdaq 100).

Prerequisites:
    - TWS or IB Gateway running on localhost (paper: 7497/4002, live: 7496/4001)
    - CME market data subscription active in IBKR account
    - API access enabled in TWS: Edit > Global Config > API > Enable Socket Clients
    - Trained model at live_futures/state/model_state.json (auto-trains if missing)

Usage:
    .venv/bin/python scripts/run_live_nautilus.py
"""
import logging
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nautilus_trader.adapters.interactive_brokers.common import IB, IBContract
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import (
    LiveDataEngineConfig,
    LoggingConfig,
    RoutingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from data.mnq_config import MNQ
from live_futures.config import load_futures_config
from live_futures.retrain_actor import RetrainActor, RetrainActorConfig
from live_futures.trainer import FuturesTrainer, ModelState
from strategies.khushi_strategy import KhushiStrategy, KhushiStrategyConfig

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_STATE_PATH = ROOT / "live_futures" / "state" / "model_state.json"
TRAINING_DATA_PATH = ROOT / "data" / "futures" / "NQ_1min_IB.csv"


def _load_or_train_model(cfg) -> ModelState:
    """Load saved model or train a fresh one from IB data."""
    if MODEL_STATE_PATH.exists():
        model = ModelState.load(MODEL_STATE_PATH)
        logger.info(
            "Loaded model from %s (trained %s, SSR=%.4f)",
            MODEL_STATE_PATH,
            model.training_timestamp,
            model.best_fitness,
        )
        return model

    # No saved model — train from historical data
    logger.info("No saved model found. Training from %s ...", TRAINING_DATA_PATH)
    if not TRAINING_DATA_PATH.exists():
        logger.error(
            "Training data not found at %s. "
            "Run: .venv/bin/python scripts/download_ib_data.py",
            TRAINING_DATA_PATH,
        )
        sys.exit(1)

    from data.pipeline import build_denoised_dataset

    df = build_denoised_dataset(
        str(TRAINING_DATA_PATH),
        denoise_columns="close",
        timeframe="15min",
    )
    # Use last N days for training
    bars_needed = cfg.bars_per_day * cfg.train_days
    if len(df) > bars_needed:
        df = df.iloc[-bars_needed:]

    trainer = FuturesTrainer(cfg)
    model = trainer.train(df)
    model.save(MODEL_STATE_PATH)
    return model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Load config ----
    cfg = load_futures_config()
    logger.info("Config loaded: IBKR %s:%d (client_id=%d, account=%s)",
                cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id,
                cfg.ibkr_account or "<auto>")

    # ---- Load or train model ----
    model = _load_or_train_model(cfg)
    logger.info("Model: %d weights, %d rule params, SSR=%.4f",
                len(model.ga_weights), len(model.rule_params), model.best_fitness)

    # ---- IB instrument provider ----
    instrument_provider_config = InteractiveBrokersInstrumentProviderConfig(
        load_contracts=frozenset([
            IBContract(
                secType="CONTFUT",
                exchange="CME",
                symbol="MNQ",
            ),
        ]),
    )

    # ---- Data client ----
    data_client_config = InteractiveBrokersDataClientConfig(
        ibg_host=cfg.ibkr_host,
        ibg_port=cfg.ibkr_port,
        ibg_client_id=cfg.ibkr_client_id,
        use_regular_trading_hours=True,
        instrument_provider=instrument_provider_config,
    )

    # ---- Execution client ----
    exec_kwargs = dict(
        ibg_host=cfg.ibkr_host,
        ibg_port=cfg.ibkr_port,
        ibg_client_id=cfg.ibkr_client_id,
        instrument_provider=instrument_provider_config,
        routing=RoutingConfig(default=True),
    )
    if cfg.ibkr_account:
        exec_kwargs["account_id"] = cfg.ibkr_account
    exec_client_config = InteractiveBrokersExecClientConfig(**exec_kwargs)

    # ---- Trading node ----
    config_node = TradingNodeConfig(
        trader_id="GAMSSR-001",
        logging=LoggingConfig(log_level="INFO"),
        data_clients={IB: data_client_config},
        exec_clients={IB: exec_client_config},
        data_engine=LiveDataEngineConfig(
            time_bars_timestamp_on_close=False,
            validate_data_sequence=True,
        ),
    )

    node = TradingNode(config=config_node)
    node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
    node.build()

    # ---- Strategy ----
    # Instrument ID follows IB adapter's default symbology.
    # The strategy's on_start() resolves the instrument from cache.
    instrument_id = InstrumentId.from_str("MNQ.CME")
    bar_type = BarType.from_str("MNQ.CME-15-MINUTE-LAST-EXTERNAL")

    strategy_config = KhushiStrategyConfig(
        instrument_id=instrument_id,
        bar_type=bar_type,
        trade_size=Decimal(str(cfg.trade_size)),
        ga_weights=model.ga_weights,
        rule_params=model.rule_params,
        warmup_bars=cfg.warmup_bars,
        position_threshold=cfg.position_threshold,
        denoise=cfg.denoise,
        wavelet=cfg.wavelet,
        hard_stop_daily=cfg.max_daily_loss_usd,
    )
    strategy = KhushiStrategy(config=strategy_config)
    node.trader.add_strategy(strategy)

    # ---- Retrain Actor (daily model update) ----
    retrain_config = RetrainActorConfig(
        bar_type=str(bar_type),
        retrain_hour_utc=cfg.retrain_hour_utc,
        train_bars=cfg.bars_per_day * cfg.train_days,
        model_save_path=str(MODEL_STATE_PATH),
    )
    retrain_actor = RetrainActor(
        config=retrain_config,
        strategy=strategy,
        futures_config=cfg,
    )
    node.trader.add_actor(retrain_actor)

    logger.info("Starting NautilusTrader node (MNQ @ 15min bars)...")
    logger.info("  Instrument: %s", instrument_id)
    logger.info("  Bar type:   %s", bar_type)
    logger.info("  Trade size: %s contracts", cfg.trade_size)
    logger.info("  Threshold:  %.2f", cfg.position_threshold)
    logger.info("  Hard stop:  $%.0f daily loss", cfg.max_daily_loss_usd)
    logger.info("  Retrain:    daily (every 24h)")

    # ---- Run (blocks until shutdown) ----
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
