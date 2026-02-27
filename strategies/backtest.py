"""
Backtest helpers for the Khushi GA-MSSR trading strategy.

Provides:
- create_nq_instrument(): NQ E-mini futures instrument definition
- create_mnq_instrument(): MNQ Micro E-mini futures instrument definition
- run_backtest(): End-to-end backtest setup and execution
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, AssetClass, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler

from data.mnq_config import MNQ
from data.nq_config import NQ
from strategies.khushi_strategy import KhushiStrategy, KhushiStrategyConfig


def create_nq_instrument(
    venue_str: str = "SIM",
    activation_ns: int = 0,
    expiration_ns: int = 0,
) -> FuturesContract:
    """
    Create an NQ E-mini Nasdaq 100 futures instrument for backtesting.

    Uses contract specs from data/nq_config.py:
    - tick_size: 0.25, point_value: $20, commission: $2.40 RT

    Parameters
    ----------
    venue_str : str
        Venue identifier string (default: "SIM").
    activation_ns : int
        Contract activation timestamp in nanoseconds.
    expiration_ns : int
        Contract expiration timestamp in nanoseconds. Must be after last bar.

    Returns
    -------
    FuturesContract
    """
    return FuturesContract(
        instrument_id=InstrumentId.from_str(f"NQ.{venue_str}"),
        raw_symbol=Symbol("NQ"),
        asset_class=AssetClass.INDEX,
        currency=Currency.from_str("USD"),
        price_precision=2,
        price_increment=Price.from_str(str(NQ.tick_size)),
        multiplier=Quantity.from_str(str(int(NQ.point_value))),
        lot_size=Quantity.from_str("1"),
        underlying="NQ",
        activation_ns=activation_ns,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.05"),
        maker_fee=Decimal("0.0"),
        taker_fee=Decimal("0.0"),
    )


def create_mnq_instrument(
    venue_str: str = "SIM",
    activation_ns: int = 0,
    expiration_ns: int = 0,
) -> FuturesContract:
    """
    Create an MNQ Micro E-mini Nasdaq 100 futures instrument.

    Uses contract specs from data/mnq_config.py:
    - tick_size: 0.25, point_value: $2, commission: $1.24 RT
    """
    return FuturesContract(
        instrument_id=InstrumentId.from_str(f"MNQ.{venue_str}"),
        raw_symbol=Symbol("MNQ"),
        asset_class=AssetClass.INDEX,
        currency=Currency.from_str("USD"),
        price_precision=2,
        price_increment=Price.from_str(str(MNQ.tick_size)),
        multiplier=Quantity.from_str(str(int(MNQ.point_value))),
        lot_size=Quantity.from_str("1"),
        underlying="MNQ",
        activation_ns=activation_ns,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.05"),
        maker_fee=Decimal("0.0"),
        taker_fee=Decimal("0.0"),
    )


def run_backtest(
    df: pd.DataFrame,
    strategy_config: KhushiStrategyConfig,
    starting_balance: float = 100_000,
    log_level: str = "ERROR",
) -> BacktestEngine:
    """
    Run a full backtest of the KhushiStrategy on OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        Must have a timestamp column or DatetimeIndex.
    strategy_config : KhushiStrategyConfig
        Strategy configuration with pre-trained GA weights and rule params.
    starting_balance : float
        Initial account balance in USD (default: 100,000).
    log_level : str
        Logging level: "ERROR", "WARNING", "INFO", "DEBUG" (default: "ERROR").

    Returns
    -------
    BacktestEngine
        The engine after run, with reports available via engine.trader.
        Caller is responsible for calling engine.dispose() when done.
    """
    venue_str = str(strategy_config.instrument_id.venue)
    venue = Venue(venue_str)

    # Engine config
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level=log_level),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    engine = BacktestEngine(config=engine_config)

    # Venue
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(starting_balance, USD)],
        bar_execution=True,
    )

    # Prepare bar data (before instrument, to derive activation/expiration)
    bar_df = _prepare_bar_dataframe(df)

    # Instrument — activation before first bar, expiration after last bar
    first_ts_ns = int(bar_df.index[0].value)
    last_ts_ns = int(bar_df.index[-1].value)
    activation_ns = first_ts_ns - 86_400_000_000_000  # 1 day before
    expiration_ns = last_ts_ns + 86_400_000_000_000   # 1 day after
    nq = create_nq_instrument(venue_str, activation_ns, expiration_ns)
    engine.add_instrument(nq)
    bar_type = BarType.from_str(str(strategy_config.bar_type))
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=nq)
    bars = wrangler.process(data=bar_df)
    engine.add_data(bars)

    # Strategy
    strategy = KhushiStrategy(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    # Run
    engine.run()

    return engine


def _prepare_bar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an OHLCV DataFrame into the format expected by BarDataWrangler.

    Requires:
    - DatetimeIndex (UTC) named 'timestamp' or existing DatetimeIndex
    - Columns: open, high, low, close, volume (float64)
    """
    result = df.copy()

    # Ensure we have a DatetimeIndex
    if not isinstance(result.index, pd.DatetimeIndex):
        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
            result = result.set_index("timestamp")
        elif result.index.name == "timestamp" or result.index.name is not None:
            # Mixed-tz index from IB data becomes object dtype; convert to UTC
            result.index = pd.to_datetime(result.index, utc=True)
        else:
            raise ValueError(
                "DataFrame must have a 'timestamp' column or DatetimeIndex"
            )

    # Ensure UTC
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    elif str(result.index.tz) != "UTC":
        result.index = result.index.tz_convert("UTC")

    # Select and convert columns
    cols = ["open", "high", "low", "close"]
    if "volume" in result.columns:
        cols.append("volume")
    result = result[cols].astype("float64")
    result = result.sort_index()

    # Enforce OHLC consistency: NautilusTrader requires
    # high >= max(open, close) and low <= min(open, close).
    # Wavelet denoising can violate this, so clamp here.
    oc_max = result[["open", "close"]].max(axis=1)
    oc_min = result[["open", "close"]].min(axis=1)
    result["high"] = result["high"].clip(lower=oc_max)
    result["low"] = result["low"].clip(upper=oc_min)

    return result
