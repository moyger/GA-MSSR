"""CME futures contract symbol resolution.

Maps contract root (e.g. "MNQ") + date → specific month symbol ("MNQH6"),
continuous symbol ("@MNQ") for data feeds, and ib_insync Future objects.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

# ib_insync requires an event loop at import time
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import Future

# CME quarterly expiration months
QUARTER_MONTHS = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Approximate roll dates: 2nd Friday of the expiration month.
# After this date, the front month moves to the next quarter.
_ROLL_DAY = 14  # conservative: roll after the 14th of expiry month


def get_front_month(root: str, dt: date | None = None) -> str:
    """Return the front-month contract symbol for a given date.

    Examples:
        get_front_month("MNQ", date(2026, 2, 26))  -> "MNQH6"
        get_front_month("MNQ", date(2026, 3, 20))  -> "MNQM6"
        get_front_month("NQ",  date(2026, 6, 1))   -> "NQU6"
    """
    if dt is None:
        dt = date.today()

    year = dt.year
    month = dt.month

    # Find the current or next quarterly month
    for q_month in sorted(QUARTER_MONTHS):
        if month < q_month or (month == q_month and dt.day <= _ROLL_DAY):
            code = QUARTER_MONTHS[q_month]
            return f"{root}{code}{year % 10}"

    # Past December roll → next year's March
    code = QUARTER_MONTHS[3]
    return f"{root}{code}{(year + 1) % 10}"


def get_continuous_symbol(root: str) -> str:
    """Return the continuous contract symbol for data feeds.

    Example: get_continuous_symbol("MNQ") -> "@MNQ"
    """
    return f"@{root}"


def get_polygon_symbol(root: str, dt: date | None = None) -> str:
    """Return the Polygon.io ticker for the front-month contract.

    Polygon uses the format: MNQH2026 (full year, no spaces).

    Example: get_polygon_symbol("MNQ", date(2026, 2, 26)) -> "MNQH2026"
    """
    if dt is None:
        dt = date.today()

    year = dt.year
    month = dt.month

    for q_month in sorted(QUARTER_MONTHS):
        if month < q_month or (month == q_month and dt.day <= _ROLL_DAY):
            code = QUARTER_MONTHS[q_month]
            return f"{root}{code}{year}"

    code = QUARTER_MONTHS[3]
    return f"{root}{code}{year + 1}"


# Approximate expiry day (3rd Friday ≈ 20th). IB qualifyContracts() resolves exact date.
_QUARTER_EXPIRY_DAY = 20


def get_ib_contract(root: str, dt: date | None = None) -> Future:
    """Return an ib_insync Future for the front-month contract.

    The expiry date is approximate; caller must call ``ib.qualifyContracts()``
    to resolve the exact contract.

    Examples:
        get_ib_contract("MNQ", date(2026, 2, 26))  -> Future('MNQ', '20260320', 'CME')
        get_ib_contract("NQ",  date(2026, 6, 20))   -> Future('NQ', '20260920', 'CME')
    """
    if dt is None:
        dt = date.today()

    year = dt.year
    month = dt.month

    for q_month in sorted(QUARTER_MONTHS):
        if month < q_month or (month == q_month and dt.day <= _ROLL_DAY):
            expiry = f"{year}{q_month:02d}{_QUARTER_EXPIRY_DAY}"
            return Future(root, expiry, exchange="CME")

    # Past December roll → next year's March
    expiry = f"{year + 1}03{_QUARTER_EXPIRY_DAY}"
    return Future(root, expiry, exchange="CME")


def next_roll_date(root: str, dt: date | None = None) -> date:
    """Return the next contract roll date after the given date."""
    if dt is None:
        dt = date.today()

    year = dt.year
    for q_month in sorted(QUARTER_MONTHS):
        roll = date(year, q_month, _ROLL_DAY)
        if roll > dt:
            return roll

    # Next year March
    return date(year + 1, 3, _ROLL_DAY)
