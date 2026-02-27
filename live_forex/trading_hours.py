"""Forex trading hours management.

Forex spot market (IDEALPRO) trades 24/5:
  - Opens:  Sunday  ~22:00 UTC (5:00 PM ET)
  - Closes: Friday  ~22:00 UTC (5:00 PM ET)
  - No daily maintenance break (unlike CME futures)

Note: exact open time varies slightly by broker; IBKR opens
forex at Sunday 17:15 ET. We use 22:00 UTC conservatively.
"""
from __future__ import annotations

from datetime import datetime, time, timezone


class ForexTradingHours:
    """Forex 24/5 session management."""

    # Weekly boundaries (UTC)
    FRIDAY_CLOSE_HOUR = 22   # Friday 22:00 UTC ≈ 5:00 PM ET
    SUNDAY_OPEN_HOUR = 22    # Sunday 22:00 UTC ≈ 5:00 PM ET

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if the forex market is currently open.

        Returns False during weekend:
          Friday 22:00 UTC → Sunday 22:00 UTC
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        weekday = dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = dt.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only after 22:00 UTC
        if weekday == 6:
            return hour >= self.SUNDAY_OPEN_HOUR

        # Friday: closed after 22:00 UTC
        if weekday == 4 and hour >= self.FRIDAY_CLOSE_HOUR:
            return False

        # Mon–Thu: always open (no daily break for forex)
        return True

    def should_flatten_for_weekend(
        self, dt: datetime | None = None, minutes_before: int = 30
    ) -> bool:
        """Check if we should flatten positions before the weekly close.

        Returns True if within `minutes_before` of Friday 22:00 UTC.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        if dt.weekday() != 4:
            return False

        close_minutes = self.FRIDAY_CLOSE_HOUR * 60
        current_minutes = dt.hour * 60 + dt.minute
        return close_minutes - minutes_before <= current_minutes < close_minutes

    def minutes_to_next_open(self, dt: datetime | None = None) -> int:
        """Approximate minutes until next market open."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        if self.is_market_open(dt):
            return 0

        weekday = dt.weekday()
        hour = dt.hour

        # Friday after close → Sunday 22:00 UTC
        if weekday == 4 and hour >= self.FRIDAY_CLOSE_HOUR:
            days_ahead = 2
            hours_to_open = (24 - hour - 1) + (days_ahead - 1) * 24 + self.SUNDAY_OPEN_HOUR
            return hours_to_open * 60 + (60 - dt.minute)

        # Saturday → Sunday 22:00
        if weekday == 5:
            hours_to_open = (24 - hour - 1) + self.SUNDAY_OPEN_HOUR
            return hours_to_open * 60 + (60 - dt.minute)

        # Sunday before 22:00
        if weekday == 6 and hour < self.SUNDAY_OPEN_HOUR:
            return (self.SUNDAY_OPEN_HOUR - hour) * 60 - dt.minute

        return 0
