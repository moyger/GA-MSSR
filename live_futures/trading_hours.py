"""CME equity futures trading hours management.

CME E-mini/Micro index futures (ES, NQ, MNQ, MES, RTY, YM) trade:
  - Sunday 5:00 PM CT → Friday 4:00 PM CT
  - With a daily maintenance break: 4:00 PM – 5:00 PM CT (Mon–Fri)

All times in this module use UTC:
  - CT = UTC-6 (standard) / UTC-5 (daylight)
  - Using CT offset of -6 (CST) for conservative calculation.
    During CDT (Mar–Nov), the break shifts by 1 hour in UTC.

Conservative UTC schedule (CST, winter):
  - Open:  Sunday  23:00 UTC
  - Break: Mon–Fri 22:00–23:00 UTC
  - Close: Friday  22:00 UTC (last break = final close for the week)
"""
from __future__ import annotations

from datetime import datetime, time, timezone


class CMETradingHours:
    """CME equity futures session management."""

    # Daily maintenance break (UTC, conservative CST)
    BREAK_START = time(22, 0)  # 4:00 PM CT
    BREAK_END = time(23, 0)    # 5:00 PM CT

    # Weekly boundaries
    # Friday close = start of Friday's break (22:00 UTC)
    # Sunday open = end of Sunday's "break" (23:00 UTC)
    FRIDAY_CLOSE_HOUR = 22
    SUNDAY_OPEN_HOUR = 23

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if the CME futures market is currently open.

        Returns False during:
          - Daily maintenance break (22:00–23:00 UTC, Mon–Fri)
          - Weekend (Friday 22:00 UTC → Sunday 23:00 UTC)
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        weekday = dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = dt.hour

        # Saturday: always closed
        if weekday == 5:
            return False

        # Sunday: open only after 23:00 UTC
        if weekday == 6:
            return hour >= self.SUNDAY_OPEN_HOUR

        # Friday: closed after 22:00 UTC (weekly close)
        if weekday == 4 and hour >= self.FRIDAY_CLOSE_HOUR:
            return False

        # Mon–Fri: closed during daily break 22:00–23:00 UTC
        if self.BREAK_START <= dt.timetz().replace(tzinfo=None) < self.BREAK_END:
            return False

        return True

    def should_flatten_for_weekend(
        self, dt: datetime | None = None, minutes_before: int = 30
    ) -> bool:
        """Check if we should flatten positions before the weekly close.

        Returns True if within `minutes_before` of Friday 22:00 UTC.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Only relevant on Friday
        if dt.weekday() != 4:
            return False

        close_minutes = self.FRIDAY_CLOSE_HOUR * 60
        current_minutes = dt.hour * 60 + dt.minute
        return close_minutes - minutes_before <= current_minutes < close_minutes

    def is_daily_break(self, dt: datetime | None = None) -> bool:
        """Check if currently in the daily maintenance break."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Weekend is not a "break" per se
        if dt.weekday() in (5, 6):
            return False

        t = dt.timetz().replace(tzinfo=None)
        return self.BREAK_START <= t < self.BREAK_END

    def minutes_to_next_open(self, dt: datetime | None = None) -> int:
        """Approximate minutes until next market open.

        Useful for logging when the market is closed.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        if self.is_market_open(dt):
            return 0

        weekday = dt.weekday()
        hour = dt.hour

        # During daily break (Mon–Thu): opens at 23:00 same day
        if weekday < 4 and hour >= 22:
            return (23 - hour) * 60 - dt.minute

        # Friday after close or Saturday or Sunday before open
        if weekday == 4 and hour >= 22:
            # Friday 22:xx → Sunday 23:00
            days_ahead = 2
            hours_to_open = (24 - hour - 1) + (days_ahead - 1) * 24 + 23
            return hours_to_open * 60 + (60 - dt.minute)
        if weekday == 5:
            # Saturday → Sunday 23:00
            hours_to_open = (24 - hour - 1) + 23
            return hours_to_open * 60 + (60 - dt.minute)
        if weekday == 6 and hour < 23:
            return (23 - hour) * 60 - dt.minute

        return 0
