"""Tests for CME trading hours management."""
from datetime import datetime, timezone

from live_futures.trading_hours import CMETradingHours


def _utc(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


class TestIsMarketOpen:
    def setup_method(self):
        self.hours = CMETradingHours()

    # Regular trading hours
    def test_monday_midday_open(self):
        # Monday 14:00 UTC -> open
        assert self.hours.is_market_open(_utc(2026, 2, 23, 14, 0)) is True

    def test_wednesday_morning_open(self):
        assert self.hours.is_market_open(_utc(2026, 2, 25, 10, 0)) is True

    # Daily break
    def test_monday_break_closed(self):
        # Monday 22:30 UTC -> daily break
        assert self.hours.is_market_open(_utc(2026, 2, 23, 22, 30)) is False

    def test_tuesday_break_start_closed(self):
        assert self.hours.is_market_open(_utc(2026, 2, 24, 22, 0)) is False

    def test_tuesday_break_end_open(self):
        # 23:00 UTC -> break ends, market reopens
        assert self.hours.is_market_open(_utc(2026, 2, 24, 23, 0)) is True

    # Weekend
    def test_saturday_closed(self):
        assert self.hours.is_market_open(_utc(2026, 2, 28, 12, 0)) is False

    def test_sunday_before_open_closed(self):
        # Sunday 20:00 UTC -> still closed
        assert self.hours.is_market_open(_utc(2026, 3, 1, 20, 0)) is False

    def test_sunday_open(self):
        # Sunday 23:00 UTC -> market opens
        assert self.hours.is_market_open(_utc(2026, 3, 1, 23, 0)) is True

    # Friday close
    def test_friday_before_close_open(self):
        assert self.hours.is_market_open(_utc(2026, 2, 27, 21, 0)) is True

    def test_friday_at_close(self):
        # Friday 22:00 UTC -> closed (weekly close)
        assert self.hours.is_market_open(_utc(2026, 2, 27, 22, 0)) is False

    def test_friday_after_close(self):
        assert self.hours.is_market_open(_utc(2026, 2, 27, 23, 0)) is False


class TestShouldFlattenForWeekend:
    def setup_method(self):
        self.hours = CMETradingHours()

    def test_not_friday(self):
        # Wednesday -> no flatten
        assert self.hours.should_flatten_for_weekend(
            _utc(2026, 2, 25, 21, 30)
        ) is False

    def test_friday_too_early(self):
        # Friday 20:00 -> not within 30 min of close
        assert self.hours.should_flatten_for_weekend(
            _utc(2026, 2, 27, 20, 0)
        ) is False

    def test_friday_within_window(self):
        # Friday 21:30 -> within 30 min of 22:00 close
        assert self.hours.should_flatten_for_weekend(
            _utc(2026, 2, 27, 21, 30)
        ) is True

    def test_friday_at_close(self):
        # Friday 22:00 -> at close, not within window
        assert self.hours.should_flatten_for_weekend(
            _utc(2026, 2, 27, 22, 0)
        ) is False

    def test_custom_minutes_before(self):
        # Friday 21:00 with 60 min buffer -> should flatten
        assert self.hours.should_flatten_for_weekend(
            _utc(2026, 2, 27, 21, 0), minutes_before=60,
        ) is True


class TestIsDailyBreak:
    def setup_method(self):
        self.hours = CMETradingHours()

    def test_during_break(self):
        assert self.hours.is_daily_break(_utc(2026, 2, 25, 22, 30)) is True

    def test_before_break(self):
        assert self.hours.is_daily_break(_utc(2026, 2, 25, 21, 59)) is False

    def test_after_break(self):
        assert self.hours.is_daily_break(_utc(2026, 2, 25, 23, 0)) is False

    def test_weekend_not_break(self):
        assert self.hours.is_daily_break(_utc(2026, 2, 28, 22, 30)) is False


class TestMinutesToNextOpen:
    def setup_method(self):
        self.hours = CMETradingHours()

    def test_already_open(self):
        assert self.hours.minutes_to_next_open(_utc(2026, 2, 25, 14, 0)) == 0

    def test_during_break(self):
        # 22:30 -> opens at 23:00 = 30 min
        result = self.hours.minutes_to_next_open(_utc(2026, 2, 25, 22, 30))
        assert result == 30
