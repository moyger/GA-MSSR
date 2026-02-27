"""Tests for CME contract symbol resolution."""
from datetime import date

from live_futures.contract_utils import (
    get_front_month,
    get_continuous_symbol,
    get_polygon_symbol,
    next_roll_date,
)


class TestGetFrontMonth:
    def test_feb_returns_march(self):
        assert get_front_month("MNQ", date(2026, 2, 26)) == "MNQH6"

    def test_jan_returns_march(self):
        assert get_front_month("NQ", date(2026, 1, 15)) == "NQH6"

    def test_march_before_roll(self):
        # Before roll day (14th) -> still March
        assert get_front_month("MNQ", date(2026, 3, 10)) == "MNQH6"

    def test_march_after_roll(self):
        # After roll day -> June
        assert get_front_month("MNQ", date(2026, 3, 20)) == "MNQM6"

    def test_june_after_roll(self):
        assert get_front_month("ES", date(2026, 6, 20)) == "ESU6"

    def test_september_after_roll(self):
        assert get_front_month("MNQ", date(2026, 9, 20)) == "MNQZ6"

    def test_december_after_roll(self):
        # After Dec roll -> next year March
        assert get_front_month("MNQ", date(2026, 12, 20)) == "MNQH7"

    def test_november_returns_december(self):
        assert get_front_month("NQ", date(2026, 11, 1)) == "NQZ6"


class TestGetContinuousSymbol:
    def test_mnq(self):
        assert get_continuous_symbol("MNQ") == "@MNQ"

    def test_nq(self):
        assert get_continuous_symbol("NQ") == "@NQ"

    def test_es(self):
        assert get_continuous_symbol("ES") == "@ES"


class TestGetPolygonSymbol:
    def test_feb_2026(self):
        assert get_polygon_symbol("MNQ", date(2026, 2, 26)) == "MNQH2026"

    def test_dec_rollover(self):
        assert get_polygon_symbol("MNQ", date(2026, 12, 20)) == "MNQH2027"


class TestNextRollDate:
    def test_feb_returns_march_roll(self):
        result = next_roll_date("MNQ", date(2026, 2, 1))
        assert result == date(2026, 3, 14)

    def test_after_march_roll_returns_june(self):
        result = next_roll_date("MNQ", date(2026, 3, 20))
        assert result == date(2026, 6, 14)

    def test_after_december_returns_next_year_march(self):
        result = next_roll_date("MNQ", date(2026, 12, 20))
        assert result == date(2027, 3, 14)
