"""Tests for NQ data loader."""
import pandas as pd
import pytest
from pathlib import Path
from data.loader import load_nq_data

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_nq_1m.csv"


class TestLoadNQData:
    def test_load_csv(self):
        df = load_nq_data(FIXTURE_PATH)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.dtypes.unique().tolist() == ["float64"]
        assert len(df) == 100

    def test_sorted_index(self):
        df = load_nq_data(FIXTURE_PATH)
        assert df.index.is_monotonic_increasing

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_nq_data("/nonexistent/path/data.csv")

    def test_date_filter_start(self):
        df = load_nq_data(FIXTURE_PATH, start="2024-01-02 10:00:00")
        assert df.index.min() >= pd.Timestamp("2024-01-02 10:00:00")

    def test_date_filter_end(self):
        df = load_nq_data(FIXTURE_PATH, end="2024-01-02 10:00:00")
        assert df.index.max() <= pd.Timestamp("2024-01-02 10:00:00")

    def test_date_filter_both(self):
        df = load_nq_data(
            FIXTURE_PATH,
            start="2024-01-02 10:00:00",
            end="2024-01-02 10:30:00",
        )
        assert df.index.min() >= pd.Timestamp("2024-01-02 10:00:00")
        assert df.index.max() <= pd.Timestamp("2024-01-02 10:30:00")

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_nq_data(bad_file)

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("timestamp,foo,bar\n2024-01-01,1,2\n")
        with pytest.raises(ValueError, match="Missing required"):
            load_nq_data(csv)

    def test_missing_volume_filled(self, tmp_path):
        csv = tmp_path / "no_vol.csv"
        csv.write_text(
            "timestamp,open,high,low,close\n"
            "2024-01-01 09:30:00,16800,16805,16798,16803\n"
            "2024-01-01 09:31:00,16803,16810,16802,16808\n"
        )
        df = load_nq_data(csv)
        assert "volume" in df.columns
        assert (df["volume"] == 0.0).all()
