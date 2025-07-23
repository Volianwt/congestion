# tests/test_loaders.py
import pandas as pd
import pytest
from src.loaders import load_volume, load_speed, load_summary, load_pedestrian_from_tmc, load_google_mobility

def test_volume_loader_basic():
    df = load_volume()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # core cols
    for col in ["time_start", "time_end", "volume_15min", "hour"]:
        assert col in df.columns
    # numeric check
    assert pd.api.types.is_integer_dtype(df["volume_15min"])

def test_speed_loader_basic():
    df = load_speed()
    assert len(df) > 0
    assert "time_start" in df.columns
    # at least one speed bin column
    assert any(c.startswith("vol_") and "kph" in c for c in df.columns)
    assert "speed_bin_total" in df.columns

def test_summary_loader_exists():
    try:
        df = load_summary()
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("summary file not downloaded - ok to skip")

def test_ped_from_tmc_basic():
    df = load_pedestrian_from_tmc()
    assert len(df) > 0
    # key columns
    for col in ["start_time", "end_time", "hour", "ped_count"]:
        assert col in df.columns
    # numeric
    assert df["ped_count"].ge(0).all()

def test_google_mobility_basic():
    try:
        df = load_google_mobility()
    except FileNotFoundError:
        import pytest
        pytest.skip("google mobility not downloaded yet")
        return
    assert len(df) > 0
    assert "date" in df.columns
    if "country_region_code" in df.columns:
        assert set(df["country_region_code"]) == {"CA"}