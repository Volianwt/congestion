# tests/test_metrics.py
import pandas as pd
from src.loaders import load_volume
from src.metrics import compute_hourly_baseline, attach_ci

def test_ci_pipeline_small():
    veh = load_volume()
    # sum to hourly
    hourly = (
        veh.groupby(["location_name", "hour"], as_index=False)["volume_15min"]
           .sum()
           .rename(columns={"volume_15min": "volume_hour"})
    )
    base = compute_hourly_baseline(hourly, "volume_hour", time_col="hour",
                                   keys=["location_name"], by="weekday_hour")
    out = attach_ci(hourly, base, "volume_hour", how="weekday_hour", keys=["location_name"])
    assert "ci" in out.columns and "ci_level" in out.columns
    assert out["ci"].notna().sum() > 0
    assert set(out["ci_level"].unique()) <= {"low", "normal", "high", "unknown"}
