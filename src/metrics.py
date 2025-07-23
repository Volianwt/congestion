# src/metrics.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np   

DERIVED_DIR = Path(__file__).resolve().parents[1] / "data" / "derived"


# ---------------------------
# Utils
# ---------------------------
def _classify_ci(ci: float, low_thr: float = 0.8, high_thr: float = 1.2) -> str:
    # 用 np.isinf 检查无穷
    if pd.isna(ci) or np.isinf(ci):
        return "unknown"
    if ci < low_thr:
        return "low"
    if ci <= high_thr:
        return "normal"
    return "high"



# ---------------------------
# Baseline (simple mean) + attach
# ---------------------------
def compute_hourly_baseline(df: pd.DataFrame,
                            value_col: str,
                            time_col: str = "hour",
                            keys: list[str] | None = None,
                            by: str = "weekday_hour") -> pd.DataFrame:
    """
    Calculate a baseline mean for each group.
    by:
        - 'weekday_hour': group by weekday (0=Mon) & hour_of_day
        - 'hour_of_day':  group only by hour_of_day
    """
    if keys is None:
        keys = ["location_name"]

    dfx = df.copy()
    if by == "weekday_hour":
        dfx["weekday"] = dfx[time_col].dt.dayofweek
        dfx["hour_of_day"] = dfx[time_col].dt.hour
        group_cols = keys + ["weekday", "hour_of_day"]
    elif by == "hour_of_day":
        dfx["hour_of_day"] = dfx[time_col].dt.hour
        group_cols = keys + ["hour_of_day"]
    else:
        raise ValueError("by must be 'weekday_hour' or 'hour_of_day'")

    baseline = (
        dfx.groupby(group_cols, dropna=False)[value_col]
           .mean()
           .rename("baseline_mean")
           .reset_index()
    )
    return baseline


def attach_ci(df: pd.DataFrame,
              baseline: pd.DataFrame,
              value_col: str,
              how: str = "weekday_hour",
              keys: list[str] | None = None,
              low_thr: float = 0.8,
              high_thr: float = 1.2) -> pd.DataFrame:
    """
    Merge baseline back and compute CI = current / baseline_mean.
    """
    if keys is None:
        keys = ["location_name"]

    dfx = df.copy()
    if how == "weekday_hour":
        dfx["weekday"] = dfx["hour"].dt.dayofweek
        dfx["hour_of_day"] = dfx["hour"].dt.hour
        merge_cols = keys + ["weekday", "hour_of_day"]
    else:
        dfx["hour_of_day"] = dfx["hour"].dt.hour
        merge_cols = keys + ["hour_of_day"]

    out = dfx.merge(baseline, on=merge_cols, how="left")
    out["ci"] = out[value_col] / out["baseline_mean"]
    out["ci_level"] = out["ci"].apply(lambda x: _classify_ci(x, low_thr, high_thr))
    return out


# ---------------------------
# Leave-one-out CI (解决 group size=1 导致 CI=1)
# ---------------------------
def attach_ci_leave1out(df: pd.DataFrame,
                        value_col: str = "volume_hour",
                        time_col: str = "hour",
                        keys: list[str] | None = None,
                        low_thr: float = 0.8,
                        high_thr: float = 1.2) -> pd.DataFrame:
    """
    对每条记录，baseline = 同组其它记录的平均值（自身被排除）。
    group = keys + weekday + hour_of_day

    若该组合仅 1 条记录，则 baseline_mean 为空，ci_level='unknown'。
    """
    if keys is None:
        keys = ["location_name"]

    dfx = df.copy()
    dfx["weekday"] = dfx[time_col].dt.dayofweek
    dfx["hour_of_day"] = dfx[time_col].dt.hour

    grp_cols = keys + ["weekday", "hour_of_day"]

    agg = (
        dfx.groupby(grp_cols)[value_col]
           .agg(sum_all="sum", n_all="count")
           .reset_index()
    )

    out = dfx.merge(agg, on=grp_cols, how="left")

    out["baseline_mean"] = (out["sum_all"] - out[value_col]) / (out["n_all"] - 1)
    out.loc[out["n_all"] <= 1, "baseline_mean"] = pd.NA

    out["ci"] = out[value_col] / out["baseline_mean"]
    # 清洗 0/NaN/inf
    zero_or_nan = out["baseline_mean"].isna() | (out["baseline_mean"] == 0)
    out.loc[zero_or_nan, "ci"] = np.nan
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out["ci_level"] = out["ci"].apply(lambda x: _classify_ci(x, low_thr, high_thr))
    return out


# ---------------------------
# Save helper
# ---------------------------
def save_parquet(df: pd.DataFrame, name: str) -> Path:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = DERIVED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
