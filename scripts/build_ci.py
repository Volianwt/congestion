# scripts/build_ci.py
# scripts/build_ci.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.loaders import load_volume
from src.metrics import compute_hourly_baseline, attach_ci, save_parquet, attach_ci_leave1out
import pandas as pd
import numpy as np

# 1. load raw 15-min volume and aggregate to hourly
veh = load_volume()  # has 'hour' and 'volume_15min'

hourly = (
    veh.groupby(["location_name", "hour"], as_index=False)["volume_15min"]
       .sum()
       .rename(columns={"volume_15min": "volume_hour"})
)

# 2. baseline (weekday-hour)
baseline = compute_hourly_baseline(hourly,
                                   value_col="volume_hour",
                                   time_col="hour",
                                   keys=["location_name"],
                                   by="weekday_hour")

# 3. attach CI (leave-one-out)
with_ci = attach_ci_leave1out(hourly,
                              value_col="volume_hour",
                              time_col="hour",
                              keys=["location_name"])

# ---- fallback: for rows where baseline_mean is NaN, use hour_of_day baseline ----
fallback = compute_hourly_baseline(hourly,
                                   value_col="volume_hour",
                                   time_col="hour",
                                   keys=["location_name"],
                                   by="hour_of_day")

with_ci = with_ci.merge(fallback, on=["location_name", "hour_of_day"], how="left",
                        suffixes=("", "_fallback"))

na_mask = with_ci["baseline_mean"].isna()
with_ci.loc[na_mask, "baseline_mean"] = with_ci.loc[na_mask, "baseline_mean_fallback"]
with_ci.loc[na_mask, "ci"] = with_ci.loc[na_mask, "volume_hour"] / with_ci.loc[na_mask, "baseline_mean"]


# 如果 baseline_mean 为 0 或 NaN，CI 设 NaN
zero_or_nan = (with_ci["baseline_mean"].isna()) | (with_ci["baseline_mean"] == 0)
with_ci.loc[zero_or_nan, "ci"] = np.nan

# 把 inf/-inf 也设为 NaN
with_ci.replace([np.inf, -np.inf], np.nan, inplace=True)

# 重新标注等级
from src.metrics import _classify_ci
with_ci["ci_level"] = with_ci["ci"].apply(_classify_ci)
# 重新打标签
from src.metrics import _classify_ci
with_ci["ci_level"] = with_ci["ci"].apply(_classify_ci)
# -------------------------------------------------------------------------------

# 4. save
save_parquet(with_ci, "vehicle_ci")
print("Done. Rows:", len(with_ci))

