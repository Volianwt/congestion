# src/loaders.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import glob

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# ---------- helpers ----------
def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, **kwargs)
    if df.empty:
        raise ValueError(f"{path} is empty")
    # normalize headers
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("&", "and")
    )
    return df

def _parse_time(df: pd.DataFrame, start_col: str = "time_start", end_col: str = "time_end") -> pd.DataFrame:
    """Parse start/end as UTC datetime, add hour column for grouping if needed."""
    if start_col in df.columns:
        df[start_col] = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    if end_col in df.columns:
        df[end_col] = pd.to_datetime(df[end_col], utc=True, errors="coerce")
    # drop rows without time
    if start_col in df.columns:
        df = df.dropna(subset=[start_col])
        df["hour"] = df[start_col].dt.floor("h")
    return df

# ---------- loaders ----------
def load_volume(pattern: str = "toronto_volume_2020_2024*.csv") -> pd.DataFrame:
    """
    Columns we expect (from your sample):
    ['id','count_id','location_name','longitude','latitude','centreline_id',
     'time_start','time_end','direction','volume_15min']
    """
    files = glob.glob(str(RAW_DIR / pattern))
    if not files:
        raise FileNotFoundError("No volume csv matched.")
    dfs = []
    for f in files:
        df = _read_csv(Path(f))
        df = _parse_time(df)
        # ensure numeric
        if "volume_15min" in df.columns:
            df["volume_15min"] = pd.to_numeric(df["volume_15min"], errors="coerce").fillna(0).astype(int)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out

def load_speed(pattern: str = "toronto_speed_2020_2024*.csv") -> pd.DataFrame:
    """
    Expect columns:
    ['id','count_id','location_name','longitude','latitude','centreline_id',
     'time_start','time_end','direction',
     'vol_1_19kph','vol_20_25kph',...,'vol_81_160kph']
    """
    files = glob.glob(str(RAW_DIR / pattern))
    if not files:
        raise FileNotFoundError("No speed csv matched.")
    dfs = []
    for f in files:
        df = _read_csv(Path(f))
        df = _parse_time(df)
        # make a total_speed_volume column (sum of all bins) for quick checks
        speed_bins = [c for c in df.columns if c.startswith("vol_") and "kph" in c]
        if speed_bins:
            df["speed_bin_total"] = df[speed_bins].sum(axis=1)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out

def load_summary(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW_DIR / "toronto_summary_recent.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_csv(path)

def load_pedestrian_from_tmc(path: Path | None = None) -> pd.DataFrame:
    """
    TMC file columns (you showed):
    ['_id','count_id','count_date','location_name','longitude','latitude','centreline_type',
     'centreline_id','px','start_time','end_time',
     'n_appr_peds','s_appr_peds','e_appr_peds','w_appr_peds', ... bikes, cars, trucks, buses ...]

    We will:
      - parse start_time/end_time to UTC datetimes
      - create 'hour' = floor(start_time)
      - sum the four *_appr_peds to a single 'ped_count'
    """
    path = Path(path) if path else RAW_DIR / "toronto_tmc_2020_2029.csv"
    df = _read_csv(path)

    # Parse times (note column names are start_time / end_time)
    df = _parse_time(df, start_col="start_time", end_col="end_time")

    # Pedestrian columns
    ped_cols = ["n_appr_peds", "s_appr_peds", "e_appr_peds", "w_appr_peds"]
    for c in ped_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            raise KeyError(f"Missing pedestrian column: {c}")

    df["ped_count"] = df[ped_cols].sum(axis=1)

    return df

def load_google_mobility(path: Path | None = None, country_code: str = "CA") -> pd.DataFrame:
    """
    Google mobility CSV columns (typical):
    'country_region_code','country_region','sub_region_1','date',
    'retail_and_recreation_percent_change_from_baseline', ...
    """
    path = Path(path) if path else RAW_DIR / "google_mobility_global.csv"
    df = _read_csv(path)
    # date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
    # filter to Canada
    if "country_region_code" in df.columns:
        df = df[df["country_region_code"] == country_code]
    return df

