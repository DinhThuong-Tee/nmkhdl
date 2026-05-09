import os
import glob
import numpy as np
import pandas as pd

def parse_lod(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("<"):
            try:
                return float(x[1:]) / 2
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan
    return x

def normalize_colname(c):
    return (
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


COLUMN_MAP = {
    # bảng tiêu chí  → cột HK sau normalize
    "DO": "dissolved_oxygen_mgl",
    "Temperature": "temperature_°c",
    "pH": "ph",
    "Salinity": "salinity_psu",
    "NH3": "unionised_ammonia_mgl",
    "PO4": "orthophosphate_phosphorus_mgl",
    "BOD5": "5_day_biochemical_oxygen_demand_mgl",
    "TSS": "suspended_solids_mgl",
    "Coliform": "faecal_coliforms_cfu100ml"
}


def load_hk_water(data_dir):
    files = glob.glob(os.path.join(data_dir, "marine_water_quality_*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)

        # normalize header
        df.columns = [normalize_colname(c) for c in df.columns]

        # rename bắt buộc
        df = df.rename(columns={
            "dates": "date",
            "station": "station",
            "depth": "depth"
        })

        # parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # parse numeric
        for c in df.columns:
            if c not in ["date", "station", "depth"]:
                df[c] = df[c].apply(parse_lod)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("❌ Không load được file HK nào")

    return pd.concat(dfs, ignore_index=True)

FINAL_COLUMNS = [
    "DO","Temperature","pH","Salinity","Alkalinity","Transparency",
    "NH3","H2S","PO4","BOD5","COD","Coliform","TSS",
    "CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"
]

def aggregate_quarter(df, depth_value):
    depth_value = depth_value.lower()
    df = df[df["depth"].str.lower().str.contains(depth_value)].copy()

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    numeric_cols = [
        c for c in df.columns
        if c not in ["date", "station", "depth", "month", "quarter"]
    ]

    monthly = (
        df.groupby(["station", "month"])[numeric_cols]
        .mean()
        .reset_index()
    )

    monthly["quarter"] = monthly["month"].dt.to_period("Q").dt.to_timestamp()

    quarterly = (
        monthly.groupby(["station", "quarter"])[numeric_cols]
        .mean()
        .reset_index()
    )

    return quarterly

