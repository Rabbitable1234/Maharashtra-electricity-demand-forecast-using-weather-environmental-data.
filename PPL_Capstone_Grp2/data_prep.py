"""Loading, cleaning and feature engineering (NumPy / Pandas)."""
import numpy as np
import pandas as pd
from config import RENAME, START_YEAR


def load_demand(path, state):
    raw = pd.read_csv(path, parse_dates=["Date"])[["Date", state]]
    print(f"Demand file: {len(raw)} rows, {raw.Date.min().date()} to {raw.Date.max().date()}")
    dup = raw.Date.duplicated().sum()
    conflict = (raw.groupby("Date")[state].nunique() > 1).sum()
    print(f"Duplicate dates: {dup} (of which {conflict} have conflicting values -> averaged)")
    s = raw.groupby("Date")[state].mean().asfreq("D")           # unique, continuous daily index
    print(f"Missing days: {s.isna().sum()} (by year: {s[s.isna()].index.year.value_counts().sort_index().to_dict()})")
    s = s.interpolate(method="time", limit=7)                    # fill short gaps only (<= 7 days)
    print(f"Still missing after short-gap interpolation: {s.isna().sum()} (long gaps are left empty, not invented)")
    return s.rename("demand")


def load_weather(path):
    raw = pd.read_csv(path, parse_dates=["date"])
    df = raw.drop_duplicates("date").sort_values("date").set_index("date").asfreq("D")
    df = df.drop(columns=[c for c in df.columns if c.startswith("utci")])   # 90 trailing missing days
    df = df[list(RENAME) + ["2m_temperature_min", "2m_temperature_max"]].rename(columns=RENAME)
    df = df.loc[:, ~df.columns.duplicated()].interpolate(method="time", limit=5)
    for c in ["t_min", "t_mean", "t_max", "dewpoint"]:
        df[c] = df[c] - 273.15                                  # K -> deg C
    df["solar"] = df["solar"] / 1e6                             # J/m2 -> MJ/m2
    df["wind_speed"] = np.hypot(df["u_wind"], df["v_wind"])
    df["diurnal_range"] = df["t_max"] - df["t_min"]
    return df.drop(columns=["u_wind", "v_wind"])


def build_dataset(dem, wx):
    """Detrend demand, add calendar/lag features, join weather."""
    # Demand grows with population & economy. Dividing by the centred 365-day mean removes that
    # growth, leaving a 'weather-driven demand index' (100 = typical level for that year).
    trend = dem.rolling(365, center=True, min_periods=300).mean()
    idx = 100 * dem / trend
    z = (idx - idx.mean()) / idx.std()
    print(f"Outliers in demand index (|z|>4): {(z.abs() > 4).sum()} -> set to NaN")
    idx = idx.mask(z.abs() > 4)

    df = wx.join(dem).join(trend.rename("trend")).join(idx.rename("demand_idx"))
    df = df.loc[str(START_YEAR):]
    df["t30"] = df["t_mean"].rolling(30, min_periods=20).mean() # 30-day mean temperature (seasonal memory)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["year"], df["month"] = df.index.year, df.index.month
    df["season"] = np.select(
        [df.month.isin([12, 1, 2]), df.month.isin([3, 4, 5]), df.month.isin([6, 7, 8, 9])],
        ["Winter", "Summer", "Monsoon"], default="Post-monsoon")
    return df


def summaries(df):
    print("\n--- Descriptive statistics ---")
    print(df[["demand", "demand_idx", "t_mean", "t_min", "t_max", "dewpoint", "solar", "cloud", "wind_speed"]]
          .describe().round(2).T)
    print("\nMean daily demand by year (MU/day):\n", df.groupby("year")["demand"].mean().round(1).to_string())
    print("\nDemand index by season:\n", df.groupby("season")["demand_idx"].agg(["mean", "std"]).round(2))
    pk = df["demand"].idxmax()
    print(f"\nPeak demand day: {pk.date()} = {df.loc[pk, 'demand']:.1f} MU, T_mean = {df.loc[pk, 't_mean']:.1f} C")
    print("First rows via iloc:\n", df.iloc[:3, :4].round(2))
