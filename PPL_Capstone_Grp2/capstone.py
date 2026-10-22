"""
Capstone Mini Project - Python Programming Laboratory (316U50L301)
Analysis and Forecasting of Electricity Demand in Maharashtra Using Weather Parameters

Run:  python capstone.py     (keep all .py files + the two CSVs in the same folder)

User-defined modules:
  config.py          paths, constants, predictor list
  data_prep.py       load / clean / feature engineering
  plots.py           Matplotlib & Seaborn charts
  stats_analysis.py  SciPy / Statsmodels tests, regression
  forecasting.py     stationarity check + SARIMA forecast
Data: Grid-India daily energy met (corrected) + ERA5 weather, Hunt & Bloomfield, Zenodo 14983361/2
"""
import warnings
warnings.filterwarnings("ignore")

from config import DEMAND_CSV, WEATHER_CSV, STATE, START_YEAR, OUT_DIR
from data_prep import load_demand, load_weather, build_dataset, summaries
from plots import (plot_trend, plot_distribution, plot_profiles,
                   plot_heatmap, plot_scatter, plot_pair)
from stats_analysis import statistical_tests
from forecasting import forecast_series


def main():
    dem = load_demand(DEMAND_CSV, STATE)
    wx = load_weather(WEATHER_CSV)
    df = build_dataset(dem, wx)
    print(f"\nAnalysis window: {df.index.min().date()} to {df.index.max().date()} ({len(df)} days)")
    summaries(df)

    plot_trend(df); plot_distribution(df); plot_profiles(df)
    plot_heatmap(df); plot_scatter(df); plot_pair(df)

    res = statistical_tests(df)
    order, mae, rmse, mape, mae_n, fc = forecast_series(df)

    ct = res["corr"]; best = max(ct, key=lambda k: abs(ct[k][0]))
    sm_ = df.groupby("season")["demand_idx"].mean()
    print("\n================ CONCLUSION ================")
    print(f"1. Demand grew {res['growth_pct']:.0f}% over 2014-2023 (+{res['slope']:.1f} MU/day each year), so trend is removed before relating it to weather.")
    print(f"2. After detrending, demand is highest in {sm_.idxmax()} (index {sm_.max():.1f}) and lowest in {sm_.idxmin()} ({sm_.min():.1f}); seasonal differences are significant (ANOVA p={res['anova_p']:.1e}).")
    print(f"3. Strongest single weather link: {best} (r={ct[best][0]:+.2f}). Weather + calendar explain R2={res['r2']:.2f} of daily demand variation; temperature terms add {res['r2']-res['r2_no_t']:.2f}.")
    print(f"4. Weekdays draw about {res['wk_gap']:.1f} index points more than weekends (likely economic-activity effect).")
    print(f"5. SARIMA{order}x(0,1,1,12) hold-out MAE={mae:.1f} MU/day ({mape:.1f}% MAPE) vs seasonal-naive {mae_n:.1f} -> {'beats' if mae < mae_n else 'does not beat'} the baseline.")
    print("6. Limitations: state-average weather, no holiday/industrial data, 2020 COVID disruption, gaps in reporting.")
    print(f"\nCharts saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
