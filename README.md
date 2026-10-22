# Analysis and Forecasting of Electricity Demand in Maharashtra Using Weather Parameters

Capstone Mini Project: Python Programming Laboratory (316U50L301), Lab Continuous Assessment
Somaiya Vidyavihar University | Theme: **Climate Energy**

**Group members:** _add names here_

---

## Overview

Electricity demand changes with time and with environmental conditions such as temperature, solar radiation, humidity, cloud cover and wind. This project analyses daily electricity demand in **Maharashtra (2014 to April 2024)** and studies how it relates to weather, using a full Python data-analysis workflow:

1. **Cleaning and manipulation** with NumPy / Pandas
2. **Visualisation** with Matplotlib / Seaborn
3. **Statistical analysis** with SciPy / Statsmodels
4. **Time-series forecasting** with a SARIMA model (Statsmodels)

## Datasets

| File | Content | Source |
|---|---|---|
| `corrected_daily_energy_met_MU.csv` | Daily energy met (demand) in Million Units (MU) for Indian states, from Grid-India | Zenodo records [14983361](https://zenodo.org/records/14983361) / [14983362](https://zenodo.org/records/14983362) (Hunt & Bloomfield) |
| `Maharashtra.csv` | Daily state-level ERA5 weather (temperature, dewpoint, wind, solar radiation, cloud cover) | Same Zenodo records |

Both files are used as downloaded; all cleaning is done in code (`data_prep.py`) so it is reproducible.

## Project structure

```
capstone.py          main script: runs the whole pipeline
config.py            file paths, constants, column renaming, regression predictors
data_prep.py         loading, cleaning, feature engineering
plots.py             Matplotlib / Seaborn charts
stats_analysis.py    SciPy / Statsmodels tests and regression
forecasting.py       stationarity checks and SARIMA forecast
results.txt          full console output of one run
outputs/             generated charts (01 to 08)
```

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels
python capstone.py
```

Keep all `.py` files and the two CSVs in the same folder. Charts are saved to `outputs/` and the full analysis is printed to the console.

## Methodology

### Data cleaning (`data_prep.py`)
- **Demand:** 3,955 rows; 55 duplicate dates (5 with conflicting values, averaged); 236 missing days. Gaps of up to 7 days are time-interpolated; longer gaps are left empty rather than invented (77 days remain missing).
- **Weather:** duplicates dropped, reindexed to a continuous daily index, gaps of up to 5 days interpolated. UTCI columns dropped (90 trailing missing days).
- **Unit conversion:** temperature and dewpoint from Kelvin to °C; solar radiation from J/m² to MJ/m².
- **Derived features:** wind speed from u/v components, diurnal temperature range, 30-day mean temperature (`t30`), weekend flag, month, year, season.
- **Detrending:** demand grows over the years, so it is divided by its centred 365-day mean to get a **demand index** (100 = typical level for that year). This isolates the weather and calendar effect. No outliers exceeded |z| > 4.
- **Analysis window:** 2014-01-01 to 2024-04-27 (3,770 days).

### Visualisation (`plots.py`)
Trend with moving averages, distribution and seasonal violin plots, monthly / weekday / yearly profiles, correlation heatmap, scatter and regression plots against key drivers, pair plot, seasonal decomposition, and the forecast chart.

### Statistical analysis (`stats_analysis.py`)
Linear regression for long-run growth, Shapiro-Wilk and KS normality tests, Pearson and Spearman correlations, Levene's test, independent t-test and Mann-Whitney U (weekday vs weekend), one-way ANOVA and Kruskal-Wallis (seasons), paired t-test and Wilcoxon (2014-18 vs 2019-23), chi-square (season vs high-demand days), and multiple OLS regression with VIF multicollinearity checks.

### Forecasting (`forecasting.py`)
Monthly mean demand, additive seasonal decomposition, ADF stationarity tests, and a SARIMA model with order chosen by AIC. It is back-tested on the last 12 months, compared with a seasonal-naive baseline, checked with a Ljung-Box residual test, and used to forecast 12 months ahead.

## Key results

| Finding | Result |
|---|---|
| Long-run growth | Mean daily demand rose from ~379 to ~549 MU/day (2014 to 2023, +45%), about +16.8 MU/day per year |
| Seasonality | Demand index is highest in **Summer (108.0)** and lowest in **Monsoon (94.5)**; seasonal differences are significant (ANOVA p ≈ 3e-291) |
| Strongest weather link | `t_max` (Pearson r = +0.71), followed by `solar` (+0.64) and `diurnal_range` (+0.56); `dewpoint` and `cloud` are negatively correlated |
| Regression | Weather + calendar variables explain **R² = 0.58** of the demand index; temperature terms add 0.11 |
| Weekday effect | Weekdays draw about 2.4 index points more than weekends (p < 1e-11) |
| Forecast | SARIMA(2,1,2)x(0,1,1,12): hold-out **MAE 25.7 MU/day, MAPE 4.5%** vs seasonal-naive MAE 48.9; Ljung-Box p = 0.95 |

## Sample outputs

| | |
|---|---|
| ![Demand trend](outputs/01_demand_trend.png) | ![Seasonal distribution](outputs/02_distribution_season.png) |
| ![Profiles](outputs/03_profiles.png) | ![Correlation heatmap](outputs/04_correlation_heatmap.png) |
| ![Scatter drivers](outputs/05_scatter_drivers.png) | ![Forecast](outputs/08_forecast.png) |

## Limitations

- Weather is a **state-level average**, so regional differences inside Maharashtra are not captured.
- No holiday, industrial-activity or tariff data; the 2020 COVID period disrupts demand.
- Reporting gaps exist (largest in 2013 and 2020); long gaps were left empty instead of filled.
- `diurnal_range` has a VIF above 10 (12.6), so individual regression coefficients should be interpreted cautiously; `t30` is not significant at the 5% level (p = 0.061).
- Some SARIMA fits in the AIC search showed convergence warnings; the chosen model passed residual diagnostics and beat the baseline, but the 12-month forecast should be treated as indicative.
- The linear growth test uses only 10 annual points.

## Tech stack

Python, NumPy, Pandas, Matplotlib, Seaborn, SciPy, Statsmodels

## Acknowledgements

Data: Grid-India daily energy-met reports and ERA5 reanalysis weather, compiled by Hunt & Bloomfield (Zenodo). Guidance from the Lab Faculty, Somaiya Vidyavihar University.
