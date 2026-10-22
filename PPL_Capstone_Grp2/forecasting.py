"""Time series stationarity checks and SARIMA forecasting (Statsmodels)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from plots import save_fig


def adf(series, label):
    stat, p = adfuller(series.dropna())[:2]; ok = p < 0.05
    print(f"ADF [{label}]: stat={stat:.3f}, p={p:.4f} -> {'STATIONARY' if ok else 'NON-stationary'}")
    return ok


def forecast_series(df, horizon=12):
    print("\n================ TIME SERIES FORECASTING ================")
    s = df["demand"].resample("MS").mean()
    if df.index.max() < s.index[-1] + pd.offsets.MonthEnd(0):
        print(f"Dropping incomplete last month {s.index[-1].date()}"); s = s.iloc[:-1]
    s = s.interpolate()
    print(f"Monthly series: {len(s)} points, {s.index[0].date()} to {s.index[-1].date()}")

    dec = seasonal_decompose(s, model="additive", period=12)
    fig = dec.plot(); fig.set_size_inches(11, 8); save_fig("07_decomposition.png")

    ok_raw = adf(s, "monthly demand")
    sd = s.diff(12).dropna()
    d = 0 if adf(sd, "after seasonal differencing (lag 12)") else 1
    if d == 1: adf(sd.diff().dropna(), "after seasonal + regular differencing")
    print(f"Chosen differencing: seasonal D=1 (lag 12), regular d={d}")

    train, test = s.iloc[:-horizon], s.iloc[-horizon:]
    best, best_aic = None, np.inf
    for p in range(3):
        for q in range(3):
            try:
                m = ARIMA(train, order=(p, d, q), seasonal_order=(0, 1, 1, 12)).fit()
                if m.aic < best_aic: best, best_aic = (p, d, q), m.aic
            except Exception: continue
    print(f"Best order by AIC: SARIMA{best}x(0,1,1,12), AIC={best_aic:.1f}")

    pred = ARIMA(train, order=best, seasonal_order=(0, 1, 1, 12)).fit().forecast(horizon)
    naive = s.shift(12).iloc[-horizon:]
    mae, rmse = np.mean(np.abs(test - pred)), np.sqrt(np.mean((test - pred) ** 2))
    mape = 100 * np.mean(np.abs((test - pred) / test)); mae_n = np.mean(np.abs(test - naive))
    print(f"Hold-out last {horizon} months: MAE={mae:.1f} MU  RMSE={rmse:.1f} MU  MAPE={mape:.1f}% | seasonal-naive MAE={mae_n:.1f}")

    final = ARIMA(s, order=best, seasonal_order=(0, 1, 1, 12)).fit()
    lb = acorr_ljungbox(final.resid.iloc[13:], lags=[12], return_df=True)
    print(f"Ljung-Box residual test (lag 12): p={lb['lb_pvalue'].iloc[0]:.3f} -> "
          f"{'residuals look like white noise (good fit)' if lb['lb_pvalue'].iloc[0] > .05 else 'some structure left in residuals'}")
    fc = final.get_forecast(horizon); ci = fc.conf_int(); m = fc.predicted_mean
    print(f"\nNext {horizon} months forecast (MU/day):\n{m.round(1).to_string()}")

    plt.figure(figsize=(12, 4.5))
    plt.plot(s.iloc[-72:], label="Actual monthly mean")
    plt.plot(test.index, pred, "--", color="orange", label="Back-test (hold-out)")
    plt.plot(m.index, m.values, color="red", label=f"{horizon}-month forecast")
    plt.fill_between(m.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=.15, label="95% CI")
    plt.title(f"SARIMA{best}x(0,1,1,12) forecast of monthly mean demand"); plt.ylabel("MU per day"); plt.legend()
    save_fig("08_forecast.png")
    return best, mae, rmse, mape, mae_n, m
