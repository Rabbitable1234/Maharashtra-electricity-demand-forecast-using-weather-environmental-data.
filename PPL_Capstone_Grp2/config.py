"""Configuration shared by all modules."""
import os

WEATHER_CSV = "Maharashtra.csv"
DEMAND_CSV = "corrected_daily_energy_met_MU.csv"
STATE = "Maharashtra"
START_YEAR = 2014
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RENAME = {
    "10m_u_component_of_wind_mean": "u_wind", "10m_v_component_of_wind_mean": "v_wind",
    "2m_temperature_min": "t_min", "2m_temperature_mean": "t_mean", "2m_temperature_max": "t_max",
    "2m_dewpoint_temperature_mean": "dewpoint",
    "surface_solar_radiation_downwards_mean": "solar", "total_cloud_cover_mean": "cloud",
}
# Regression predictors (t_min / t_max omitted: >0.9 correlated with t_mean)
PREDS = ["t_mean", "t30", "diurnal_range", "dewpoint", "solar", "cloud", "wind_speed", "is_weekend"]
