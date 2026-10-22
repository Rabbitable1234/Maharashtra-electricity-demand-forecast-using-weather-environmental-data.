"""All Matplotlib / Seaborn charts."""
import os
import matplotlib
matplotlib.use("Agg")                # delete this line if you want charts to pop up
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUT_DIR

sns.set_theme(style="whitegrid")


def save_fig(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, name), dpi=130); plt.close()


def plot_trend(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["demand"], lw=0.5, alpha=0.6, label="Daily demand")
    plt.plot(df["trend"], color="red", lw=2, label="365-day centred mean (trend)")
    plt.plot(df["demand"].ewm(span=30).mean(), color="black", lw=1, ls="--", label="Exp. weighted avg (span 30)")
    plt.title("Daily Electricity Demand - Maharashtra"); plt.xlabel("Date"); plt.ylabel("MU per day"); plt.legend()
    save_fig("01_demand_trend.png")


def plot_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["demand_idx"].dropna(), kde=True, ax=ax[0]); ax[0].set_title("Distribution of demand index")
    sns.violinplot(data=df, x="season", y="demand_idx", order=["Winter", "Summer", "Monsoon", "Post-monsoon"],
                   ax=ax[1], inner="quartile"); ax[1].set_title("Demand index by season")
    save_fig("02_distribution_season.png")


def plot_profiles(df):
    """Seasonal (month), weekly (weekday) and long-run (year) demand patterns."""
    d = df.assign(dev=df["demand_idx"] - 100)           # deviation from the yearly average
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=d, x="month", y="dev", errorbar=None, ax=ax[0], color="tab:orange")
    ax[0].set_title("Seasonal pattern: demand vs yearly average"); ax[0].set_xlabel("Month")
    ax[0].set_ylabel("Demand index - 100 (%)")
    sns.barplot(data=d, x=d.index.dayofweek, y="dev", errorbar=None, ax=ax[1], color="tab:blue")
    ax[1].set_title("Weekly pattern: Sunday is lowest"); ax[1].set_xlabel("Day of week (0=Mon, 6=Sun)")
    ax[1].set_ylabel("Demand index - 100 (%)")
    yearly = df.loc[df.year <= 2023].groupby("year")["demand"].mean()   # 2024 is a partial year
    ax[2].bar(yearly.index, yearly.values, color="tab:green")
    ax[2].set_title("Long-run growth (full years only)"); ax[2].set_xlabel("Year")
    ax[2].set_ylabel("Mean daily demand (MU/day)")
    save_fig("03_profiles.png")


def plot_heatmap(df):
    cols = ["demand_idx", "t_mean", "t_min", "t_max", "t30", "dewpoint", "solar", "cloud", "wind_speed", "diurnal_range"]
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix"); save_fig("04_correlation_heatmap.png")
    return corr


def plot_scatter(df):
    """Scatter + regression line of demand index against three key weather drivers."""
    d = df.dropna(subset=["demand_idx", "t_mean", "t30", "dewpoint"]).sample(2000, random_state=1)
    specs = [("t_mean", "mean temperature"), ("t30", "30-day mean temperature"), ("dewpoint", "dewpoint")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, name) in zip(axes, specs):
        sns.regplot(data=d, x=col, y="demand_idx", ax=ax, scatter_kws={"s": 5, "alpha": .3}, line_kws={"color": "red"})
        ax.set_title(f"Demand index vs {name}")
        ax.set_xlabel(col); ax.set_ylabel("Demand index (100 = yearly avg)")
    save_fig("05_scatter_drivers.png")


def plot_pair(df):
    d = df[["demand_idx", "t_mean", "dewpoint", "solar", "season"]].dropna().sample(1500, random_state=1)
    g = sns.pairplot(d, hue="season", plot_kws={"s": 6, "alpha": .5}, corner=True)
    g.savefig(os.path.join(OUT_DIR, "06_pairplot.png"), dpi=110); plt.close("all")
