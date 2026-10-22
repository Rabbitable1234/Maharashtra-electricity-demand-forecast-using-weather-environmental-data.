"""Statistical tests and regression (SciPy / Statsmodels)."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import PREDS, START_YEAR


def statistical_tests(df):
    print("\n================ STATISTICAL ANALYSIS ================")
    res = {}
    y = "demand_idx"
    d = df.dropna(subset=[y] + PREDS)

    # (a) Growth of raw demand: linear regression on annual means (full years only)
    ann = df.loc[(df.year >= START_YEAR) & (df.year <= 2023)].groupby("year")["demand"].mean()
    lr = stats.linregress(ann.index, ann.values)
    print(f"[Growth] demand rises {lr.slope:.1f} MU/day per year (r={lr.rvalue:.2f}, p={lr.pvalue:.1e}); "
          f"{ann.iloc[0]:.0f} -> {ann.iloc[-1]:.0f} MU/day ({100*(ann.iloc[-1]/ann.iloc[0]-1):.0f}% rise)")
    res["slope"], res["growth_pct"] = lr.slope, 100 * (ann.iloc[-1] / ann.iloc[0] - 1)

    # (b) Normality
    sh = stats.shapiro(d[y].sample(2000, random_state=1))
    z = (d[y] - d[y].mean()) / d[y].std(); ks = stats.kstest(z, "norm")
    print(f"\n[Normality] Shapiro p={sh.pvalue:.2e} | KS p={ks.pvalue:.2e} -> "
          f"{'normal' if sh.pvalue > .05 else 'NOT normal -> non-parametric tests used alongside'}")

    # (c) Correlation of weather with demand index
    print(f"\n[Correlation with demand index] Pearson r | Spearman rho | p")
    ct = {}
    for c in ["t_mean", "t_min", "t_max", "t30", "dewpoint", "solar", "cloud", "wind_speed", "diurnal_range"]:
        r, p = stats.pearsonr(d[c], d[y]); rho, _ = stats.spearmanr(d[c], d[y]); ct[c] = (r, rho, p)
        print(f"  {c:14s} r={r:+.3f}  rho={rho:+.3f}  p={p:.1e}")
    res["corr"] = ct

    # (d) Levene + weekday vs weekend
    groups = [g[y].values for _, g in d.groupby("season")]
    print(f"\n[Levene] equal variance across seasons? p={stats.levene(*groups).pvalue:.2e}")
    wd, we = d.loc[d.is_weekend == 0, y], d.loc[d.is_weekend == 1, y]
    t = stats.ttest_ind(wd, we, equal_var=False); mw = stats.mannwhitneyu(wd, we)
    print(f"[Independent t-test] weekday {wd.mean():.1f} vs weekend {we.mean():.1f}: t={t.statistic:.1f}, p={t.pvalue:.1e}")
    print(f"[Mann-Whitney U]     weekday vs weekend: p={mw.pvalue:.1e}")
    res["wk_gap"] = wd.mean() - we.mean()

    # (e) ANOVA + Kruskal-Wallis across seasons
    an = stats.f_oneway(*groups); kw = stats.kruskal(*groups)
    print(f"[One-way ANOVA] seasons: F={an.statistic:.1f}, p={an.pvalue:.1e} | [Kruskal-Wallis] H={kw.statistic:.1f}, p={kw.pvalue:.1e}")
    res["anova_p"] = an.pvalue

    # (f) Paired t-test + Wilcoxon: same 12 calendar months, 2014-18 vs 2019-23 (raw demand)
    early = df.loc[(df.year >= 2014) & (df.year <= 2018)].groupby("month")["demand"].mean()
    late = df.loc[(df.year >= 2019) & (df.year <= 2023)].groupby("month")["demand"].mean()
    pt = stats.ttest_rel(late, early); wl = stats.wilcoxon(late, early)
    print(f"[Paired t-test] monthly mean demand 2019-23 vs 2014-18: +{np.mean(late-early):.1f} MU/day, t={pt.statistic:.1f}, p={pt.pvalue:.1e}")
    print(f"[Wilcoxon signed-rank] p={wl.pvalue:.1e}")

    # (g) Chi-square: high-demand day (top 25% of index) vs season
    high = d[y] > d[y].quantile(.75); tab = pd.crosstab(d.season, high)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    print(f"[Chi-square] season vs high-demand day: chi2={chi2:.1f}, dof={dof}, p={p:.1e}\n{tab}")

    # (h) Multiple regression + multicollinearity check
    X = sm.add_constant(d[PREDS]); ols = sm.OLS(d[y], X).fit()
    print(f"\n[OLS regression on demand index] R2={ols.rsquared:.3f}, adj R2={ols.rsquared_adj:.3f}")
    print(ols.summary().tables[1])
    vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
    print("VIF (multicollinearity; >10 = problem):", {k: round(v, 1) for k, v in vif.items()})
    # same model WITHOUT temperature terms, to show how much temperature alone adds
    X2 = sm.add_constant(d[[c for c in PREDS if c not in ("t_mean", "t30")]])
    r2_no_t = sm.OLS(d[y], X2).fit().rsquared
    print(f"R2 without temperature terms = {r2_no_t:.3f} -> temperature adds {ols.rsquared - r2_no_t:.3f}")
    res.update(r2=ols.rsquared, ols=ols, r2_no_t=r2_no_t)
    return res
