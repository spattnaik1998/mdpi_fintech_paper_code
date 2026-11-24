"""Econometric and reporting helpers shared by the notebooks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from .config import FIGURES_DIR, TABLES_DIR


# ------------------------------------------------------------------ panel regressions
def panel_fe(df: pd.DataFrame, y: str, xs: list[str], entity="code", time="year",
             entity_effects=True, time_effects=True, cov="clustered"):
    """Two-way fixed-effects OLS via linearmodels.

    cov: 'clustered' (by industry), 'twoway' (industry and year), or 'driscoll-kraay'
    (robust to cross-sectional and serial correlation).
    """
    from linearmodels.panel import PanelOLS

    d = df[[entity, time, y, *xs]].dropna().set_index([entity, time])
    mod = PanelOLS(d[y], d[xs], entity_effects=entity_effects, time_effects=time_effects,
                   drop_absorbed=True, check_rank=False)
    if cov == "clustered":
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    elif cov == "twoway":
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    elif cov == "driscoll-kraay":
        res = mod.fit(cov_type="kernel", kernel="bartlett")
    else:
        res = mod.fit()
    return res


def tidy(res, model: str = "", keep: list[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"coef": res.params, "se": res.std_errors, "t": res.tstats, "p": res.pvalues})
    ci = res.conf_int()
    out["ci_low"], out["ci_high"] = ci.iloc[:, 0], ci.iloc[:, 1]
    out = out.reset_index().rename(columns={"index": "term"})
    if keep:
        out = out[out["term"].isin(keep)]
    out["model"] = model
    out["nobs"] = int(res.nobs)
    out["n_entities"] = int(res.entity_info["total"]) if hasattr(res, "entity_info") else np.nan
    # R2 of the fully demeaned model (industry and year effects removed). linearmodels'
    # 'rsquared_within' ignores year effects and can be negative, so it is not reported.
    out["r2_within"] = getattr(res, "rsquared", np.nan)
    return out


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def regression_table(tidies: list[pd.DataFrame], terms: list[str], digits: int = 3) -> pd.DataFrame:
    """Publication-style table: coefficients with stars and SEs in brackets, one column per model."""
    cols = {}
    for t in tidies:
        name = t["model"].iloc[0]
        col = {}
        for term in terms:
            r = t[t["term"] == term]
            if len(r):
                r = r.iloc[0]
                col[term] = f"{r.coef:.{digits}f}{stars(r.p)}"
                col[term + " (se)"] = f"({r.se:.{digits}f})"
            else:
                col[term], col[term + " (se)"] = "", ""
        col["Observations"] = f"{int(t['nobs'].iloc[0])}"
        col["Industries"] = f"{int(t['n_entities'].iloc[0])}" if pd.notna(t["n_entities"].iloc[0]) else ""
        col["R2 (net of FE)"] = f"{t['r2_within'].iloc[0]:.3f}" if pd.notna(t["r2_within"].iloc[0]) else ""
        cols[name] = col
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ growth & breaks
def loglinear_trend(years: pd.Series, y: pd.Series, hac_lags: int = 2) -> dict:
    """Fit log(y) = a + g*t with Newey-West SEs; return growth rate, doubling time and fit stats."""
    m = pd.DataFrame({"t": years.values - years.min(), "ly": np.log(y.values)}).replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(m["t"])
    r = sm.OLS(m["ly"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    g = r.params["t"]
    # compare exponential vs linear trend in levels (RMSE)
    lvl = np.exp(m["ly"])
    lin = sm.OLS(lvl, X).fit()
    rmse_exp = np.sqrt(np.mean((lvl - np.exp(r.fittedvalues)) ** 2))
    rmse_lin = np.sqrt(np.mean((lvl - lin.fittedvalues) ** 2))
    return {"growth_rate": np.expm1(g), "log_slope": g, "se": r.bse["t"], "p": r.pvalues["t"],
            "doubling_years": np.log(2) / g if g > 0 else np.nan, "r2_log": r.rsquared,
            "rmse_exponential": rmse_exp, "rmse_linear": rmse_lin,
            "exp_beats_linear": bool(rmse_exp < rmse_lin), "n": len(m)}


def chow_test(years: np.ndarray, y: np.ndarray, break_year: int) -> dict:
    """Chow test for a break in a linear trend at break_year."""
    t = years - years.min()
    X = sm.add_constant(t)
    full = sm.OLS(y, X).fit()
    a, b = years < break_year, years >= break_year
    if a.sum() < 3 or b.sum() < 3:
        return {"F": np.nan, "p": np.nan}
    r1, r2 = sm.OLS(y[a], X[a]).fit(), sm.OLS(y[b], X[b]).fit()
    k, n = X.shape[1], len(y)
    ssr_u = r1.ssr + r2.ssr
    F = ((full.ssr - ssr_u) / k) / (ssr_u / (n - 2 * k))
    return {"F": F, "p": 1 - sps.f.cdf(F, k, n - 2 * k)}


def sup_f_break(years: np.ndarray, y: np.ndarray, trim: float = 0.15) -> dict:
    """Quandt-Andrews sup-F test for an unknown break date in a linear trend.

    Reports the sup-F statistic (Andrews, 1993), the most likely break year and a
    moving-block-bootstrap p-value under the null of a single stable trend.
    """
    n = len(y)
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    fs = []
    for i in range(max(lo, 3), min(hi, n - 3)):
        fs.append((years[i], chow_test(years, y, years[i])["F"]))
    if not fs:
        return {"supF": np.nan, "break_year": np.nan, "p_boot": np.nan}
    by, sf = max(fs, key=lambda x: x[1])
    # residual bootstrap under the null of no break
    t = years - years.min()
    X = sm.add_constant(t)
    base = sm.OLS(y, X).fit()
    # moving-block bootstrap of residuals (preserves serial correlation)
    rng = np.random.default_rng(0)
    resid = np.asarray(base.resid)
    L = max(2, int(round(n ** (1 / 3))))
    exceed, B = 0, 499
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
        eb = np.concatenate([resid[s:s + L] for s in starts])[:n]
        yb = np.asarray(base.fittedvalues) + eb
        fb = max(chow_test(years, yb, years[i])["F"] for i in range(max(lo, 3), min(hi, n - 3)))
        exceed += fb >= sf
    return {"supF": sf, "break_year": int(by), "p_boot": (exceed + 1) / (B + 1)}


def pelt_breaks(y: np.ndarray, years: np.ndarray, penalty: float | None = None, min_size: int = 4) -> list[int]:
    """Multiple change points in the mean of a series (e.g., growth rates) via PELT (ruptures)."""
    import ruptures as rpt
    y = np.asarray(y, dtype=float)
    if penalty is None:
        penalty = 2 * np.log(len(y)) * np.var(y)
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(y.reshape(-1, 1))
    bkps = algo.predict(pen=penalty)
    return [int(years[b]) for b in bkps[:-1]]


def growth_break_hac(years: np.ndarray, log_level: np.ndarray, break_year: int, lags: int = 3) -> dict:
    """Difference in mean annual log growth after vs before break_year, Newey-West SEs."""
    g = np.diff(log_level)
    post = (years[1:] >= break_year).astype(float)
    r = sm.OLS(g, sm.add_constant(post)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"diff": r.params[1], "se": r.bse[1], "p": r.pvalues[1],
            "growth_pre": r.params[0], "growth_post": r.params[0] + r.params[1]}


def lincomb(res, terms: list[str]) -> dict:
    """Estimate, SE and two-sided p of a sum of coefficients (e.g. beta + beta_finance)."""
    b = sum(res.params[t] for t in terms)
    V = res.cov
    var = sum(V.loc[a, c] for a in terms for c in terms)
    se = float(np.sqrt(var))
    p = 2 * (1 - sps.norm.cdf(abs(b / se)))
    return {"estimate": float(b), "se": se, "p": float(p)}


def directional(estimate: float, p_two_sided: float, expected: str) -> float:
    """One-sided p-value for H1: sign(estimate) == expected ('+' or '-')."""
    right_sign = (estimate > 0) if expected == "+" else (estimate < 0)
    return p_two_sided / 2 if right_sign else 1 - p_two_sided / 2


# ------------------------------------------------------------------ forecasting tests
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """DM test (Harvey-Leybourne-Newbold small-sample correction). Positive stat => model 2 better."""
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var) if var > 0 else np.nan
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    p = 2 * (1 - sps.t.cdf(abs(stat), df=n - 1))
    return {"DM": stat, "p": p, "n": n}


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = pvals.astype(float)
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


# ------------------------------------------------------------------ output helpers
def save_table(df: pd.DataFrame, name: str, index: bool = True, caption: str = "") -> None:
    """Write CSV and a LaTeX tabular (no jinja2 dependency) to results/tables/."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    d = df.reset_index() if index else df.copy()
    def esc(x):
        s = f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x)
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    lines = [r"\begin{table}[H]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{esc(caption)}}}")
    lines += [r"\begin{tabular}{" + "l" * d.shape[1] + "}", r"\toprule",
              " & ".join(esc(c) for c in d.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in d.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "legend.frameon": False, "figure.autolayout": True,
    })


PALETTE = {"finance": "#1F4E79", "other": "#A6A6A6", "information": "#C55A11", "accent": "#2E75B6"}


def savefig(fig, name: str):
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
