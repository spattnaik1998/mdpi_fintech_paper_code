"""Build the industry-year panel used throughout the project.

Sources
-------
BEA GDPbyIndustry : nominal value added, real value added, components of value added
BEA NIPA          : full-time-equivalent (FTE) employees by industry (Table 6.5D)
BEA FixedAssets   : intellectual-property-products (IPP) and equipment stocks/investment by industry
FRED              : macro-financial conditions and price deflators
BLS               : finance-subsector employment/earnings and economy-wide labour share (checks)
Census            : Business Formation Statistics by sector (dynamism control)

Every BEA table is located by its *description* at run time (not by a hard-coded ID),
so the pipeline survives BEA renumbering. Every choice is logged in
data/processed/pipeline_log.txt for transparency.
"""
from __future__ import annotations

import difflib
import re
from datetime import date

import numpy as np
import pandas as pd

from . import api_clients as api
from .config import (BREAK_YEAR, FINANCE_PREFIX, IT_PRODUCING_CODES, MAIN_SAMPLE_EXCLUDE, PROCESSED_DIR,
                     START_YEAR)

LOG: list[str] = []


def log(msg: str):
    LOG.append(msg)
    print(msg)


# ------------------------------------------------------------------ helpers
def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\\\d+\\", " ", s)          # BEA footnote markers like \1\
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z ]", " ", s)
    words = [w for w in s.split() if w not in {"and", "the", "of"}]
    return " ".join(words)


def _std_gdpi(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a GDPbyIndustry GetData frame -> code, desc, year, value."""
    cols = {c.lower(): c for c in df.columns}
    if df.empty or "industry" not in cols:
        return pd.DataFrame(columns=["code", "desc", "year", "value"])
    desc_col = next((c for lc, c in cols.items() if "industr" in lc and "desc" in lc), None)
    out = pd.DataFrame({
        "code": df[cols["industry"]].astype(str).str.strip(),
        "desc": df[desc_col].astype(str).str.strip() if desc_col else "",
        "year": pd.to_numeric(df[cols["year"]], errors="coerce"),
        "value": api.to_number(df[cols["datavalue"]]),
    })
    if "frequency" in cols:
        out = out[df[cols["frequency"]].astype(str).str.upper().values == "A"]
    return out.dropna(subset=["year"]).astype({"year": int})


def _std_lines(df: pd.DataFrame, to_billions: bool = True) -> pd.DataFrame:
    """Standardise NIPA / FixedAssets GetData frames -> line, desc, year, value."""
    cols = {c.lower(): c for c in df.columns}
    val = api.to_number(df[cols["datavalue"]])
    mult = pd.to_numeric(df[cols["unit_mult"]], errors="coerce").fillna(0) if "unit_mult" in cols else 0
    if to_billions:
        val = val * (10.0 ** mult) / 1e9
    else:
        val = val * (10.0 ** mult)
    tp = df[cols["timeperiod"]].astype(str)
    out = pd.DataFrame({
        "line": pd.to_numeric(df[cols["linenumber"]], errors="coerce"),
        "desc": df[cols["linedescription"]].astype(str).str.strip(),
        "year": pd.to_numeric(tp.str[:4], errors="coerce"),
        "is_annual": ~tp.str.contains("Q|M", regex=True),
        "value": val,
    })
    out = out[out["is_annual"]].drop(columns="is_annual").dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out["key"] = out["desc"].map(norm_name)
    return out


# BEA industry names that are split across lines in other BEA tables.
ALIASES = {
    norm_name("Federal Reserve banks, credit intermediation, and related activities"):
        [norm_name("Federal Reserve banks"), norm_name("Credit intermediation and related activities")],
    # BEA's 2023 comprehensive update renamed these in NIPA / Fixed Assets but not in GDP-by-industry
    norm_name("Publishing industries, except internet (includes software)"):
        [norm_name("Publishing industries (includes software)")],
    norm_name("Data processing, internet publishing, and other information services"):
        [norm_name("Information and data processing services")],
    # NIPA / Fixed Assets report only 'Real estate' (= housing + other real estate). Housing has no
    # employees and negligible IPP/equipment, so 'Real estate' is used for 'Other real estate'.
    norm_name("Other real estate"): [norm_name("Real estate")],
}
# aliases not applied for some measures (total fixed assets of 'Real estate' include rental housing)
ALIAS_SKIP = {("fa_stock", norm_name("Other real estate"))}


def match_lines(targets: pd.DataFrame, lines: pd.DataFrame, label: str) -> pd.DataFrame:
    """Map each panel industry (by name) onto line-item data (NIPA / Fixed Assets)."""
    first = lines.sort_values("line").drop_duplicates(["key", "year"])
    keys = set(first["key"])
    rows, unmatched = [], []
    for _, t in targets.iterrows():
        k = norm_name(t["name"])
        src = None
        if k in keys:
            src = [k]
        elif k in ALIASES and (label, k) not in ALIAS_SKIP and all(a in keys for a in ALIASES[k]):
            src = ALIASES[k]
        else:
            close = difflib.get_close_matches(k, list(keys), n=1, cutoff=0.88)
            if close:
                src = close
        if src is None:
            unmatched.append(t["name"])
            continue
        sub = first[first["key"].isin(src)].groupby("year", as_index=False)["value"].sum(min_count=1)
        sub["code"] = t["code"]
        rows.append(sub)
    log(f"  [{label}] matched {len(targets) - len(unmatched)}/{len(targets)} industries"
        + (f"; unmatched: {unmatched}" if unmatched else ""))
    if not rows:
        return pd.DataFrame(columns=["code", "year", label])
    return pd.concat(rows).rename(columns={"value": label})[["code", "year", label]]


# ------------------------------------------------------------------ industry selection
# Aggregates in BEA's GDP-by-industry hierarchy that do not start with a child's code.
EXPLICIT_AGGREGATES = {"31G", "33DG", "31ND", "44RT", "48TW", "531", "FIRE", "PROF", "6", "7"}
# Excluded by design: owner-occupied housing (imputed output, no employees) and government.
EXCLUDE_CODES = {"HS"}


def select_leaf_industries(names: pd.DataFrame) -> pd.DataFrame:
    """Most-detailed private industries (BEA ~71-industry level, minus government & housing)."""
    codes = names["code"].tolist()
    has_digit = [c for c in codes if re.search(r"\d", c)] + [c for c in codes if c in {"HS", "ORE"}]
    leaves = []
    for c in has_digit:
        if c in EXPLICIT_AGGREGATES or c in EXCLUDE_CODES:
            continue
        if any(o != c and o.startswith(c) for o in has_digit):   # parent of a more detailed code
            continue
        leaves.append(c)
    out = names[names["code"].isin(leaves)].copy()
    out = out[~out["code"].str.upper().str.startswith("G")]      # government
    return out.sort_values("code").reset_index(drop=True)


def sector_of(code: str) -> str:
    c = code.upper()
    if c in {"HS", "ORE"}:
        return "53"
    if c.startswith("4A"):
        return "44"
    m = re.match(r"(\d{2})", c)
    return m.group(1) if m else "NA"


# ------------------------------------------------------------------ BEA blocks
def fetch_gdp_by_industry(refresh=False) -> dict[str, pd.DataFrame]:
    tables = api.bea_param_values("GDPbyIndustry", "TableID", refresh)
    tables.to_csv(PROCESSED_DIR / "bea_gdpbyindustry_tables.csv", index=False)
    spec = {
        "va_nom": dict(include=["value added by industry"],
                       exclude=["percent", "real", "quantity", "price", "components", "contribution", "chain"]),
        "va_real": dict(include=["real value added by industry"], exclude=["percent", "contribution"]),
        "components": dict(include=["components of value added by industry"], exclude=["percent"]),
        "components_pct": dict(include=["components of value added by industry", "percent"], exclude=[]),
    }
    out = {}
    for name, s in spec.items():
        t = api.pick_table(tables, s["include"], s["exclude"], required=name in {"va_nom", "va_real"})
        if t is None:
            log(f"  GDPbyIndustry: no table for {name} (optional)")
            continue
        log(f"  GDPbyIndustry {name}: TableID {t['key']} - {t['desc']}")
        df = api.bea_get_data("GDPbyIndustry", f"gdpi_{t['key']}", refresh=refresh,
                              TableID=t["key"], Frequency="A", Industry="ALL", Year="ALL")
        out[name] = _std_gdpi(df)
    return out


def labour_share_from_components(comp: pd.DataFrame | None, comp_pct: pd.DataFrame | None,
                                 va: pd.DataFrame) -> pd.DataFrame:
    """Compensation of employees and gross operating surplus by industry-year."""
    def pick(df, word):
        d = df[df["desc"].str.lower().str.contains(word)]
        return d.groupby(["code", "year"], as_index=False)["value"].first()

    if comp is not None and comp["desc"].str.lower().str.contains("compensation").any():
        c = pick(comp, "compensation").rename(columns={"value": "comp"})
        g = pick(comp, "gross operating surplus").rename(columns={"value": "gos"})
        out = c.merge(g, on=["code", "year"], how="outer").merge(va, on=["code", "year"], how="left")
        out["labor_share"] = out["comp"] / out["va_nom"]
        out["gos_share"] = out["gos"] / out["va_nom"]
        log("  labour share computed from components-of-value-added levels")
        return out.drop(columns="va_nom")
    if comp_pct is not None and comp_pct["desc"].str.lower().str.contains("compensation").any():
        c = pick(comp_pct, "compensation").rename(columns={"value": "labor_share"})
        g = pick(comp_pct, "gross operating surplus").rename(columns={"value": "gos_share"})
        out = c.merge(g, on=["code", "year"], how="outer")
        out[["labor_share", "gos_share"]] /= 100.0
        out = out.merge(va, on=["code", "year"], how="left")
        out["comp"] = out["labor_share"] * out["va_nom"]
        out["gos"] = out["gos_share"] * out["va_nom"]
        log("  labour share computed from components-of-value-added percentages")
        return out.drop(columns="va_nom")
    sample = (comp if comp is not None else comp_pct)
    if sample is not None:
        sample.head(200).to_csv(PROCESSED_DIR / "DEBUG_components_sample.csv", index=False)
    raise ValueError("Could not find 'Compensation of employees' rows in the BEA components table. "
                     "A sample was saved to data/processed/DEBUG_components_sample.csv - share it "
                     "so the parser can be adapted.")


def fetch_nipa_fte(refresh=False) -> pd.DataFrame:
    tables = api.bea_param_values("NIPA", "TableName", refresh)
    tables.to_csv(PROCESSED_DIR / "bea_nipa_tables.csv", index=False)
    t = api.pick_table(tables, ["full-time equivalent employees by industry"], prefer_suffix="D")
    log(f"  NIPA FTE: {t['key']} - {t['desc']}")
    df = api.bea_get_data("NIPA", f"nipa_{t['key']}", refresh=refresh, TableName=t["key"], Frequency="A")
    return _std_lines(df, to_billions=False)        # persons


def fetch_fixed_assets(refresh=False) -> dict[str, pd.DataFrame]:
    tables = api.bea_param_values("FixedAssets", "TableName", refresh)
    tables.to_csv(PROCESSED_DIR / "bea_fixedassets_tables.csv", index=False)
    spec = {
        "ipp_stock": (["current-cost net stock", "intellectual property products", "by industry"], ["chain", "quantity"], True),
        "ipp_qty": (["quantity indexes for net stock", "intellectual property products", "by industry"], [], False),
        "ipp_inv": (["investment in private intellectual property products by industry"], ["chain", "quantity", "historical"], False),
        "eq_stock": (["current-cost net stock", "equipment", "by industry"], ["chain", "quantity", "intellectual", "structures"], False),
        "fa_stock": (["current-cost net stock of private fixed assets by industry"], ["chain", "quantity", "equipment", "structures", "intellectual"], False),
    }
    out = {}
    for name, (inc, exc, req) in spec.items():
        t = api.pick_table(tables, inc, exc, required=req)
        if t is None:
            log(f"  FixedAssets: no table for {name} (optional)")
            continue
        log(f"  FixedAssets {name}: {t['key']} - {t['desc']}")
        df = api.bea_get_data("FixedAssets", f"fa_{t['key']}", refresh=refresh, TableName=t["key"])
        out[name] = _std_lines(df, to_billions=(name != "ipp_qty"))
    return out


# ------------------------------------------------------------------ FRED / BLS / Census
FRED_SERIES = {
    "FEDFUNDS": "fedfunds",       # effective federal funds rate
    "DGS10": "dgs10",             # 10-year Treasury yield
    "BAA10Y": "baa_spread",       # Moody's Baa minus 10y Treasury (credit risk)
    "VIXCLS": "vix",              # equity volatility
    "NASDAQCOM": "nasdaq",        # technology-heavy equity index
    "PCEPI": "pcepi",             # PCE price index (consumer deflator)
    "ECOMPCTSA": "ecom_share",    # e-commerce share of retail sales (Census via FRED)
    "USREC": "recession",         # NBER recession indicator (share of months)
    "PRS85006173": "ls_nonfarm",  # BLS labour share, nonfarm business (via FRED)
}


def fetch_macro(refresh=False) -> pd.DataFrame:
    frames = []
    for sid, name in FRED_SERIES.items():
        try:
            d = api.fred_series(sid, refresh=refresh)
            d["year"] = d["date"].dt.year
            frames.append(d.groupby("year")[sid].mean().rename(name))
        except Exception as e:  # noqa: BLE001
            log(f"  FRED {sid} skipped: {e}")
    macro = pd.concat(frames, axis=1).reset_index()
    if "nasdaq" in macro:
        macro["nasdaq_ret"] = np.log(macro["nasdaq"]).diff()
    log(f"  FRED: {len(frames)}/{len(FRED_SERIES)} series, years {macro['year'].min()}-{macro['year'].max()}")
    return macro


BLS_SERIES = {
    "CEU5552000001": ("52", "employment"),       # finance & insurance, all employees (thousands)
    "CEU5552200001": ("522", "employment"),      # credit intermediation
    "CEU5552300001": ("523", "employment"),      # securities, commodity contracts, investments
    "CEU5552400001": ("524", "employment"),      # insurance carriers
    "CEU5552000003": ("52", "avg_hourly_earnings"),
    "CEU5552200003": ("522", "avg_hourly_earnings"),
    "CEU5552300003": ("523", "avg_hourly_earnings"),
    "CEU5552400003": ("524", "avg_hourly_earnings"),
    "PRS85006092": ("nonfarm", "labor_productivity"),
    "PRS85006173": ("nonfarm", "labor_share"),
}


def fetch_bls(start: int, end: int, refresh=False) -> pd.DataFrame:
    try:
        df = api.bls_series(list(BLS_SERIES), start, end, refresh)
    except Exception as e:  # noqa: BLE001
        log(f"  BLS skipped: {e}")
        return pd.DataFrame()
    df["naics"] = df["series_id"].map(lambda s: BLS_SERIES[s][0])
    df["measure"] = df["series_id"].map(lambda s: BLS_SERIES[s][1])
    log(f"  BLS: {df['series_id'].nunique()}/{len(BLS_SERIES)} series")
    return df


BFS_SECTOR = {"31": "NAICSMNF", "32": "NAICSMNF", "33": "NAICSMNF", "44": "NAICSRET", "45": "NAICSRET",
              "48": "NAICSTW", "49": "NAICSTW"}


def fetch_bfs(refresh=False) -> pd.DataFrame:
    try:
        df = api.census_bfs(2004, refresh)
    except Exception as e:  # noqa: BLE001
        log(f"  Census BFS skipped: {e}")
        return pd.DataFrame(columns=["bfs_cat", "year", "business_apps"])
    d = df[(df["data_type_code"] == "BA_BA") & (df["seasonally_adj"].str.lower() == "no")].copy()
    d["year"] = pd.to_numeric(d["time"].astype(str).str[:4], errors="coerce")
    out = (d.groupby(["category_code", "year"], as_index=False)["cell_value"].sum()
             .rename(columns={"category_code": "bfs_cat", "cell_value": "business_apps"}))
    # drop the incomplete current year
    counts = d.groupby("year")["time"].nunique()
    full = counts[counts >= 12].index
    out = out[out["year"].isin(full)]
    log(f"  Census BFS: {out['bfs_cat'].nunique()} categories, {out['year'].min()}-{out['year'].max()}")
    return out


# ------------------------------------------------------------------ assemble
def build_panel(refresh: bool = False, end_year: int | None = None) -> pd.DataFrame:
    LOG.clear()
    log(f"Pipeline run {date.today().isoformat()}")
    log("BEA GDP by industry ...")
    g = fetch_gdp_by_industry(refresh)

    names = (g["va_nom"].groupby("code")["desc"].agg(lambda s: s.mode().iloc[0]).reset_index()
             .rename(columns={"desc": "name"}))
    names.to_csv(PROCESSED_DIR / "bea_industry_codes_all.csv", index=False)
    leaves = select_leaf_industries(names)
    log(f"  selected {len(leaves)} leaf private industries "
        f"({leaves['code'].str.startswith(FINANCE_PREFIX).sum()} in finance & insurance)")

    va = g["va_nom"].groupby(["code", "year"], as_index=False)["value"].first().rename(columns={"value": "va_nom"})
    var = g["va_real"].groupby(["code", "year"], as_index=False)["value"].first().rename(columns={"value": "va_real"})
    shares = labour_share_from_components(g.get("components"), g.get("components_pct"), va)

    panel = (leaves.merge(va, on="code").merge(var, on=["code", "year"], how="left")
                   .merge(shares, on=["code", "year"], how="left"))

    log("BEA NIPA employment ...")
    fte = match_lines(leaves, fetch_nipa_fte(refresh), "fte")
    panel = panel.merge(fte, on=["code", "year"], how="left")

    log("BEA Fixed Assets ...")
    fa = fetch_fixed_assets(refresh)
    for name, lines in fa.items():
        panel = panel.merge(match_lines(leaves, lines, name), on=["code", "year"], how="left")

    log("FRED macro-financial series ...")
    macro = fetch_macro(refresh)

    log("Census Business Formation Statistics ...")
    bfs = fetch_bfs(refresh)

    # ---------------- variable construction
    panel = panel[panel["year"] >= START_YEAR]
    if end_year:
        panel = panel[panel["year"] <= end_year]
    panel["sector2"] = panel["code"].map(sector_of)
    panel["finance"] = panel["code"].str.startswith(FINANCE_PREFIX).astype(int)
    panel["information"] = (panel["sector2"] == "51").astype(int)
    panel["it_producing"] = panel["code"].isin(IT_PRODUCING_CODES).astype(int)

    panel = panel.merge(macro, on="year", how="left")
    panel["bfs_cat"] = panel["sector2"].map(lambda s: BFS_SECTOR.get(s, f"NAICS{s}"))
    panel = panel.merge(bfs, on=["bfs_cat", "year"], how="left")

    panel = add_derived(panel)
    panel["main_sample"] = (~panel["code"].isin(MAIN_SAMPLE_EXCLUDE)).astype(int)
    for c, why in MAIN_SAMPLE_EXCLUDE.items():
        log(f"  main-sample exclusion: {c} ({why})")
    panel = add_capital_per_worker(panel)
    panel = panel.sort_values(["code", "year"]).reset_index(drop=True)

    macro.to_csv(PROCESSED_DIR / "macro_annual.csv", index=False)
    bls = fetch_bls(START_YEAR, int(panel["year"].max()), refresh)
    if len(bls):
        bls.to_csv(PROCESSED_DIR / "bls_finance_checks.csv", index=False)
    panel.to_csv(PROCESSED_DIR / "panel.csv", index=False)
    write_data_dictionary()
    log(f"Panel saved: {panel['code'].nunique()} industries x {panel['year'].nunique()} years "
        f"({panel['year'].min()}-{panel['year'].max()}), {len(panel)} rows")
    (PROCESSED_DIR / "pipeline_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    return panel


def add_derived(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["code", "year"]).copy()
    g = p.groupby("code")
    p["ipp_intensity"] = p["ipp_stock"] / p["va_nom"]
    p["log_ipp_intensity"] = np.log(p["ipp_intensity"].where(p["ipp_intensity"] > 0))
    if "eq_stock" in p:
        p["eq_intensity"] = p["eq_stock"] / p["va_nom"]
        p["log_eq_intensity"] = np.log(p["eq_intensity"].where(p["eq_intensity"] > 0))
    if "fa_stock" in p:
        p["ipp_share_of_capital"] = p["ipp_stock"] / p["fa_stock"]
    if "ipp_inv" in p:
        p["ipp_inv_rate"] = p["ipp_inv"] / p["va_nom"]
    p["log_ipp_stock"] = np.log(p["ipp_stock"].where(p["ipp_stock"] > 0))
    if "ipp_qty" in p:
        p["log_ipp_qty"] = np.log(p["ipp_qty"].where(p["ipp_qty"] > 0))
    # productivity and pay per FTE (VA in billions -> dollars)
    p["productivity"] = p["va_real"] * 1e9 / p["fte"]
    p["pay_nominal"] = p["comp"] * 1e9 / p["fte"]
    if "pcepi" in p:
        base = p.loc[p["year"] == 2017, "pcepi"].mean()
        p["pay_real"] = p["pay_nominal"] / (p["pcepi"] / base)
    p["log_productivity"] = np.log(p["productivity"])
    p["log_pay_real"] = np.log(p.get("pay_real", p["pay_nominal"]))
    # productivity-pay gap, cumulative since each industry's first year
    for v in ["log_productivity", "log_pay_real"]:
        p[v + "_idx"] = p[v] - g[v].transform("first")
    p["prod_pay_gap"] = p["log_productivity_idx"] - p["log_pay_real_idx"]
    p["log_va_price"] = np.log(p["va_nom"] / p["va_real"])
    # growth / change variables
    p["d_labor_share"] = g["labor_share"].diff()
    p["d_log_ipp_intensity"] = g["log_ipp_intensity"].diff()
    p["g_ipp_stock"] = g["log_ipp_stock"].diff()
    p["g_productivity"] = g["log_productivity"].diff()
    p["va_share_of_total"] = p["va_nom"] / p.groupby("year")["va_nom"].transform("sum")
    p["post_break"] = (p["year"] >= BREAK_YEAR).astype(int)
    if "business_apps" in p:
        p["log_business_apps"] = np.log(p["business_apps"].where(p["business_apps"] > 0))
    return p


DATA_DICTIONARY = {
    "code": "BEA GDP-by-industry code", "name": "Industry name", "year": "Calendar year",
    "va_nom": "Value added, current $ billions (BEA GDPbyIndustry)",
    "va_real": "Real value added, chained (2017) $ billions (BEA GDPbyIndustry)",
    "comp": "Compensation of employees, $ billions", "gos": "Gross operating surplus, $ billions",
    "labor_share": "Compensation of employees / value added",
    "gos_share": "Gross operating surplus / value added",
    "fte": "Full-time-equivalent employees, persons (BEA NIPA Table 6.5D)",
    "ipp_stock": "Current-cost net stock of private intellectual property products, $ billions (BEA Fixed Assets)",
    "ipp_qty": "Chain-type quantity index, net stock of IPP (BEA Fixed Assets)",
    "ipp_inv": "Investment in private IPP, $ billions", "eq_stock": "Current-cost net stock of equipment, $ billions",
    "fa_stock": "Current-cost net stock of all private fixed assets, $ billions",
    "ipp_intensity": "IPP net stock / value added", "eq_intensity": "Equipment net stock / value added",
    "ipp_share_of_capital": "IPP stock / total fixed-asset stock", "ipp_inv_rate": "IPP investment / value added",
    "productivity": "Real value added per FTE (2017 $)", "pay_real": "Compensation per FTE deflated by PCE price index (2017 $)",
    "prod_pay_gap": "Cumulative log growth of productivity minus log growth of real pay since first year",
    "finance": "1 if NAICS 52 (finance & insurance)", "information": "1 if NAICS 51",
    "it_producing": "1 if IT-producing industry (robustness exclusion)", "post_break": f"1 if year >= {BREAK_YEAR}",
    "fedfunds": "Effective federal funds rate, annual average (FRED)", "dgs10": "10-year Treasury yield (FRED)",
    "baa_spread": "Moody's Baa - 10y Treasury spread (FRED)", "vix": "CBOE VIX annual average (FRED)",
    "nasdaq_ret": "Log change in NASDAQ Composite (FRED)", "ecom_share": "E-commerce % of retail sales (FRED/Census)",
    "recession": "Share of months in NBER recession", "ls_nonfarm": "BLS nonfarm business labour share index (FRED)",
    "business_apps": "Census business applications in the industry's sector (BFS)",
    "main_sample": "1 if in the main estimation sample (see config.MAIN_SAMPLE_EXCLUDE)",
    "log_ipp_pw": "log real IPP capital per FTE (quantity index / FTE) - main treatment, no shared denominator with labour share",
    "log_ipp_pw_nom": "log nominal IPP stock per FTE deflated by PCE prices",
    "log_ipp_inv_pw": "log IPP investment per FTE deflated by PCE prices",
    "log_eq_pw": "log equipment stock per FTE deflated by PCE prices (control)",
}


def write_data_dictionary():
    pd.DataFrame(DATA_DICTIONARY.items(), columns=["variable", "definition"]).to_csv(
        PROCESSED_DIR / "data_dictionary.csv", index=False)


def add_capital_per_worker(p: pd.DataFrame) -> pd.DataFrame:
    """Capital-deepening measures that do NOT share a denominator with the labour share.

    IPP / value added and compensation / value added both move when value added moves,
    which creates a spurious correlation (division bias). Capital per worker avoids this.
      log_ipp_pw     : log(real IPP net stock quantity index / FTE)    - main treatment
      log_ipp_pw_nom : log(nominal IPP stock / FTE / PCE price index)  - alternative
      log_ipp_inv_pw : log(nominal IPP investment / FTE / PCE)         - alternative (flow)
      log_eq_pw      : log(nominal equipment stock / FTE / PCE)        - control
    """
    p = p.copy()
    pce = p["pcepi"] / 100 if "pcepi" in p else 1.0
    def safe_log(x):
        return np.log(x.where(x > 0))
    if "ipp_qty" in p:
        p["log_ipp_pw"] = safe_log(p["ipp_qty"] / p["fte"])
    p["log_ipp_pw_nom"] = safe_log(p["ipp_stock"] * 1e9 / p["fte"] / pce)
    if "log_ipp_pw" not in p or p["log_ipp_pw"].notna().sum() == 0:
        p["log_ipp_pw"] = p["log_ipp_pw_nom"]
    if "ipp_inv" in p:
        p["log_ipp_inv_pw"] = safe_log(p["ipp_inv"] * 1e9 / p["fte"] / pce)
    if "eq_stock" in p:
        p["log_eq_pw"] = safe_log(p["eq_stock"] * 1e9 / p["fte"] / pce)
    return p


def load_panel(main_only: bool = True) -> pd.DataFrame:
    """Load the panel. By default drops industries in config.MAIN_SAMPLE_EXCLUDE."""
    f = PROCESSED_DIR / "panel.csv"
    if not f.exists():
        raise FileNotFoundError("data/processed/panel.csv not found - run 01_data_pipeline.ipynb first.")
    p = pd.read_csv(f, dtype={"code": str, "sector2": str})
    p = add_capital_per_worker(p)
    if main_only:
        p = p[~p["code"].isin(MAIN_SAMPLE_EXCLUDE)].reset_index(drop=True)
    return p
