"""Thin, cached, key-safe clients for the BEA, BLS, FRED and Census APIs.

* Every raw response is cached as JSON under data/raw/<source>/ so the pipeline
  can be re-run offline and results are reproducible.
* Keys are passed only as request parameters; any error message is scrubbed of
  key values before it is raised, so a traceback can never leak a key.
* Cached files never contain keys (only the response body is stored).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

from .config import RAW_DIR, all_secrets, get_key


class APIError(RuntimeError):
    pass


def _redact(text: str) -> str:
    for s in all_secrets():
        text = text.replace(s, "***")
    return text


def _request(method: str, url: str, *, params=None, json_body=None,
             retries: int = 3, timeout: int = 90) -> requests.Response:
    last = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.request(method, url, params=params, json=json_body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise APIError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = _redact(f"{type(e).__name__}: {e}")
            time.sleep(2 * (attempt + 1))
    raise APIError(f"Request to {url} failed after {retries} attempts: {last}")


def _cache_path(source: str, name: str) -> Path:
    d = RAW_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return d / f"{safe}.json"


def cached_json(source: str, name: str, fetch: Callable[[], object], refresh: bool = False):
    path = _cache_path(source, name)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def to_number(s: pd.Series) -> pd.Series:
    """BEA/BLS values come as strings with commas and footnote codes like (D)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


# =========================================================================== BEA
BEA_URL = "https://apps.bea.gov/api/data"


def _bea_error(js: dict):
    api = js.get("BEAAPI", {}) if isinstance(js, dict) else {}
    err = api.get("Error")
    res = api.get("Results")
    if not err and isinstance(res, dict):
        err = res.get("Error")
    if not err and isinstance(res, list):
        for r in res:
            if isinstance(r, dict) and r.get("Error"):
                err = r["Error"]
    return err


def bea_call(params: dict, cache_name: str, refresh: bool = False) -> dict:
    def fetch():
        key = get_key("BEA_API_KEY")
        r = _request("GET", BEA_URL, params={"UserID": key, "ResultFormat": "JSON", **params})
        js = r.json()
        err = _bea_error(js)
        if err:
            raise APIError("BEA API error: " + _redact(json.dumps(err)[:600]))
        return js
    return cached_json("bea", cache_name, fetch, refresh)


def bea_param_values(dataset: str, parameter: str, refresh: bool = False) -> pd.DataFrame:
    js = bea_call({"method": "GetParameterValues", "DataSetName": dataset,
                   "ParameterName": parameter}, f"params_{dataset}_{parameter}", refresh)
    res = js["BEAAPI"]["Results"]
    if isinstance(res, list):
        res = res[0]
    df = pd.DataFrame(res["ParamValue"])
    # standardise to columns: key, desc
    cols = {c.lower(): c for c in df.columns}
    key_col = cols.get("key") or cols.get("tablename") or df.columns[0]
    desc_col = cols.get("desc") or cols.get("description") or df.columns[-1]
    return df.rename(columns={key_col: "key", desc_col: "desc"})[["key", "desc"]]


def bea_records(js: dict) -> list[dict]:
    res = js["BEAAPI"]["Results"]
    if isinstance(res, list):
        out = []
        for r in res:
            out.extend(r.get("Data", []))
        return out
    return res.get("Data", [])


def bea_get_data(dataset: str, cache_name: str, refresh: bool = False, **params) -> pd.DataFrame:
    """GetData with a fallback for the 'all years' token (ALL vs X differs by dataset)."""
    last_err = None
    year_tokens = [params.pop("Year")] if "Year" in params else ["ALL", "X"]
    for yt in year_tokens:
        try:
            js = bea_call({"method": "GetData", "DataSetName": dataset, "Year": yt, **params},
                          f"{cache_name}_{yt}", refresh)
            return pd.DataFrame(bea_records(js))
        except APIError as e:
            last_err = e
    raise APIError(f"BEA {dataset} {params}: {last_err}")


def find_tables(tables: pd.DataFrame, include: Iterable[str], exclude: Iterable[str] = ()) -> pd.DataFrame:
    d = tables["desc"].astype(str).str.lower()
    mask = pd.Series(True, index=tables.index)
    for w in include:
        mask &= d.str.contains(w.lower(), regex=False)
    for w in exclude:
        mask &= ~d.str.contains(w.lower(), regex=False)
    out = tables[mask].copy()
    out["len"] = out["desc"].str.len()
    return out.sort_values("len").drop(columns="len")


def pick_table(tables: pd.DataFrame, include, exclude=(), required=True, prefer_suffix=None):
    m = find_tables(tables, include, exclude)
    if prefer_suffix is not None and len(m):
        pref = m[m["key"].astype(str).str.upper().str.endswith(prefer_suffix)]
        if len(pref):
            m = pref
    if m.empty:
        if required:
            raise APIError(f"No BEA table matches include={list(include)} exclude={list(exclude)}. "
                           "Inspect the saved table list in data/raw/bea/.")
        return None
    return m.iloc[0]


# =========================================================================== BLS
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def bls_series(series_ids: list[str], start: int, end: int, refresh: bool = False) -> pd.DataFrame:
    """Annual averages (period M13) where published, else annual (A01) values."""
    key = get_key("BLS_API_KEY", required=False)

    def post(ids, y0, y1, use_key):
        body = {"seriesid": ids, "startyear": str(y0), "endyear": str(y1), "annualaverage": True}
        if use_key:
            body["registrationkey"] = key
        js = _request("POST", BLS_URL, json_body=body).json()
        if js.get("status") != "REQUEST_SUCCEEDED":
            raise APIError("BLS: " + _redact(" | ".join(js.get("message", []))[:600]))
        return js

    # A registered key allows 20 years per request; without one the limit is 10 years.
    def cache_names(win):
        return [_cache_path("bls", "bls_" + hashlib.md5(",".join(series_ids[i:i + 25]).encode()).hexdigest()[:10]
                            + f"_{y0}_{min(y0 + win - 1, end)}")
                for i in range(0, len(series_ids), 25) for y0 in range(start, end + 1, win)]

    use_key = bool(key)
    window = None
    if not refresh and all(p.exists() for p in cache_names(20)):
        window = 20                            # fully cached from a keyed run
    elif not refresh and all(p.exists() for p in cache_names(10)):
        use_key, window = False, 10            # fully cached from a key-less run
    elif use_key:
        try:
            post(series_ids[:1], end, end, True)
        except APIError as e:
            if "key" in str(e).lower():
                print("  BLS key rejected - continuing without a key (lower limits: 10 years per request). "
                      "Check BLS_API_KEY in .env if you want the higher limits.")
                use_key = False
            else:
                raise
    if window is None:
        window = 20 if use_key else 10
    frames = []
    for i in range(0, len(series_ids), 25):
        ids = series_ids[i:i + 25]
        for y0 in range(start, end + 1, window):
            y1 = min(y0 + window - 1, end)

            def fetch(ids=ids, y0=y0, y1=y1):
                return post(ids, y0, y1, use_key)

            js = cached_json("bls", "bls_" + hashlib.md5(",".join(ids).encode()).hexdigest()[:10] + f"_{y0}_{y1}", fetch, refresh)
            for s in js["Results"]["series"]:
                for obs in s.get("data", []):
                    frames.append({"series_id": s["seriesID"], "year": int(obs["year"]),
                                   "period": obs["period"], "value": obs["value"]})
    df = pd.DataFrame(frames)
    if df.empty:
        return pd.DataFrame(columns=["series_id", "year", "value"])
    df["value"] = to_number(df["value"])
    ann = df[df["period"].isin(["M13", "A01", "Q05"])]
    # series without an annual average: average the sub-annual observations
    missing = set(df["series_id"]) - set(ann["series_id"])
    if missing:
        sub = (df[df["series_id"].isin(missing)].groupby(["series_id", "year"], as_index=False)["value"].mean())
        ann = pd.concat([ann, sub])
    return ann.groupby(["series_id", "year"], as_index=False)["value"].mean()


# =========================================================================== FRED
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fred_series(series_id: str, frequency: str = "a", aggregation: str = "avg",
                start: str = "1990-01-01", refresh: bool = False) -> pd.DataFrame:
    def fetch():
        key = get_key("FRED_API_KEY")
        p = {"series_id": series_id, "api_key": key, "file_type": "json",
             "observation_start": start, "frequency": frequency, "aggregation_method": aggregation}
        return _request("GET", FRED_URL, params=p).json()

    js = cached_json("fred", f"{series_id}_{frequency}_{aggregation}", fetch, refresh)
    df = pd.DataFrame(js.get("observations", []))
    if df.empty:
        return pd.DataFrame(columns=["date", series_id])
    df = df[["date", "value"]].rename(columns={"value": series_id})
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================================================================== Census
CENSUS_BFS_URL = "https://api.census.gov/data/timeseries/eits/bfs"


def census_bfs(start_year: int = 2004, refresh: bool = False) -> pd.DataFrame:
    """Business Formation Statistics (monthly business applications by NAICS sector)."""
    def fetch():
        key = get_key("CENSUS_API_KEY", required=False)
        p = {"get": "cell_value,data_type_code,category_code,time_slot_id,seasonally_adj",
             "for": "us:*", "time": f"from {start_year}"}
        if key:
            p["key"] = key
        return _request("GET", CENSUS_BFS_URL, params=p).json()

    rows = cached_json("census", f"bfs_from_{start_year}", fetch, refresh)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["cell_value"] = to_number(df["cell_value"])
    return df
