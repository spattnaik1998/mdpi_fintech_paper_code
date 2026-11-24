"""Project configuration: paths, study parameters and safe API-key loading.

API keys are read from a .env file and are NEVER printed, logged or written to disk.
The loader accepts the common formats people use in .env files:

    FRED_API_KEY=abc
    FRED_API_KEY = abc
    BEA API_KEY: abc          (spaces in names become underscores -> BEA_API_KEY)
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "exgap").exists():
            return p
    return Path.cwd()


ROOT = _find_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study design
START_YEAR = 1997           # BEA GDP-by-industry (NAICS) begins in 1997
END_YEAR = None             # None = latest year available
BREAK_YEAR = 2010           # fintech / smartphone era (H1b, H3 interaction)
COVID_YEARS = (2020, 2021)  # excluded in robustness checks
RANDOM_STATE = 42

# Industries treated as financial services (BEA GDP-by-industry codes, NAICS 52)
FINANCE_PREFIX = "52"
# Excluded from the MAIN estimation sample (kept in the panel, re-added in a robustness check).
# 525 Funds, trusts & other financial vehicles: value added is tiny and erratic (real > nominal in
# some years), giving labour shares above 3 and meaningless productivity figures.
MAIN_SAMPLE_EXCLUDE = {"525": "Funds, trusts & other financial vehicles: tiny/erratic value added"}
# IT-producing industries, excluded in a robustness check
IT_PRODUCING_CODES = {"334", "511", "512", "513", "514", "5415", "518", "519"}

# ---------------------------------------------------------------- key loading
_KEY_NAMES = ("BEA_API_KEY", "BLS_API_KEY", "FRED_API_KEY", "CENSUS_API_KEY")


def _normalise_name(name: str) -> str:
    name = name.strip().strip('"').strip("'").upper()
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        # split on the first '=' or ':' (whichever comes first)
        m = re.match(r"^([^=:]+)[=:](.*)$", line)
        if not m:
            continue
        name, value = _normalise_name(m.group(1)), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            out[name] = value
    return out


def find_env_file() -> Path | None:
    """Look for .env in the project root, then in its parent folders."""
    for p in [ROOT, *ROOT.parents][:4]:
        f = p / ".env"
        if f.exists():
            return f
    return None


_ENV_CACHE: dict[str, str] | None = None


def _env() -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        f = find_env_file()
        _ENV_CACHE = _parse_env_file(f) if f else {}
    return _ENV_CACHE


def get_key(name: str, required: bool = True) -> str | None:
    """Return an API key from the environment or .env, without ever displaying it."""
    name = _normalise_name(name)
    value = os.environ.get(name) or _env().get(name)
    if not value and required:
        raise KeyError(
            f"{name} not found. Add a line '{name}=<your key>' to the .env file "
            f"(searched: {find_env_file() or 'no .env found'})."
        )
    return value


def key_status() -> dict[str, bool]:
    """Which keys are available (True/False only - values are never returned)."""
    return {k: bool(get_key(k, required=False)) for k in _KEY_NAMES}


def all_secrets() -> list[str]:
    """Used internally to redact keys from any error message."""
    return [v for v in (get_key(k, required=False) for k in _KEY_NAMES) if v]
