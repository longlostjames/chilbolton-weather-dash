#!/usr/bin/env python3
"""
Chilbolton Surface Meteorology Explorer — OPeNDAP / THREDDS edition.

Replaces direct filesystem access with THREDDS catalog browsing and remote
file retrieval via the CEDA OPeNDAP server.

Usage
-----
    pip install siphon                  # in addition to existing requirements
    python app_thredds.py [--port 8050]

Authentication
--------------
CEDA now uses OAuth2 access tokens for scripted access (certificates are
deprecated).  Generate a token at https://services.ceda.ac.uk/cedasite/
myceda/account/ and set it in the environment before starting the app:

    export CEDA_TOKEN=<your_access_token>

Alternatively you can store it in a file and load it:

    export CEDA_TOKEN=$(cat ~/.ceda_token)

Caching
-------
Remote files are cached locally so that the overview (which loads many
months of data) is fast after the first load.  Set CACHE_DIR to override
the default temporary location:

    export CACHE_DIR=/path/to/writable/scratch

Notes on dataset availability on THREDDS
-----------------------------------------
The CEDA THREDDS server (http://dap.ceda.ac.uk) primarily serves
datasets archived in /badc.  Data held only in group workspaces (/gws)
may NOT be available there.  Adjust the catalog URL constants below once
you have confirmed availability, or contact the CEDA helpdesk.

    TRH  (/badc/ncas-cao/...)    → almost certainly available
    Pressure, Rain, Anem (/gws)  → check with CEDA; placeholders provided
"""

import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
import tempfile
import threading
import time

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import xarray as xr
from dash import Input, Output, Patch, State, callback, dcc, html
from plotly.subplots import make_subplots
from siphon.catalog import TDSCatalog

# ---------------------------------------------------------------------------
# THREDDS / OPeNDAP configuration
# ---------------------------------------------------------------------------

THREDDS_BASE = os.environ.get(
    "THREDDS_BASE",
    "http://dap.ceda.ac.uk/thredds",
)

# Catalog index URLs — one per instrument.
# Each must point to the *top-level* catalog for that instrument
# (i.e. the catalog whose sub-catalogs are year directories).
#
# Verify these URLs by browsing:
#   http://dap.ceda.ac.uk/thredds/catalog/catalog.html
#
PRESSURE_CATALOG = os.environ.get(
    "PRESSURE_CATALOG_URL",
    # NOTE: /gws data may not be on THREDDS — check and update this URL.
    f"{THREDDS_BASE}/catalog/badc/ncas-cao/data/ncas-pressure-1/",
    "ncas-pressure-1/data/long-term/level1a/catalog.xml",
)
TRH_CATALOG = os.environ.get(
    "TRH_CATALOG_URL",
    f"{THREDDS_BASE}/catalog/badc/ncas-cao/data/"
    "ncas-temperature-rh-1/20150415_longterm/v1.1/catalog.xml",
)
RAIN_CATALOG = os.environ.get(
    "RAIN_CATALOG_URL",
    # NOTE: /gws data may not be on THREDDS — check and update this URL.
    f"{THREDDS_BASE}/catalog/gws/pw/j07/ncas_obs_vol2/cao/processing/"
    "ncas-rain-gauge-1/data/long-term/catalog.xml",
)
ANEM_CATALOG = os.environ.get(
    "ANEM_CATALOG_URL",
    # NOTE: /gws data may not be on THREDDS — check and update this URL.
    f"{THREDDS_BASE}/catalog/gws/pw/j07/ncas_obs_vol2/cao/processing/"
    "ncas-anemometer-2/data/long-term/level1a/catalog.xml",
)

# ---------------------------------------------------------------------------
# Optional authentication
# ---------------------------------------------------------------------------

# Token resolution priority (highest → lowest):
#   1. CEDA_TOKEN env var           (explicit override)
#   2. ~/.cedatoken cache file      (written by CEDA's own scripts)
#   3. CEDA_USERNAME + CEDA_PASSWORD env vars → fetch from token API
#   4. No token (open-access datasets only)

_TOKEN_API = "https://services-beta.ceda.ac.uk/api/token/create/"
_TOKEN_CACHE = os.path.expanduser("~/.cedatoken")


def _load_cached_token() -> str | None:
    """Return a non-expired token from ~/.cedatoken, or None."""
    try:
        with open(_TOKEN_CACHE) as fh:
            data = json.loads(fh.read())
        token = data.get("access_token")
        expires_str = data.get("expires", "")
        if token and expires_str:
            from datetime import timezone
            expires = datetime.datetime.strptime(expires_str, "%Y-%m-%dT%H:%M:%S.%f%z")
            if expires > datetime.datetime.now(tz=timezone.utc):
                return token
    except Exception:
        pass
    return None


def _fetch_token_from_api(username: str, password: str) -> str | None:
    """Fetch a fresh token from the CEDA token API and cache it."""
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    try:
        resp = requests.post(
            _TOKEN_API,
            headers={"Authorization": f"Basic {credentials}"},
            timeout=15,
        )
        if resp.status_code == 200:
            # Cache the full JSON response for future use (including expiry)
            with open(_TOKEN_CACHE, "w") as fh:
                fh.write(resp.text)
            return resp.json().get("access_token")
        else:
            print(f"[auth] token API returned {resp.status_code}")
    except Exception as exc:
        print(f"[auth] token fetch failed: {exc}")
    return None


def _resolve_token() -> str:
    """Return the best available Bearer token, or empty string."""
    # 1. Explicit env override
    tok = os.environ.get("CEDA_TOKEN", "")
    if tok:
        return tok
    # 2. Cached token (e.g. written by remote_nc_with_token.py)
    tok = _load_cached_token()
    if tok:
        print("[auth] Using cached token from ~/.cedatoken")
        return tok
    # 3. Fetch via credentials from env
    user = os.environ.get("CEDA_USERNAME", "")
    pw = os.environ.get("CEDA_PASSWORD", "")
    if user and pw:
        print("[auth] Fetching token from CEDA API...")
        tok = _fetch_token_from_api(user, pw)
        if tok:
            print("[auth] Token obtained and cached.")
            return tok
    return ""


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a shared requests.Session with a Bearer token if one is available."""
    global _session
    if _session is None:
        _session = requests.Session()
        token = _resolve_token()
        if token:
            _session.headers["Authorization"] = f"Bearer {token}"
        else:
            print("[auth] No CEDA token found — only open-access data will be reachable.")
    return _session


# ---------------------------------------------------------------------------
# Local disk cache for downloaded NetCDF files
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get(
    "CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "chilbolton_thredds_cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(url: str) -> str:
    """Return the local cache file path for a given remote URL."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    # Keep the original filename suffix so xarray picks the right engine
    suffix = os.path.splitext(url.split("?")[0])[-1] or ".nc"
    return os.path.join(CACHE_DIR, digest + suffix)


def _file_server_url(opendap_url: str) -> str:
    """Convert an OPeNDAP /dodsC/ URL to its HTTP /fileServer/ equivalent."""
    return opendap_url.replace("/dodsC/", "/fileServer/", 1)


def _download_to_cache(opendap_url: str) -> str | None:
    """
    Download the file at *opendap_url* (via the HTTP fileServer endpoint)
    and store it in the local cache.  Returns the local path, or None on
    failure.
    """
    local = _cache_path(opendap_url)
    if os.path.exists(local):
        return local

    dl_url = _file_server_url(opendap_url)
    try:
        sess = _get_session()
        resp = sess.get(dl_url, timeout=60, stream=True)
        resp.raise_for_status()
        tmp = local + ".tmp"
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        os.replace(tmp, local)
        return local
    except Exception as exc:
        print(f"[cache] download failed for {dl_url}: {exc}")
        if os.path.exists(local + ".tmp"):
            os.remove(local + ".tmp")
        return None


# ---------------------------------------------------------------------------
# THREDDS catalog helpers
# ---------------------------------------------------------------------------

# In-memory catalog cache: url → (fetched_at, TDSCatalog)
_catalog_cache: dict[str, tuple[float, TDSCatalog]] = {}
_catalog_lock = threading.Lock()
CATALOG_TTL = 300  # seconds — re-fetch catalog XML after 5 minutes


def _open_catalog(url: str) -> TDSCatalog | None:
    """Open a THREDDS catalog, caching the result for CATALOG_TTL seconds."""
    now = time.monotonic()
    with _catalog_lock:
        entry = _catalog_cache.get(url)
        if entry is not None and (now - entry[0]) < CATALOG_TTL:
            return entry[1]
    try:
        cat = TDSCatalog(url)
        with _catalog_lock:
            _catalog_cache[url] = (now, cat)
        return cat
    except Exception as exc:
        print(f"[catalog] cannot open {url}: {exc}")
        return None


def _navigate_catalog(base_url: str, year: int, month: int | None = None) -> TDSCatalog | None:
    """
    Navigate from a top-level catalog down to the year (and optionally the
    year-month) sub-catalog.

    Expected hierarchy
    ------------------
    Monthly layout (pressure, rain):
        top → <year>/ → <yearmonth>/ → *.nc files

    Yearly layout (TRH, anem):
        top → <year>/ → *.nc files
    """
    cat = _open_catalog(base_url)
    if cat is None:
        return None

    # Find the year sub-catalog
    year_str = str(year)
    year_ref = next(
        (ref for key, ref in cat.catalog_refs.items() if year_str in key),
        None,
    )
    if year_ref is None:
        return None

    year_cat = _open_catalog(year_ref.href)
    if year_cat is None:
        return None

    if month is None:
        return year_cat

    # Find the year-month sub-catalog
    month_str = f"{year}{month:02d}"
    month_ref = next(
        (ref for key, ref in year_cat.catalog_refs.items() if month_str in key),
        None,
    )
    if month_ref is None:
        return None

    return _open_catalog(month_ref.href)


def _opendap_urls_for_day(
    base_catalog_url: str,
    year: int,
    month: int,
    day: int,
    layout: str = "monthly",
) -> list[str]:
    """
    Return sorted OPeNDAP URLs for NetCDF files matching a specific date.

    Parameters
    ----------
    layout : "monthly" | "yearly"
        "monthly" means files sit in <year>/<yearmonth>/ sub-directories.
        "yearly"  means files sit directly in <year>/.
    """
    if layout == "monthly":
        cat = _navigate_catalog(base_catalog_url, year, month)
    else:
        cat = _navigate_catalog(base_catalog_url, year)

    if cat is None:
        return []

    date_str = f"{year}{month:02d}{day:02d}"
    urls = []
    for name, ds in cat.datasets.items():
        if date_str in name and name.lower().endswith(".nc"):
            url = ds.access_urls.get("OPENDAP") or ds.access_urls.get("dods")
            if url:
                urls.append(url)

    return sorted(urls)


def _opendap_urls_for_month(
    base_catalog_url: str,
    year: int,
    month: int,
    layout: str = "monthly",
) -> list[str]:
    """
    Return sorted OPeNDAP URLs for all NetCDF files in a given month.
    """
    if layout == "monthly":
        cat = _navigate_catalog(base_catalog_url, year, month)
    else:
        # For yearly layout, filter by month prefix
        cat = _navigate_catalog(base_catalog_url, year)

    if cat is None:
        return []

    month_str = f"{year}{month:02d}"
    urls = []
    for name, ds in cat.datasets.items():
        if name.lower().endswith(".nc"):
            if layout == "monthly" or month_str in name:
                url = ds.access_urls.get("OPENDAP") or ds.access_urls.get("dods")
                if url:
                    urls.append(url)

    return sorted(urls)


# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

VARIABLES = {
    "air_pressure": {
        "label": "Air Pressure",
        "var": "air_pressure",
        "qc": "qc_flag_air_pressure",
        "ylabel": "Air pressure (hPa)",
        "source": "pressure",
    },
    "air_temperature": {
        "label": "Air Temperature",
        "var": "air_temperature",
        "qc": "qc_flag_air_temperature",
        "ylabel": "Air temperature (\u00b0C)",
        "source": "trh",
    },
    "relative_humidity": {
        "label": "Relative Humidity",
        "var": "relative_humidity",
        "qc": "qc_flag_relative_humidity",
        "ylabel": "Relative humidity (%)",
        "source": "trh",
    },
    "rainfall_rate": {
        "label": "Rainfall Rate",
        "var": "rainfall_rate",
        "qc": "qc_flag",
        "ylabel": "Rainfall rate (mm hr\u207b\u00b9)",
        "source": "rain",
    },
    "wind_speed": {
        "label": "Wind Speed",
        "var": "wind_speed",
        "qc": None,
        "ylabel": "Wind speed (m s\u207b\u00b9)",
        "source": "anem",
    },
    "wind_from_direction": {
        "label": "Wind Direction",
        "var": "wind_from_direction",
        "qc": None,
        "ylabel": "Wind direction (\u00b0)",
        "source": "anem",
    },
}

VAR_ORDER = [
    "air_pressure",
    "air_temperature",
    "relative_humidity",
    "rainfall_rate",
    "wind_speed",
    "wind_from_direction",
]

# ---------------------------------------------------------------------------
# Year / month discovery via catalog
# ---------------------------------------------------------------------------


def available_years() -> list[int]:
    """Query the pressure catalog for available year sub-catalogs."""
    cat = _open_catalog(PRESSURE_CATALOG)
    if cat is None:
        return []
    years = []
    for key in cat.catalog_refs:
        try:
            years.append(int(key.strip("/")))
        except ValueError:
            pass
    return sorted(years)


def available_months(year: int) -> list[int]:
    """Query the pressure catalog for available month sub-catalogs in *year*."""
    year_cat = _navigate_catalog(PRESSURE_CATALOG, year)
    if year_cat is None:
        return []
    months = []
    for key in year_cat.catalog_refs:
        try:
            val = int(key.strip("/"))
            if val > 100:          # yearmonth like 202301 → strip year
                val = val % 100
            months.append(val)
        except ValueError:
            pass
    return sorted(months)


# ---------------------------------------------------------------------------
# Per-instrument file openers (accept a local path or URL)
# ---------------------------------------------------------------------------

def _open_nc(path_or_url: str) -> xr.Dataset | None:
    """Open a pressure NetCDF file and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path_or_url)
        if "time" not in ds or ds.time.size == 0 or "air_pressure" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        return ds
    except Exception:
        return None


def _open_nc_trh(path_or_url: str) -> xr.Dataset | None:
    """Open a temp/RH NetCDF file and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path_or_url)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "air_temperature" not in ds and "relative_humidity" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        if "air_temperature" in ds:
            if (
                ds["air_temperature"].attrs.get("units", "").upper() in ("K", "KELVIN")
                or float(ds["air_temperature"].values.mean()) > 200
            ):
                ds["air_temperature"].values[:] -= 273.15
                ds["air_temperature"].attrs["units"] = "degC"
        return ds
    except Exception:
        return None


def _open_nc_rain(path_or_url: str) -> xr.Dataset | None:
    """Open a rain gauge NetCDF, compute rainfall_rate (mm/hr), return Dataset or None."""
    try:
        ds = xr.open_dataset(path_or_url)
        if "time" not in ds or ds.time.size == 0 or "thickness_of_rainfall_amount" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        times = pd.to_datetime(ds["time"].values)
        dt_s = pd.Series(times).diff().dt.total_seconds().fillna(10).values
        rate = ds["thickness_of_rainfall_amount"].values / dt_s * 3600.0
        qc = ds["qc_flag"].values if "qc_flag" in ds else None
        result = xr.Dataset(
            {
                "rainfall_rate": (
                    "time",
                    rate.astype("float32"),
                    {"long_name": "Rainfall rate", "units": "mm hr-1"},
                ),
            },
            coords={"time": ds["time"]},
        )
        if qc is not None:
            result["qc_flag"] = ("time", qc)
        return result
    except Exception:
        return None


def _open_nc_anem(path_or_url: str) -> xr.Dataset | None:
    """Open an anemometer NetCDF and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path_or_url)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "wind_speed" not in ds and "wind_from_direction" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        squeeze_dims = [
            d for d in ["latitude", "longitude"] if d in ds.sizes and ds.sizes[d] == 1
        ]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims, drop=True)
        return ds
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generic "open via cache" wrapper
# ---------------------------------------------------------------------------

def _cached_open(opendap_url: str, open_fn):
    """
    Download *opendap_url* to the local cache (if not already present) then
    call *open_fn* on the local file.
    """
    local = _download_to_cache(opendap_url)
    if local is None:
        # Fall back to opening directly via OPeNDAP
        return open_fn(opendap_url)
    return open_fn(local)


# ---------------------------------------------------------------------------
# Day-level loaders
# ---------------------------------------------------------------------------

def load_day(year: int, month: int, day: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_day(PRESSURE_CATALOG, year, month, day, layout="monthly")
    for url in urls:
        ds = _cached_open(url, _open_nc)
        if ds is not None:
            return ds
    return None


def load_day_trh(year: int, month: int, day: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_day(TRH_CATALOG, year, month, day, layout="yearly")
    for url in urls:
        ds = _cached_open(url, _open_nc_trh)
        if ds is not None:
            return ds
    return None


def load_day_rain(year: int, month: int, day: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_day(RAIN_CATALOG, year, month, day, layout="monthly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc_rain)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]


def load_day_anem(year: int, month: int, day: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_day(ANEM_CATALOG, year, month, day, layout="yearly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc_anem)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time").sortby("time") if len(datasets) > 1 else datasets[0]


# ---------------------------------------------------------------------------
# Month-level loaders (used by date-range loading)
# ---------------------------------------------------------------------------

def _load_month_pressure(year: int, month: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_month(PRESSURE_CATALOG, year, month, layout="monthly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time")


def _load_month_trh(year: int, month: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_month(TRH_CATALOG, year, month, layout="yearly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc_trh)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time")


def _load_month_rain(year: int, month: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_month(RAIN_CATALOG, year, month, layout="monthly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc_rain)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time")


def _load_month_anem(year: int, month: int) -> xr.Dataset | None:
    urls = _opendap_urls_for_month(ANEM_CATALOG, year, month, layout="yearly")
    datasets = [ds for url in urls if (ds := _cached_open(url, _open_nc_anem)) is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time").sortby("time")


# ---------------------------------------------------------------------------
# Date-range loaders
# ---------------------------------------------------------------------------

def _iter_months(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        yield cur.year, cur.month
        cur += pd.DateOffset(months=1)


def _slice_combined(ds, start, end):
    return ds.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )


def load_date_range(start_date, end_date) -> xr.Dataset | None:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    datasets = [ds for y, m in _iter_months(start, end)
                if (ds := _load_month_pressure(y, m)) is not None]
    if not datasets:
        return None
    return _slice_combined(xr.concat(datasets, dim="time"), start, end)


def load_date_range_trh(start_date, end_date) -> xr.Dataset | None:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    datasets = [ds for y, m in _iter_months(start, end)
                if (ds := _load_month_trh(y, m)) is not None]
    if not datasets:
        return None
    return _slice_combined(xr.concat(datasets, dim="time"), start, end)


def load_date_range_rain(start_date, end_date) -> xr.Dataset | None:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    datasets = [ds for y, m in _iter_months(start, end)
                if (ds := _load_month_rain(y, m)) is not None]
    if not datasets:
        return None
    return _slice_combined(xr.concat(datasets, dim="time"), start, end)


def load_date_range_anem(start_date, end_date) -> xr.Dataset | None:
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    datasets = [ds for y, m in _iter_months(start, end)
                if (ds := _load_month_anem(y, m)) is not None]
    if not datasets:
        return None
    return _slice_combined(xr.concat(datasets, dim="time").sortby("time"), start, end)


# ---------------------------------------------------------------------------
# Download dataset builder (preserves original file metadata)
# ---------------------------------------------------------------------------

MAX_OVERVIEW_POINTS = 50_000


def _build_download_dataset(source: str, start, end) -> xr.Dataset | None:
    """
    Re-fetch raw files for *source* and return a Dataset with full metadata.
    Files are retrieved from cache (already warm after browsing) where possible.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.date() == end.date() and start.time() == end.time() == datetime.time(0):
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    catalog_map = {
        "pressure": (PRESSURE_CATALOG, "monthly", _open_nc),
        "trh":      (TRH_CATALOG,      "yearly",  _open_nc_trh),
        "rain":     (RAIN_CATALOG,      "monthly", _open_nc_rain),
        "anem":     (ANEM_CATALOG,      "yearly",  _open_nc_anem),
    }
    catalog_url, layout, open_fn = catalog_map[source]

    datasets = []
    for year, month in _iter_months(start, end):
        urls = _opendap_urls_for_month(catalog_url, year, month, layout=layout)
        for url in urls:
            ds = _cached_open(url, open_fn)
            if ds is not None:
                datasets.append(ds)

    if not datasets:
        return None

    combined = xr.concat(datasets, dim="time", combine_attrs="override").sortby("time")
    return combined.sel(time=slice(start, end))


# ---------------------------------------------------------------------------
# Plotting helpers  (identical to app.py — no filesystem dependency)
# ---------------------------------------------------------------------------

def _thin_arrays(times, pressure, qc):
    n = len(times)
    if n <= MAX_OVERVIEW_POINTS:
        return times, pressure, qc
    step = max(1, n // MAX_OVERVIEW_POINTS)
    idx = np.arange(0, n, step)
    return times[idx], pressure[idx], (qc[idx] if qc is not None else None)


def _empty_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def _make_multi_fig(title: str):
    n = len(VAR_ORDER)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[VARIABLES[k]["label"] for k in VAR_ORDER],
        vertical_spacing=0.06,
    )
    for i, key in enumerate(VAR_ORDER, start=1):
        if key == "wind_from_direction":
            fig.update_yaxes(
                title_text=VARIABLES[key]["ylabel"],
                tickvals=[0, 90, 180, 270, 360],
                ticktext=["N", "E", "S", "W", "N"],
                range=[0, 360],
                row=i,
                col=1,
            )
        else:
            fig.update_yaxes(title_text=VARIABLES[key]["ylabel"], row=i, col=1)
    fig.update_xaxes(title_text="Date / Time (UTC)", row=n, col=1)
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=80, r=20, t=80, b=50),
        hovermode="x unified",
        clickmode="event+select",
    )
    return fig


def _add_var_traces(fig, row, times, values, qc, use_gl=False, use_lines=False):
    Cls = go.Scattergl if use_gl else go.Scatter
    mode = "lines+markers" if use_lines else "markers"
    msize = 2 if use_gl else 3
    if qc is not None:
        good = qc == 1
        bad_mask = qc == 2
        fig.add_trace(
            Cls(
                x=times[good],
                y=values[good],
                mode=mode,
                marker=dict(color="steelblue", size=msize),
                line=dict(color="steelblue", width=1) if use_lines else {},
                name="Good (flag=1)",
                legendgroup="good",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        if bad_mask.any():
            fig.add_trace(
                Cls(
                    x=times[bad_mask],
                    y=values[bad_mask],
                    mode="markers",
                    marker=dict(
                        color="crimson",
                        size=4 if use_gl else 5,
                        symbol="circle" if use_gl else "x",
                    ),
                    name="Bad (flag=2)",
                    legendgroup="bad",
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )
    else:
        fig.add_trace(
            Cls(
                x=times,
                y=values,
                mode=mode,
                marker=dict(color="steelblue", size=msize),
                line=dict(color="steelblue", width=1) if use_lines else {},
                name="Data",
                legendgroup="good",
                showlegend=False,
            ),
            row=row,
            col=1,
        )


def _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date):
    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    fig = _make_multi_fig(f"Detail \u2014 {date}")
    source_map = {
        "pressure": ds_pressure,
        "trh": ds_trh,
        "rain": ds_rain,
        "anem": ds_anem,
    }
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        ds = source_map[cfg["source"]]
        if ds is None or cfg["var"] not in ds:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        _add_var_traces(fig, i, times, values, qc, use_gl=False)
    fig.update_xaxes(range=[day_start, day_end])
    return fig


def _build_multi_range_fig(ds_pressure, ds_trh, ds_rain, ds_anem, start, end):
    duration = end - start
    use_lines = duration < pd.Timedelta(hours=1)
    title = (
        f"Detail \u2014 {start.date()} to {end.date()}"
        if duration >= pd.Timedelta(days=1)
        else f"Detail \u2014 {start}"
    )
    fig = _make_multi_fig(title)
    source_map = {
        "pressure": ds_pressure,
        "trh": ds_trh,
        "rain": ds_rain,
        "anem": ds_anem,
    }
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        ds = source_map[cfg["source"]]
        if ds is None or cfg["var"] not in ds or ds.time.size == 0:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        _add_var_traces(fig, i, times, values, qc, use_gl=not use_lines, use_lines=use_lines)
    return fig


# ---------------------------------------------------------------------------
# App layout  (identical to app.py)
# ---------------------------------------------------------------------------

default_end = datetime.date.today().isoformat()
default_start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()

app = dash.Dash(__name__, title="Chilbolton Surface Met (THREDDS)")

app.layout = html.Div(
    [
        html.H2(
            "Chilbolton Surface Meteorology Explorer",
            style={"fontFamily": "sans-serif", "margin": "16px 16px 8px"},
        ),
        html.Div(
            [
                html.Label("Start:", style={"marginRight": "6px"}),
                dcc.Input(
                    id="start-date-input",
                    type="date",
                    value=default_start,
                    debounce=False,
                    style={"marginRight": "12px"},
                ),
                html.Label("End:", style={"marginRight": "6px"}),
                dcc.Input(
                    id="end-date-input",
                    type="date",
                    value=default_end,
                    debounce=False,
                ),
                html.Button(
                    "Load",
                    id="load-btn",
                    n_clicks=0,
                    style={"marginLeft": "12px", "padding": "6px 18px",
                           "fontSize": "0.95em", "cursor": "pointer"},
                ),
                html.Span(
                    "Y-axis scale:",
                    style={"marginLeft": "24px", "marginRight": "8px", "fontFamily": "sans-serif"},
                ),
                dcc.RadioItems(
                    id="autorange-radio",
                    options=[
                        {"label": "Auto", "value": "auto"},
                        {"label": "Fixed", "value": "fixed"},
                    ],
                    value="fixed",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px", "fontFamily": "sans-serif"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "fontFamily": "sans-serif",
                "margin": "0 16px 8px",
            },
        ),
        html.Div(
            id="load-status",
            style={"fontFamily": "monospace", "fontSize": "0.85em",
                   "color": "#555", "margin": "0 16px 6px", "minHeight": "1.2em"},
        ),
        html.Div(
            [
                # Overview
                html.Div(
                    [
                        html.H3("Overview",
                                style={"fontFamily": "sans-serif", "margin": "8px 0 4px"}),
                        html.P(
                            "Drag to zoom; click a point to jump to that day. "
                            "Files are fetched from THREDDS and cached locally.",
                            style={"fontFamily": "sans-serif", "color": "#666",
                                   "margin": "0 0 4px", "fontSize": "0.9em"},
                        ),
                        dcc.Loading(
                            id="overview-loading",
                            type="circle",
                            children=dcc.Graph(id="overview-graph", style={"height": "900px"}),
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0", "padding": "0 8px 0 0"},
                ),
                # Detail
                html.Div(
                    [
                        html.H3("Detail view",
                                style={"fontFamily": "sans-serif", "margin": "8px 0 4px"}),
                        html.Div(
                            [
                                html.Label("Jump to date:",
                                           style={"marginRight": "6px", "fontFamily": "sans-serif"}),
                                dcc.DatePickerSingle(id="day-picker", display_format="YYYY-MM-DD"),
                                html.Button(
                                    "Download NetCDF (zip)",
                                    id="download-btn",
                                    n_clicks=0,
                                    style={"marginLeft": "16px", "padding": "5px 14px",
                                           "fontSize": "0.9em", "cursor": "pointer"},
                                ),
                            ],
                            style={"marginBottom": "4px", "display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            id="detail-status",
                            style={"fontFamily": "monospace", "fontSize": "0.85em",
                                   "color": "#555", "marginBottom": "4px", "minHeight": "1.2em"},
                        ),
                        dcc.Loading(
                            id="detail-loading",
                            type="circle",
                            children=dcc.Graph(id="detail-graph", style={"height": "900px"}),
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0", "padding": "0 0 0 8px"},
                ),
            ],
            style={"display": "flex", "alignItems": "flex-start", "margin": "0 16px"},
        ),
        dcc.Store(id="detail-range-store"),
        dcc.Download(id="download-nc"),
    ],
    style={"maxWidth": "1300px", "margin": "0 auto"},
)

# ---------------------------------------------------------------------------
# Callbacks  (identical logic to app.py)
# ---------------------------------------------------------------------------


@callback(
    Output("overview-graph", "figure"),
    Output("day-picker", "min_date_allowed"),
    Output("day-picker", "max_date_allowed"),
    Output("day-picker", "date"),
    Output("load-status", "children"),
    Input("load-btn", "n_clicks"),
    State("start-date-input", "value"),
    State("end-date-input", "value"),
)
def update_overview(n_clicks, start_date, end_date):
    no_dates = (None, None, None)
    if not start_date or not end_date:
        return _empty_fig(), *no_dates, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            "pressure": pool.submit(load_date_range, start_date, end_date),
            "trh":      pool.submit(load_date_range_trh, start_date, end_date),
            "rain":     pool.submit(load_date_range_rain, start_date, end_date),
            "anem":     pool.submit(load_date_range_anem, start_date, end_date),
        }
    ds_pressure = futs["pressure"].result()
    ds_trh      = futs["trh"].result()
    ds_rain     = futs["rain"].result()
    ds_anem     = futs["anem"].result()

    source_status = {
        "Pressure": ds_pressure,
        "Temp/RH":  ds_trh,
        "Rain":     ds_rain,
        "Anem":     ds_anem,
    }
    status_parts = [
        "{}: {}".format(name, "\u2714" if ds is not None else "\u2718 no data")
        for name, ds in source_status.items()
    ]
    status_msg = "  |  ".join(status_parts)

    if all(ds is None for ds in (ds_pressure, ds_trh, ds_rain, ds_anem)):
        fig = _empty_fig()
        fig.update_layout(title=f"No data for {start_date} \u2013 {end_date}")
        return fig, *no_dates, status_msg

    start_label = pd.Timestamp(start_date).strftime("%d %b %Y")
    end_label = pd.Timestamp(end_date).strftime("%d %b %Y")
    fig = _make_multi_fig(f"Overview \u2014 {start_label} to {end_label}")

    source_map = {
        "pressure": ds_pressure,
        "trh": ds_trh,
        "rain": ds_rain,
        "anem": ds_anem,
    }
    all_min, all_max = [], []
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        ds = source_map[cfg["source"]]
        if ds is None or cfg["var"] not in ds:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        all_min.append(times.min())
        all_max.append(times.max())
        times, values, qc = _thin_arrays(times, values, qc)
        _add_var_traces(fig, i, times, values, qc, use_gl=True)

    if not all_min:
        return fig, *no_dates, status_msg

    min_date = min(all_min).date().isoformat()
    max_date = max(all_max).date().isoformat()
    return fig, min_date, max_date, max_date, status_msg


@callback(
    Output("detail-graph", "figure"),
    Output("day-picker", "date", allow_duplicate=True),
    Output("detail-range-store", "data"),
    Output("detail-status", "children"),
    Input("overview-graph", "relayoutData"),
    Input("overview-graph", "clickData"),
    Input("day-picker", "date"),
    State("start-date-input", "value"),
    State("end-date-input", "value"),
    prevent_initial_call=True,
)
def update_detail(relayout_data, click_data, date_str, start_date, end_date):
    triggered_prop = dash.ctx.triggered[0]["prop_id"] if dash.ctx.triggered else None

    def _load_day_parallel(y, m, d):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fp = pool.submit(load_day, y, m, d)
            ft = pool.submit(load_day_trh, y, m, d)
            fr = pool.submit(load_day_rain, y, m, d)
            fa = pool.submit(load_day_anem, y, m, d)
        return fp.result(), ft.result(), fr.result(), fa.result()

    def _load_range_parallel(start, end):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fp = pool.submit(load_date_range, start, end)
            ft = pool.submit(load_date_range_trh, start, end)
            fr = pool.submit(load_date_range_rain, start, end)
            fa = pool.submit(load_date_range_anem, start, end)
        return fp.result(), ft.result(), fr.result(), fa.result()

    def _detail_status(ds_p, ds_t, ds_r, ds_a, label):
        _ok = "\u2714"
        _fail = "\u2718"
        parts = [
            "Pressure: " + (_ok if ds_p is not None else _fail),
            "Temp/RH: " + (_ok if ds_t is not None else _fail),
            "Rain: " + (_ok if ds_r is not None else _fail),
            "Anem: " + (_ok if ds_a is not None else _fail),
        ]
        return f"{label}  \u2014  " + "  |  ".join(parts)

    if triggered_prop == "overview-graph.clickData":
        if not click_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        point_x = click_data["points"][0].get("x", "")
        if not point_x:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        clicked_date_str = str(point_x)[:10]
        date = datetime.date.fromisoformat(clicked_date_str)
        ds_p, ds_t, ds_r, ds_a = _load_day_parallel(date.year, date.month, date.day)
        store = {"start": clicked_date_str, "end": clicked_date_str}
        msg = _detail_status(ds_p, ds_t, ds_r, ds_a, clicked_date_str)
        return _build_multi_day_fig(ds_p, ds_t, ds_r, ds_a, date), clicked_date_str, store, msg

    if triggered_prop == "day-picker.date":
        if not date_str:
            return _empty_fig(), dash.no_update, dash.no_update, ""
        date = datetime.date.fromisoformat(str(date_str)[:10])
        ds_p, ds_t, ds_r, ds_a = _load_day_parallel(date.year, date.month, date.day)
        day_str = date.isoformat()
        store = {"start": day_str, "end": day_str}
        msg = _detail_status(ds_p, ds_t, ds_r, ds_a, day_str)
        return _build_multi_day_fig(ds_p, ds_t, ds_r, ds_a, date), dash.no_update, store, msg

    if triggered_prop == "overview-graph.relayoutData" and relayout_data:
        if "xaxis.range[0]" in relayout_data:
            detail_start = pd.Timestamp(relayout_data["xaxis.range[0]"])
            detail_end = pd.Timestamp(relayout_data["xaxis.range[1]"])
            ds_p, ds_t, ds_r, ds_a = _load_range_parallel(detail_start, detail_end)
            store = {"start": detail_start.isoformat(), "end": detail_end.isoformat()}
            label = f"{detail_start:%Y-%m-%d %H:%M} – {detail_end:%Y-%m-%d %H:%M}"
            msg = _detail_status(ds_p, ds_t, ds_r, ds_a, label)
            return _build_multi_range_fig(ds_p, ds_t, ds_r, ds_a, detail_start, detail_end), dash.no_update, store, msg
        if "xaxis.autorange" in relayout_data:
            return _empty_fig(), dash.no_update, None, ""

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("detail-graph", "figure", allow_duplicate=True),
    Input("detail-graph", "relayoutData"),
    State("detail-graph", "figure"),
    prevent_initial_call=True,
)
def update_detail_mode(relayout_data, current_figure):
    if not relayout_data or current_figure is None:
        return dash.no_update
    if not current_figure.get("data"):
        return dash.no_update
    if "xaxis.range[0]" in relayout_data:
        zoom_start = pd.Timestamp(relayout_data["xaxis.range[0]"])
        zoom_end = pd.Timestamp(relayout_data["xaxis.range[1]"])
        new_mode = "lines+markers" if (zoom_end - zoom_start) < pd.Timedelta(hours=1) else "markers"
    elif "xaxis.autorange" in relayout_data:
        new_mode = "markers"
    else:
        return dash.no_update
    patched = Patch()
    for i in range(len(current_figure["data"])):
        patched["data"][i]["mode"] = new_mode
    return patched


@callback(
    Output("overview-graph", "figure", allow_duplicate=True),
    Output("detail-graph", "figure", allow_duplicate=True),
    Input("autorange-radio", "value"),
    prevent_initial_call=True,
)
def toggle_autorange(mode):
    def make_patch():
        p = Patch()
        for i, key in enumerate(VAR_ORDER, start=1):
            axis_key = "yaxis" if i == 1 else f"yaxis{i}"
            if key == "wind_from_direction":
                p["layout"][axis_key]["autorange"] = False
                p["layout"][axis_key]["range"] = [0, 360]
            else:
                p["layout"][axis_key]["autorange"] = (mode == "auto")
        return p
    return make_patch(), make_patch()


@callback(
    Output("download-nc", "data"),
    Input("download-btn", "n_clicks"),
    State("detail-range-store", "data"),
    prevent_initial_call=True,
)
def download_netcdf(n_clicks, store):
    import io
    import zipfile

    if not store:
        return dash.no_update

    start = pd.Timestamp(store["start"])
    end = pd.Timestamp(store["end"])
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    instrument_names = {
        "pressure": "ncas-pressure-1",
        "trh": "ncas-temperature-rh-1",
        "rain": "ncas-rain-gauge-1",
        "anem": "ncas-anemometer-2",
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for source, name in instrument_names.items():
            ds = _build_download_dataset(source, start, end)
            if ds is None:
                continue
            nc_buf = io.BytesIO()
            ds.to_netcdf(nc_buf)
            nc_buf.seek(0)
            zf.writestr(f"cao_{name}_{start_str}_{end_str}.nc", nc_buf.read())

    zip_buf.seek(0)
    return dcc.send_bytes(zip_buf.read(), f"chilbolton_{start_str}_{end_str}.zip")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chilbolton Surface Met Explorer — THREDDS/OPeNDAP edition."
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
