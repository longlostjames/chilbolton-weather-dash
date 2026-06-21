#!/usr/bin/env python3
"""
Interactive Dash app for exploring Chilbolton meteorology data — wx2026 machine.

Run with:
    python app_wx2026.py [--port 8050] [--host 127.0.0.1]

Access via SSH local port forwarding:
    ssh -L 8050:localhost:8050 <username>@wx2026
Then open http://localhost:8050 in your browser.
"""

import argparse
import contextlib
import datetime
import glob
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from dash import Input, Output, Patch, State, callback, dcc, html

# ---------------------------------------------------------------------------
# Data roots — wx2026 machine paths
# ---------------------------------------------------------------------------
MET_SENSORS_ROOT = "/data/wexp/cwalden/met-sensors"

PRESSURE_ROOTS = [
    ("/data/wexp/cwalden/pressure", "yearly"),
    (MET_SENSORS_ROOT, "yearly"),
]
TRH_ROOTS = [
    ("/data/wexp/cwalden/temperature-rh", "yearly"),
    (MET_SENSORS_ROOT, "yearly"),
]
RAIN_ROOTS = [
    ("/data/wexp/cwalden/precipitation", "yearly"),
]
ANEM_ROOTS = [
    ("/data/wexp/cwalden/mean-winds", "yearly"),
    (MET_SENSORS_ROOT, "yearly"),
]

# Keep simple aliases for _build_download_dataset compatibility
DATA_ROOT = PRESSURE_ROOTS[0][0]
TRH_DATA_ROOT = TRH_ROOTS[0][0]  # BADC (primary archive)
RAIN_DATA_ROOT = RAIN_ROOTS[0][0]
ANEM_DATA_ROOT = ANEM_ROOTS[0][0]

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------
VARIABLES = {
    "air_pressure": {
        "label": "Air Pressure",
        "var": "air_pressure",
        "qc": "qc_flag_air_pressure",
        "ylabel": "Air pressure<br>(hPa)",
        "source": "pressure",
    },
    "air_temperature": {
        "label": "Air Temperature",
        "var": "air_temperature",
        "qc": "qc_flag_air_temperature",
        "ylabel": "Air temperature<br>(\u00b0C)",
        "source": "trh",
    },
    "relative_humidity": {
        "label": "Relative Humidity",
        "var": "relative_humidity",
        "qc": "qc_flag_relative_humidity",
        "ylabel": "Relative humidity<br>(%)",
        "source": "trh",
    },
    "rainfall_rate": {
        "label": "Rainfall",
        "var": "rainfall_rate",       # synthetic — computed from thickness_of_rainfall_amount
        "qc": "qc_flag",
        "ylabel": "Rainfall rate<br>(mm hr\u207b\u00b9)",
        "source": "rain",
    },
    "wind_speed": {
        "label": "Wind Speed",
        "var": "wind_speed",
        "qc": "qc_flag_wind_speed",
        "ylabel": "Wind speed<br>(m s\u207b\u00b9)",
        "source": "anem",
    },
    "wind_from_direction": {
        "label": "Wind Direction",
        "var": "wind_from_direction",
        "qc": "qc_flag_wind_from_direction",
        "ylabel": "Wind direction<br>(\u00b0)",
        "source": "anem",
    },
}

# Global lock to serialise HDF5/netCDF4 file access across threads.
# The HDF5 library is not thread-safe in its default (serial) build;
# concurrent open() calls from multiple threads cause SIGSEGV.
# Set REOBS_NC_PARALLEL=1 to disable the lock on systems where your
# HDF5/netCDF4 build is thread-safe (most modern conda installs are).
_NC_PARALLEL = os.environ.get("REOBS_NC_PARALLEL", "0") == "1"
_nc_lock = contextlib.nullcontext() if _NC_PARALLEL else threading.Lock()

# Per-variable per-year result cache: avoids reloading files on repeated
# "Generate" clicks for the same variable.
_reobs_cache: dict = {}  # key: (variable_key, year) → ndarray | None

# ---------------------------------------------------------------------------
# Helpers — multi-root file discovery
# ---------------------------------------------------------------------------

def _version_key(path):
    """Extract (major, minor) version tuple from a filename, or (0, 0) if absent."""
    m = re.search(r'_v(\d+)\.(\d+)\.nc$', path, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def _best_file(candidates):
    """Return the single best file from a list: highest version, then newest mtime."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda p: (_version_key(p), os.path.getmtime(p)))


def _glob_day_files(roots, year, month, day):
    """
    Collect all candidate files for a date from multiple (root, layout) pairs,
    then return the single best file (highest version / newest mtime).
    Each root entry is (root, layout[, min_date[, max_date]]) where
    min_date/max_date are strings like "20240401" to restrict the date range.
    Layouts: "yearly" -> root/<year>/*.nc
             "monthly" -> root/<year>/<YYYYMM>/*.nc
             "monthly_mm" -> root/<year>/<MM>/*.nc
    """
    date_str = f"{year}{month:02d}{day:02d}"
    month_str = f"{year}{month:02d}"
    month_mm = f"{month:02d}"
    candidates = []
    for root_entry in roots:
        root, layout = root_entry[0], root_entry[1]
        min_date = root_entry[2] if len(root_entry) > 2 else None
        max_date = root_entry[3] if len(root_entry) > 3 else None
        if min_date and date_str < min_date:
            continue
        if max_date and date_str > max_date:
            continue
        if not os.path.isdir(root):
            continue
        if layout == "yearly":
            d = os.path.join(root, str(year))
        elif layout == "monthly_mm":
            d = os.path.join(root, str(year), month_mm)
        else:  # monthly (YYYYMM subdirs)
            d = os.path.join(root, str(year), month_str)
        candidates += glob.glob(os.path.join(d, f"*{date_str}*.nc"))
    return _best_file(candidates)


def _glob_month_files(roots, year, month):
    """
    Collect all candidate files for a month from multiple (root, layout) pairs.
    For each calendar date found, keep only the best-version file.
    Each root entry is (root, layout[, min_date[, max_date]]) where
    min_date/max_date are strings like "20240401" to restrict the date range.
    Layouts: "yearly" -> root/<year>/*.nc
             "monthly" -> root/<year>/<YYYYMM>/*.nc
             "monthly_mm" -> root/<year>/<MM>/*.nc
    Returns a sorted list of file paths.
    """
    month_str = f"{year}{month:02d}"
    month_mm = f"{month:02d}"
    by_date = {}
    for root_entry in roots:
        root, layout = root_entry[0], root_entry[1]
        min_date = root_entry[2] if len(root_entry) > 2 else None
        max_date = root_entry[3] if len(root_entry) > 3 else None
        if not os.path.isdir(root):
            continue
        if layout == "yearly":
            d = os.path.join(root, str(year))
            files = glob.glob(os.path.join(d, f"*{month_str}*.nc"))
        elif layout == "monthly_mm":
            d = os.path.join(root, str(year), month_mm)
            files = glob.glob(os.path.join(d, "*.nc"))
        else:  # monthly (YYYYMM subdirs)
            d = os.path.join(root, str(year), month_str)
            files = glob.glob(os.path.join(d, "*.nc"))
        for f in files:
            m = re.search(r'(\d{8})', os.path.basename(f))
            if m and m.group(1).startswith(month_str):
                if min_date and m.group(1) < min_date:
                    continue
                if max_date and m.group(1) > max_date:
                    continue
                by_date.setdefault(m.group(1), []).append(f)
    return [_best_file(v) for _, v in sorted(by_date.items()) if v]


def available_years():
    years = set()
    for root_entry in PRESSURE_ROOTS:
        root = root_entry[0]
        max_date = root_entry[3] if len(root_entry) > 3 else None
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(root, d)):
                if max_date and d > max_date[:4]:
                    continue
                years.add(int(d))
    return sorted(years)


def available_months(year):
    months = set()
    for root_entry in PRESSURE_ROOTS:
        root, layout = root_entry[0], root_entry[1]
        max_date = root_entry[3] if len(root_entry) > 3 else None
        if layout == "yearly":
            year_dir = os.path.join(root, str(year))
            if not os.path.isdir(year_dir):
                continue
            for fname in os.listdir(year_dir):
                m = re.search(r'(\d{8})', fname)
                if m and m.group(1)[:4] == str(year):
                    if max_date and m.group(1) > max_date:
                        continue
                    months.add(int(m.group(1)[4:6]))
        elif layout == "monthly_mm":
            year_dir = os.path.join(root, str(year))
            if not os.path.isdir(year_dir):
                continue
            for d in os.listdir(year_dir):
                if d.isdigit() and len(d) == 2:
                    month_num = int(d)
                    if max_date and f"{year}{d}" > max_date[:6]:
                        continue
                    months.add(month_num)
        else:  # monthly (YYYYMM subdirs)
            year_dir = os.path.join(root, str(year))
            if not os.path.isdir(year_dir):
                continue
            for d in os.listdir(year_dir):
                if d.isdigit() and len(d) == 6 and d[:4] == str(year):
                    if max_date and d > max_date[:6]:
                        continue
                    months.add(int(d[4:6]))
    return sorted(months)


def _open_nc(path):
    """Open a NetCDF file and return an in-memory Dataset, or None."""
    try:
        with _nc_lock:
            ds = xr.open_dataset(path)
            ds.load()   # pull all data into numpy arrays
            ds.close()  # release file handle (data stays in RAM)
        if "time" not in ds or ds.time.size == 0:
            return None
        # Old met-sensors format uses 'pressure'; rename to AMOF standard name
        if "air_pressure" not in ds:
            if "pressure" in ds:
                ds = ds.rename({"pressure": "air_pressure"})
            else:
                return None
        # Convert Pa → hPa if units attribute says Pa or values are implausibly large
        if ds["air_pressure"].attrs.get("units", "").upper() in ("PA", "PASCAL", "PASCALS") \
                or ds["air_pressure"].values.mean() > 10000:
            ds["air_pressure"].values[:] /= 100.0
            ds["air_pressure"].attrs["units"] = "hPa"
        # Mask unphysical values (sensor artefacts: 0 Pa, stuck-at-90000 Pa, etc.)
        # Real data at Chilbolton sits ~950–1040 hPa.  The all-time UK record low
        # is ~914 hPa; artefact values cluster at 0 and ~900–902 hPa.  Use 920 hPa
        # as a lower bound: safely above the artefacts, safely below any real data.
        p = ds["air_pressure"].values
        bad = (p < 920) | (p > 1100)
        if bad.any():
            p[bad] = np.nan
        return ds
    except Exception:
        return None


def _open_nc_trh(path):
    """Open a temp/RH NetCDF file and return an in-memory Dataset, or None."""
    try:
        with _nc_lock:
            ds = xr.open_dataset(path)
            ds.load()   # pull all data into numpy arrays
            ds.close()  # release file handle (data stays in RAM)
        if "time" not in ds or ds.time.size == 0:
            return None
        # Old met-sensors format uses 'temperature' and 'rh'; rename to AMOF standard
        if "air_temperature" not in ds and "temperature" in ds:
            ds = ds.rename({"temperature": "air_temperature"})
        if "relative_humidity" not in ds and "rh" in ds:
            ds = ds.rename({"rh": "relative_humidity"})
        if "air_temperature" not in ds and "relative_humidity" not in ds:
            return None
        # Convert temperature from Kelvin to Celsius if stored in K
        if "air_temperature" in ds:
            if ds["air_temperature"].attrs.get("units", "").upper() in ("K", "KELVIN") \
                    or ds["air_temperature"].values.mean() > 200:
                ds["air_temperature"].values[:] -= 273.15
                ds["air_temperature"].attrs["units"] = "degC"
            # Mask unphysical values (0 K artefacts, sensor faults, un-declared fill values).
            # Antarctic record low is -89.2°C; use -80°C as lower bound.
            # Global record high is +56.7°C; use +65°C as upper bound.
            t = ds["air_temperature"].values
            bad_t = (t < -80) | (t > 65)
            if bad_t.any():
                t[bad_t] = np.nan
        # Derive relative humidity from dew point when RH is not directly available
        # (met-sensors files before ~2008 have 'dew_point' but no 'rh').
        # Uses the August-Roche-Magnus approximation (T in °C):
        #   e_s(T) ∝ exp(17.625·T / (243.04 + T))  →  RH = 100 · e_s(Td) / e_s(Ta)
        if "relative_humidity" not in ds and "dew_point" in ds and "air_temperature" in ds:
            td = ds["dew_point"].values.copy().astype(float)
            if ds["dew_point"].attrs.get("units", "").upper() in ("K", "KELVIN") \
                    or np.nanmean(td[np.isfinite(td)]) > 100:
                td -= 273.15  # K → °C
            ta = ds["air_temperature"].values  # already in °C
            rh_derived = 100.0 * (
                np.exp(17.625 * td / (243.04 + td)) /
                np.exp(17.625 * ta / (243.04 + ta))
            )
            rh_derived = np.where(np.isfinite(rh_derived), rh_derived, np.nan)
            rh_derived = np.clip(rh_derived, -200.0, 200.0)  # prevent float32 overflow; bad_rh mask handles >110
            ds["relative_humidity"] = xr.DataArray(
                rh_derived.astype(np.float32),
                dims=["time"],
                attrs={"units": "%", "long_name": "relative humidity derived from dew point temperature"},
            )
        # Mask unphysical relative humidity values
        if "relative_humidity" in ds:
            rh = ds["relative_humidity"].values
            bad_rh = (rh < 0) | (rh > 110)
            if bad_rh.any():
                rh[bad_rh] = np.nan
        # Add placeholder QC flags (all-1 = good) for any absent QC variable so that
        # xr.concat across files with mixed QC presence does not produce NaN-filled arrays
        # that would cause data points to be silently dropped from plots.
        for qc_var, data_var in [
            ("qc_flag_air_temperature", "air_temperature"),
            ("qc_flag_relative_humidity", "relative_humidity"),
        ]:
            if data_var in ds and qc_var not in ds:
                ds[qc_var] = xr.DataArray(
                    np.zeros(ds.time.size, dtype=np.int8),
                    dims=["time"],
                    attrs={"flag_values": "0b", "flag_meanings": "not_used"},
                )
        return ds
    except Exception:
        return None


def load_month(year, month):
    """Return an in-memory pressure Dataset covering the whole month, or None."""
    files = _glob_month_files(PRESSURE_ROOTS, year, month)
    datasets = [ds for ds in (_open_nc(f) for f in files) if ds is not None]
    if not datasets:
        return None
    return xr.concat(datasets, dim="time")


def load_day(year, month, day):
    """Return an in-memory pressure Dataset for a single day, or None."""
    f = _glob_day_files(PRESSURE_ROOTS, year, month, day)
    return _open_nc(f) if f else None


def load_day_trh(year, month, day):
    """Return an in-memory temp/RH Dataset for a single day, or None."""
    f = _glob_day_files(TRH_ROOTS, year, month, day)
    return _open_nc_trh(f) if f else None


def load_date_range(start_date, end_date):
    """Load data from start_date to end_date (dates inclusive). Returns combined Dataset or None."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    all_files = []
    while cur <= end:
        all_files.extend(_glob_month_files(PRESSURE_ROOTS, cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    if not all_files:
        return None
    with ThreadPoolExecutor(max_workers=min(len(all_files), 8)) as ex:
        datasets = [ds for ds in ex.map(_open_nc, all_files) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    combined = combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )
    return combined


def load_date_range_trh(start_date, end_date):
    """Load temp/RH data for a date range, searching all TRH roots."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    all_files = []
    while cur <= end:
        all_files.extend(_glob_month_files(TRH_ROOTS, cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    if not all_files:
        return None
    with ThreadPoolExecutor(max_workers=min(len(all_files), 8)) as ex:
        datasets = [ds for ds in ex.map(_open_nc_trh, all_files) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    combined = combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )
    return combined


def _open_nc_rain(path):
    """Open a rain gauge NetCDF, compute rainfall_rate (mm/hr), return in-memory Dataset or None."""
    try:
        with _nc_lock:
            ds = xr.open_dataset(path)
            ds.load()
            ds.close()
        if "time" not in ds or ds.time.size == 0:
            return None
        # Old BADC cfarr-multiple-raingauges format: derive thickness_of_rainfall_amount.
        # Prefer drop_count_b (RAL Mk.IV drop-counter, ~0.004 mm/drop, higher resolution).
        # Fall back to tipping_bucket_a (R.W.Munro R100, 0.2 mm/tip).
        if "thickness_of_rainfall_amount" not in ds:
            if "drop_count_b" in ds:
                mm_per_drop = 0.004  # RAL Mk.IV: 0.004 mm/drop (fixed calibration)
                drops = ds["drop_count_b"].values.astype(float)
                drops = np.where(np.isfinite(drops) & (drops >= 0), drops, np.nan)
                ds["thickness_of_rainfall_amount"] = xr.DataArray(
                    (drops * mm_per_drop).astype("float32"), dims=["time"],
                    attrs={"units": "mm",
                           "long_name": "thickness of rainfall amount (from drop count)"},
                )
            elif "tipping_bucket_a" in ds:
                tips = ds["tipping_bucket_a"].values.astype(float)
                tips = np.where(tips >= 0, tips, np.nan)
                ds["thickness_of_rainfall_amount"] = xr.DataArray(
                    (tips * 0.2).astype("float32"), dims=["time"],
                    attrs={"units": "mm",
                           "long_name": "thickness of rainfall amount (from tipping bucket count)"},
                )
        if "thickness_of_rainfall_amount" not in ds:
            return None
        # Derive rainfall rate: accumulation (mm) per 10-s interval → mm/hr
        times = pd.to_datetime(ds["time"].values)
        dt_s = pd.Series(times).diff().dt.total_seconds().fillna(10).values
        rate = ds["thickness_of_rainfall_amount"].values / dt_s * 3600.0
        qc = ds["qc_flag"].values if "qc_flag" in ds else None
        raw_amount = ds["thickness_of_rainfall_amount"].values.astype("float32")
        result = xr.Dataset(
            {
                "rainfall_rate": ("time", rate.astype("float32"),
                                  {"long_name": "Rainfall rate", "units": "mm hr-1"}),
                "rainfall_amount": ("time", raw_amount,
                                    {"long_name": "Rainfall amount per interval", "units": "mm"}),
            },
            coords={"time": ds["time"]},
        )
        if qc is not None:
            result["qc_flag"] = ("time", qc)
        return result
    except Exception:
        return None


def _add_rain_accumulation(ds):
    """Add rainfall_accumulation (cumulative mm) to a rain Dataset in-place."""
    if ds is not None and "rainfall_amount" in ds:
        cumulative = np.nancumsum(ds["rainfall_amount"].values)
        ds["rainfall_accumulation"] = xr.DataArray(
            cumulative.astype("float32"), dims=["time"],
            attrs={"long_name": "Rainfall accumulation", "units": "mm"},
        )
    return ds


def load_day_rain(year, month, day):
    """Return an in-memory rain Dataset for a single day, or None."""
    f = _glob_day_files(RAIN_ROOTS, year, month, day)
    return _add_rain_accumulation(_open_nc_rain(f) if f else None)


def load_date_range_rain(start_date, end_date):
    """Load rain data for a date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    all_files = []
    while cur <= end:
        all_files.extend(_glob_month_files(RAIN_ROOTS, cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    if not all_files:
        return None
    with ThreadPoolExecutor(max_workers=min(len(all_files), 8)) as ex:
        datasets = [ds for ds in ex.map(_open_nc_rain, all_files) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    combined = combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )
    return _add_rain_accumulation(combined)


def _open_nc_anem(path):
    """Open an anemometer NetCDF and return an in-memory Dataset, or None."""
    try:
        with _nc_lock:
            ds = xr.open_dataset(path)
            ds.load()
            ds.close()
        if "time" not in ds or ds.time.size == 0:
            return None
        if "wind_speed" not in ds and "wind_from_direction" not in ds and "wind_direction" not in ds:
            return None
        # Old cfarr-format files use wind_direction; rename to match AMOF standard
        if "wind_direction" in ds and "wind_from_direction" not in ds:
            ds = ds.rename({"wind_direction": "wind_from_direction"})
        # Drop scalar lat/lon dimensions so concat works cleanly
        squeeze_dims = [d for d in ["latitude", "longitude"] if d in ds.sizes and ds.sizes[d] == 1]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims, drop=True)
        # Mask unphysical values (e.g. whole-day sensor faults where all channels
        # contain garbage values in the 0–300 range).
        # WMO surface wind speed record is ~113 m/s; use 100 m/s as upper bound.
        if "wind_speed" in ds:
            ws = ds["wind_speed"].values
            bad_ws = (ws < 0) | (ws > 100)
            if bad_ws.any():
                ws[bad_ws] = np.nan
        if "wind_from_direction" in ds:
            wd = ds["wind_from_direction"].values
            bad_wd = (wd < 0) | (wd > 360)
            if bad_wd.any():
                wd[bad_wd] = np.nan
        return ds
    except Exception:
        return None


def load_day_anem(year, month, day):
    """Return an in-memory anemometer Dataset for a single day, or None."""
    f = _glob_day_files(ANEM_ROOTS, year, month, day)
    return _open_nc_anem(f) if f else None


def load_date_range_anem(start_date, end_date):
    """Load anemometer data for a date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    all_files = []
    while cur <= end:
        all_files.extend(_glob_month_files(ANEM_ROOTS, cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    if not all_files:
        return None
    with ThreadPoolExecutor(max_workers=min(len(all_files), 8)) as ex:
        datasets = [ds for ds in ex.map(_open_nc_anem, all_files) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time").sortby("time")
    return combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )


MAX_OVERVIEW_POINTS = 50_000


def _build_download_dataset(source, start, end):
    """Re-read raw files for a source and return a Dataset with full metadata preserved."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Ensure end covers the full final day when start==end (single day)
    if start.date() == end.date() and start.time() == datetime.time(0) and end.time() == datetime.time(0):
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if source == "pressure":
        roots = PRESSURE_ROOTS
    elif source == "trh":
        roots = TRH_ROOTS
    elif source == "rain":
        roots = RAIN_ROOTS
    else:  # anem
        roots = ANEM_ROOTS

    # Collect best file per date across all roots
    files = []
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        files += _glob_month_files(roots, cur.year, cur.month)
        cur += pd.DateOffset(months=1)

    files = [f for f in files if f]  # remove any None
    if not files:
        return None

    datasets = []
    for f in files:
        try:
            with _nc_lock:
                ds = xr.open_dataset(f)
                ds.load()
                ds.close()
            if source == "anem":
                squeeze_dims = [d for d in ["latitude", "longitude"]
                                if d in ds.sizes and ds.sizes[d] == 1]
                if squeeze_dims:
                    ds = ds.squeeze(squeeze_dims, drop=True)
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        return None

    combined = xr.concat(datasets, dim="time", combine_attrs="override").sortby("time")
    combined = combined.sel(
        time=slice(start, end)
    )

    return combined


def _thin_arrays(times, pressure, qc):
    """Subsample arrays to at most MAX_OVERVIEW_POINTS points."""
    n = len(times)
    if n <= MAX_OVERVIEW_POINTS:
        return times, pressure, qc
    step = max(1, n // MAX_OVERVIEW_POINTS)
    idx = np.arange(0, n, step)
    return times[idx], pressure[idx], (qc[idx] if qc is not None else None)


def bad_intervals(qc, times):
    """Group contiguous flag=2 samples into (start, end) interval pairs."""
    mask = qc == 2
    intervals = []
    start = None
    for i, bad in enumerate(mask):
        if bad and start is None:
            start = times[i]
        elif not bad and start is not None:
            intervals.append((start, times[i - 1]))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    return intervals


def _empty_fig():
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#f9f9f9", plot_bgcolor="#f9f9f9",
                      margin=dict(l=60, r=20, t=40, b=50))
    return fig


VAR_ORDER = ["air_pressure", "air_temperature", "relative_humidity", "rainfall_rate", "wind_speed", "wind_from_direction"]


def _make_multi_fig(title):
    """Create a subplot figure with shared x-axis, one row per variable."""
    n = len(VAR_ORDER)
    wd_row = VAR_ORDER.index("wind_from_direction") + 1
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[VARIABLES[k]["label"] for k in VAR_ORDER],
        vertical_spacing=0.06,
    )
    # Per-axis title_standoff chosen so that (tick_label_width + standoff)
    # is approximately equal across all panels, placing all rotated titles
    # at the same absolute x position regardless of tick-label width.
    #   pressure  4-digit "1010" ~28px → standoff 5  → sum 33
    #   temp      2-digit "20"   ~14px → standoff 19 → sum 33
    #   RH        3-digit "100"  ~21px → standoff 12 → sum 33
    #   rainfall  1-digit "5"     ~7px → standoff 26 → sum 33
    #   wind spd  2-digit "15"   ~14px → standoff 19 → sum 33
    #   wind dir  3-digit "360"  ~21px → standoff 12 → sum 33
    STANDOFF = {
        "air_pressure":       5,
        "air_temperature":   19,
        "relative_humidity": 12,
        "rainfall_rate":     26,
        "wind_speed":        19,
        "wind_from_direction": 12,
    }
    for i, key in enumerate(VAR_ORDER, start=1):
        kwargs = dict(
            title_text=VARIABLES[key]["ylabel"],
            title_standoff=STANDOFF[key],
            automargin=True,
            row=i, col=1,
        )
        if key == "wind_from_direction":
            kwargs["tickvals"] = [0, 90, 180, 270, 360]
            kwargs["range"] = [0, 360]
        fig.update_yaxes(**kwargs)
    fig.update_xaxes(title_text="Date / Time (UTC)", ticklabelstandoff=15, row=n, col=1)
    fig.update_xaxes(type="date")  # force datetime on all axes before dummy trace is added
    # Add a secondary y-axis overlaying the wind direction primary axis,
    # showing compass points on the right.
    wd_primary = "y" if wd_row == 1 else f"y{wd_row}"
    wd_xaxis = "x" if wd_row == 1 else f"x{wd_row}"
    compass_axis = f"y{n + 1}"
    fig.update_layout({
        f"yaxis{n + 1}": dict(
            overlaying=wd_primary,
            side="right",
            tickvals=[0, 90, 180, 270, 360],
            ticktext=["N", "E", "S", "W", "N"],
            range=[0, 360],
            showgrid=False,
        )
    })
    # Rainfall accumulation secondary y-axis (right side of the rainfall rate row).
    rain_row = VAR_ORDER.index("rainfall_rate") + 1
    rain_primary = "y" if rain_row == 1 else f"y{rain_row}"
    rain_xaxis = "x" if rain_row == 1 else f"x{rain_row}"
    accum_axis_num = n + 2
    fig.update_layout({
        f"yaxis{accum_axis_num}": dict(
            overlaying=rain_primary,
            side="right",
            title_text="Accumulation<br>(mm)",
            showgrid=False,
            rangemode="tozero",
        )
    })
    # Plotly only renders a y-axis if at least one trace references it.
    # The xaxis must also be specified explicitly so the trace is properly anchored.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            xaxis=wd_xaxis,
            yaxis=compass_axis,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=10, r=20, t=80, b=80),
        hovermode="x unified",
        clickmode="event+select",
    )
    return fig


def _nice_step(value, n_intervals):
    """Return a 'nice' tick step such that n_intervals steps cover at least value."""
    import math
    if value <= 0:
        return 1
    raw = value / n_intervals
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1:
        nice = 1
    elif norm <= 2:
        nice = 2
    elif norm <= 5:
        nice = 5
    else:
        nice = 10
    return nice * mag


def _align_rain_axes(fig, ds_rain, n_vars, n_intervals=5):
    """Set rainfall-rate and accumulation y-axis ranges so their gridlines align.

    Both axes are forced to start at 0 with the same number of intervals,
    choosing 'nice' tick spacings independently for each.
    """
    if ds_rain is None:
        return
    rain_row = VAR_ORDER.index("rainfall_rate") + 1
    rate_axis_num = rain_row  # yaxis1 → y, yaxis2 → y2, …
    accum_axis_num = n_vars + 2

    max_rate = float(np.nanmax(ds_rain["rainfall_rate"].values)) if "rainfall_rate" in ds_rain else 0
    max_accum = float(np.nanmax(ds_rain["rainfall_accumulation"].values)) if "rainfall_accumulation" in ds_rain else 0

    rate_step = _nice_step(max_rate, n_intervals)
    accum_step = _nice_step(max_accum, n_intervals)
    rate_max = rate_step * n_intervals
    accum_max = accum_step * n_intervals

    rate_key = "yaxis" if rate_axis_num == 1 else f"yaxis{rate_axis_num}"
    accum_key = f"yaxis{accum_axis_num}"
    fig.update_layout({
        rate_key: dict(range=[0, rate_max], dtick=rate_step),
        accum_key: dict(range=[0, accum_max], dtick=accum_step),
    })


def _add_var_traces(fig, row, times, values, qc, use_gl=False, use_lines=False):
    """Add good/bad traces for one variable to a subplot row."""
    Cls = go.Scattergl if use_gl else go.Scatter
    mode = "lines+markers" if use_lines else "markers"
    msize = 2 if use_gl else 3
    if qc is not None:
        good = qc == 1
        bad_mask = qc == 2
        no_qc = qc == 0
        if good.any():
            fig.add_trace(
                Cls(
                    x=times[good], y=values[good], mode=mode,
                    marker=dict(color="steelblue", size=msize),
                    line=dict(color="steelblue", width=1) if use_lines else {},
                    name="Good (flag=1)", legendgroup="good", showlegend=(row == 1),
                ),
                row=row, col=1,
            )
        if bad_mask.any():
            fig.add_trace(
                Cls(
                    x=times[bad_mask], y=values[bad_mask], mode="markers",
                    marker=dict(color="crimson",
                                size=4 if use_gl else 5,
                                symbol="circle" if use_gl else "x"),
                    name="Bad (flag=2)", legendgroup="bad", showlegend=(row == 1),
                ),
                row=row, col=1,
            )
        if no_qc.any():
            fig.add_trace(
                Cls(
                    x=times[no_qc], y=values[no_qc], mode=mode,
                    marker=dict(color="grey", size=msize),
                    line=dict(color="grey", width=1) if use_lines else {},
                    name="No QC applied (flag=0)", legendgroup="noqc", showlegend=(row == 1),
                ),
                row=row, col=1,
            )
    else:
        fig.add_trace(
            Cls(
                x=times, y=values, mode=mode,
                marker=dict(color="steelblue", size=msize),
                line=dict(color="steelblue", width=1) if use_lines else {},
                name="Data", legendgroup="good", showlegend=False,
            ),
            row=row, col=1,
        )


def _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date):
    """Build a multi-panel detail figure for a single day."""
    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    fig = _make_multi_fig(f"Detail \u2014 {date}")
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        if cfg["source"] == "pressure":
            ds = ds_pressure
        elif cfg["source"] == "trh":
            ds = ds_trh
        elif cfg["source"] == "rain":
            ds = ds_rain
        else:
            ds = ds_anem
        if ds is None or cfg["var"] not in ds:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        _add_var_traces(fig, i, times, values, qc, use_gl=False)
    if ds_rain is not None and "rainfall_accumulation" in ds_rain:
        rain_row = VAR_ORDER.index("rainfall_rate") + 1
        rain_xaxis = "x" if rain_row == 1 else f"x{rain_row}"
        accum_axis = f"y{len(VAR_ORDER) + 2}"
        times = pd.to_datetime(ds_rain["time"].values)
        accum = ds_rain["rainfall_accumulation"].values
        fig.add_trace(go.Scatter(
            x=times, y=accum,
            xaxis=rain_xaxis, yaxis=accum_axis,
            mode="lines", line=dict(color="darkorange", width=1, shape="hv"),
            marker=dict(size=0, opacity=0),
            name="Accumulation (mm)", showlegend=True,
        ))
    _align_rain_axes(fig, ds_rain, len(VAR_ORDER))
    return fig


def _build_multi_range_fig(ds_pressure, ds_trh, ds_rain, ds_anem, start, end):
    """Build a multi-panel detail figure for an arbitrary zoomed range."""
    duration = end - start
    use_lines = duration < pd.Timedelta(hours=1)
    title = f"Detail \u2014 {start.date()} to {end.date()}" if duration >= pd.Timedelta(days=1) else f"Detail \u2014 {start}"
    fig = _make_multi_fig(title)
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        if cfg["source"] == "pressure":
            ds = ds_pressure
        elif cfg["source"] == "trh":
            ds = ds_trh
        elif cfg["source"] == "rain":
            ds = ds_rain
        else:
            ds = ds_anem
        if ds is None or cfg["var"] not in ds or ds.time.size == 0:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        _add_var_traces(fig, i, times, values, qc, use_gl=not use_lines, use_lines=use_lines)
    if ds_rain is not None and "rainfall_accumulation" in ds_rain:
        rain_row = VAR_ORDER.index("rainfall_rate") + 1
        rain_xaxis = "x" if rain_row == 1 else f"x{rain_row}"
        accum_axis = f"y{len(VAR_ORDER) + 2}"
        times = pd.to_datetime(ds_rain["time"].values)
        accum = ds_rain["rainfall_accumulation"].values
        fig.add_trace(go.Scatter(
            x=times, y=accum,
            xaxis=rain_xaxis, yaxis=accum_axis,
            mode="lines", line=dict(color="darkorange", width=1, shape="hv"),
            marker=dict(size=0, opacity=0),
            name="Accumulation (mm)", showlegend=True,
        ))
    _align_rain_axes(fig, ds_rain, len(VAR_ORDER))
    return fig


# ---------------------------------------------------------------------------
# ReObs plot
# ---------------------------------------------------------------------------

# Discrete rainfall colorscale matching the UK Met Office / NIMROD radar palette.
# Eight flat-colour bands; zmin=0, zmax=32 mm/hr.  Values > 32 clip to dark red.
_RAIN_REOBS_COLORSCALE = [
    [0.000000, "#000080"], [0.015625, "#000080"],  # < 0.5  navy
    [0.015625, "#3399FF"], [0.031250, "#3399FF"],  # 0.5–1  sky blue
    [0.031250, "#00AA00"], [0.062500, "#00AA00"],  # 1–2    green
    [0.062500, "#AAFF00"], [0.125000, "#AAFF00"],  # 2–4    lime
    [0.125000, "#FFCC00"], [0.250000, "#FFCC00"],  # 4–8    amber
    [0.250000, "#FF8000"], [0.500000, "#FF8000"],  # 8–16   orange
    [0.500000, "#FF0000"], [0.999999, "#FF0000"],  # 16–32  red
    [0.999999, "#880000"], [1.000000, "#880000"],  # > 32   dark red
]

_REOBS_COLORSCALE = {
    "air_pressure":        "RdYlBu",
    "air_temperature":     "RdYlBu_r",
    "relative_humidity":   "YlGnBu",
    "rainfall_rate":       _RAIN_REOBS_COLORSCALE,
    "wind_speed":          "YlOrRd",
    "wind_from_direction": "HSV",
}


def _build_reobs_fig(variable_key, year_min=None, year_max=None):
    """Build a ReObs-style heatmap: time-of-day × day-of-year, stacked by year."""
    cfg = VARIABLES[variable_key]
    source = cfg["source"]
    var_name = cfg["var"]
    qc_key = cfg.get("qc")

    source_map = {
        "pressure": (PRESSURE_ROOTS, _open_nc),
        "trh":      (TRH_ROOTS,      _open_nc_trh),
        "rain":     (RAIN_ROOTS,     _open_nc_rain),
        "anem":     (ANEM_ROOTS,     _open_nc_anem),
    }
    roots, open_fn = source_map[source]

    # Discover available years from directory listings
    all_years = set()
    for root_entry in roots:
        root = root_entry[0]
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(root, d)):
                all_years.add(int(d))

    if not all_years:
        fig = _empty_fig()
        fig.update_layout(title=f"No data available for {cfg['label']}")
        return fig

    years = sorted(all_years)
    # Apply user-selected year range
    if year_min is not None:
        years = [y for y in years if y >= year_min]
    if year_max is not None:
        years = [y for y in years if y <= year_max]
    if not years:
        fig = _empty_fig()
        fig.update_layout(title=f"No data in selected year range for {cfg['label']}")
        return fig
    # Rainfall ReObs uses hourly accumulations (sum of per-interval mm amounts).
    # All other variables use 30-minute mean bins.
    if variable_key == "rainfall_rate":
        BIN_MINUTES = 60
        reobs_var = "rainfall_amount"   # mm per instrument interval, summed per hour
        reobs_agg = "sum"
        colorbar_title = "Hourly rainfall (mm)"
    else:
        BIN_MINUTES = 30
        reobs_var = var_name
        reobs_agg = "mean"
        colorbar_title = cfg["ylabel"].replace("<br>", " ")
    BINS_PER_DAY = 24 * 60 // BIN_MINUTES
    N_DAYS = 366
    GAP_ROWS = 1                             # NaN separator between year bands
    y_per_year = BINS_PER_DAY + GAP_ROWS

    def process_year(year):
        """Load one year's files and return a binned (BINS_PER_DAY, N_DAYS) array, using cache."""
        _key = (variable_key, year)
        if _key in _reobs_cache:
            return year, _reobs_cache[_key]
        try:
            all_files = []
            for month in range(1, 13):
                all_files.extend(_glob_month_files(roots, year, month))
            if not all_files:
                _reobs_cache[_key] = None
                return year, None
            datasets = [ds for ds in (open_fn(f) for f in all_files) if ds is not None]
            if not datasets:
                _reobs_cache[_key] = None
                return year, None
            # Strip each dataset to only the needed variables + time dimension,
            # dropping non-index coordinates (latitude, longitude, etc.) that
            # differ between met-sensors and NCAS file formats and cause concat to fail.
            keep_vars = {reobs_var}
            if qc_key:
                keep_vars.add(qc_key)
            datasets_clean = []
            for ds in datasets:
                if reobs_var not in ds:
                    continue
                ds_clean = ds[[v for v in keep_vars if v in ds]].reset_coords(drop=True)
                datasets_clean.append(ds_clean)
            if not datasets_clean:
                return year, None
            try:
                combined = xr.concat(datasets_clean, dim="time", data_vars="all")
            except Exception:
                return year, None
            if reobs_var not in combined:
                return year, None
            times = pd.to_datetime(combined["time"].values)
            values = combined[reobs_var].values.astype(float)
            if qc_key and qc_key in combined:
                qc = combined[qc_key].values
                values = np.where(qc == 2, np.nan, values)
            del combined
            doy = times.day_of_year - 1   # 0-indexed (0–365)
            bin_idx = ((times.hour * 60 + times.minute) // BIN_MINUTES).astype(int)
            df = pd.DataFrame({"doy": doy, "bin": bin_idx, "v": values})
            df = df[np.isfinite(df["v"])]
            if df.empty:
                return year, None
            pivot = (
                df.groupby(["bin", "doy"])["v"]
                .agg(reobs_agg)
                .unstack(level=1)
                .reindex(index=range(BINS_PER_DAY), columns=range(N_DAYS))
            )
            result = pivot.values
            _reobs_cache[_key] = result
            return year, result
        except Exception:
            return year, None

    with ThreadPoolExecutor(max_workers=min(len(years), 4)) as ex:
        year_results = dict(ex.map(process_year, years))

    # Assemble full z matrix — oldest year at top rows (row 0 = oldest, bin 0 = 00:00)
    z = np.full((len(years) * y_per_year, N_DAYS), np.nan)
    for yi, year in enumerate(years):
        arr = year_results.get(year)
        if arr is not None:
            z[yi * y_per_year: yi * y_per_year + BINS_PER_DAY, :] = arr

    # Y-axis ticks: 0, 6, 12, 18 for every year band
    tickvals, ticktext = [], []
    for yi in range(len(years)):
        ro = yi * y_per_year
        for h in [0, 6, 12, 18]:
            tickvals.append(ro + (h * 60) // BIN_MINUTES)
            ticktext.append(str(h))

    # Year labels as right-hand annotations (rotated 90°)
    annotations = [
        dict(
            xref="paper", x=1.01,
            yref="y", y=yi * y_per_year + BINS_PER_DAY // 2,
            text=str(year), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=10),
            textangle=90,
        )
        for yi, year in enumerate(years)
    ]

    # White separator lines between year bands
    shapes = [
        dict(
            type="line",
            x0=0.5, x1=N_DAYS + 0.5,
            y0=yi * y_per_year - 0.5, y1=yi * y_per_year - 0.5,
            xref="x", yref="y",
            line=dict(color="white", width=1),
        )
        for yi in range(1, len(years))
    ]

    colorscale = _REOBS_COLORSCALE.get(variable_key, "RdYlBu_r")

    heatmap_kwargs = dict(
        z=z,
        x=list(range(1, N_DAYS + 1)),
        colorscale=colorscale,
        colorbar=dict(title=dict(text=colorbar_title, side="right"), x=1.08),
        hovertemplate="Day %{x}<br>Value: %{z:.2f}<extra></extra>",
    )
    if variable_key == "rainfall_rate":
        heatmap_kwargs["zmin"] = 0
        heatmap_kwargs["zmax"] = 32
        heatmap_kwargs["colorbar"] = dict(
            title=dict(text=colorbar_title, side="right"),
            x=1.08,
            tickvals=[0, 0.5, 1, 2, 4, 8, 16, 32],
            ticktext=["0", "0.5", "1", "2", "4", "8", "16", "≥32"],
        )

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))
    fig.update_layout(
        title=f"ReObs \u2014 {cfg['label']}",
        xaxis=dict(title="Day of year", tickvals=list(range(10, 367, 10))),
        yaxis=dict(
            title="Time (UTC)",
            tickvals=tickvals,
            ticktext=ticktext,
            autorange="reversed",   # oldest year (row 0) at top
        ),
        shapes=shapes,
        annotations=annotations,
        height=max(600, len(years) * 70),
        margin=dict(l=60, r=120, t=60, b=60),
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
    )
    return fig


# ---------------------------------------------------------------------------

default_end = datetime.date.today()
default_start = (default_end - datetime.timedelta(days=30)).isoformat()
default_end = default_end.isoformat()

app = dash.Dash(
    __name__,
    title="Chilbolton Surface Meteorology Explorer",
    requests_pathname_prefix=os.environ.get("DASH_REQUESTS_PATHNAME_PREFIX", "/"),
)

app.layout = html.Div(
    [
        html.H2(
            "Chilbolton Surface Meteorology Explorer",
            style={"fontFamily": "sans-serif", "margin": "16px 16px 8px"},
        ),
        # ── Controls ────────────────────────────────────────────────────────
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
                    style={
                        "marginLeft": "12px",
                        "padding": "6px 18px",
                        "fontSize": "0.95em",
                        "cursor": "pointer",
                    },
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
        # ── Side-by-side overview + detail ───────────────────────────────────
        html.Div(
            [
                # Left: Overview
                html.Div(
                    [
                        html.H3(
                            "Overview",
                            style={"fontFamily": "sans-serif", "margin": "8px 0 4px"},
                        ),
                        html.P(
                            "Drag to zoom into a time period — the detail view updates automatically. "
                            "Click a point to jump to that day.",
                            style={"fontFamily": "sans-serif", "color": "#666",
                                   "margin": "0 0 4px", "fontSize": "0.9em"},
                        ),
                        dcc.Graph(id="overview-graph", figure=_empty_fig(), style={"height": "900px"}),
                    ],
                    style={"flex": "1", "minWidth": "0", "padding": "0 8px 0 0"},
                ),
                # Right: Detail
                html.Div(
                    [
                        html.H3(
                            "Detail view",
                            style={"fontFamily": "sans-serif", "margin": "8px 0 4px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Jump to date:",
                                    style={"marginRight": "6px", "fontFamily": "sans-serif"},
                                ),
                                dcc.DatePickerSingle(id="day-picker", display_format="YYYY-MM-DD"),
                                html.Button(
                                    "Download NetCDF (zip)",
                                    id="download-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "16px",
                                        "padding": "5px 14px",
                                        "fontSize": "0.9em",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                            style={"marginBottom": "4px", "display": "flex", "alignItems": "center"},
                        ),
                        dcc.Graph(id="detail-graph", figure=_empty_fig(), style={"height": "900px"}),
                    ],
                    style={"flex": "1", "minWidth": "0", "padding": "0 0 0 8px"},
                ),
            ],
            style={"display": "flex", "alignItems": "flex-start", "margin": "0 16px"},
        ),
        dcc.Store(id="detail-range-store"),
        dcc.Store(id="reobs-years-store"),
        dcc.Download(id="download-nc"),
        # ── ReObs ────────────────────────────────────────────────────────────
        html.Hr(style={"margin": "24px 0 16px"}),
        html.Div(
            [
                html.H3(
                    "ReObs Plot",
                    style={"fontFamily": "sans-serif", "margin": "8px 0 4px"},
                ),
                html.P(
                    "Time-of-day vs day-of-year heatmap across all available years. "
                    "Generation may take a minute or two depending on data volume.",
                    style={"fontFamily": "sans-serif", "color": "#666",
                           "margin": "0 0 8px", "fontSize": "0.9em"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Variable:",
                            style={"marginRight": "6px", "fontFamily": "sans-serif"},
                        ),
                        dcc.Dropdown(
                            id="reobs-variable",
                            options=[
                                {"label": VARIABLES[k]["label"], "value": k}
                                for k in VAR_ORDER
                            ],
                            value=VAR_ORDER[0],
                            clearable=False,
                            style={"width": "280px", "marginRight": "12px"},
                        ),
                        html.Label(
                            "From:",
                            style={"marginRight": "6px", "fontFamily": "sans-serif"},
                        ),
                        dcc.Dropdown(
                            id="reobs-year-min",
                            options=[{"label": str(y), "value": y}
                                     for y in available_years()],
                            value=None,
                            clearable=True,
                            placeholder="earliest",
                            style={"width": "110px", "marginRight": "12px"},
                        ),
                        html.Label(
                            "To:",
                            style={"marginRight": "6px", "fontFamily": "sans-serif"},
                        ),
                        dcc.Dropdown(
                            id="reobs-year-max",
                            options=[{"label": str(y), "value": y}
                                     for y in available_years()],
                            value=None,
                            clearable=True,
                            placeholder="latest",
                            style={"width": "110px", "marginRight": "12px"},
                        ),
                        html.Button(
                            "Generate",
                            id="reobs-btn",
                            n_clicks=0,
                            style={
                                "padding": "6px 18px",
                                "fontSize": "0.95em",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                ),
                dcc.Loading(
                    dcc.Graph(id="reobs-graph", figure=_empty_fig()),
                    type="circle",
                ),
                html.Div(
                    id="reobs-detail-container",
                    children=[
                        html.Hr(style={"margin": "16px 0 8px"}),
                        html.P(
                            "Click a cell in the ReObs heatmap to load the full day detail below.",
                            style={"fontFamily": "sans-serif", "color": "#666",
                                   "fontSize": "0.9em", "margin": "0 0 4px"},
                        ),
                        dcc.Loading(
                            dcc.Graph(id="reobs-detail-graph", figure=_empty_fig(),
                                      style={"height": "900px"}),
                            type="circle",
                        ),
                    ],
                    style={"display": "none"},
                ),
            ],
            style={"margin": "0 16px 24px"},
        ),
    ],
    style={"maxWidth": "1300px", "margin": "0 auto"},
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("overview-graph", "figure"),
    Output("day-picker", "min_date_allowed"),
    Output("day-picker", "max_date_allowed"),
    Output("day-picker", "date"),
    Input("load-btn", "n_clicks"),
    State("start-date-input", "value"),
    State("end-date-input", "value"),
    prevent_initial_call=True,
)
def update_overview(n_clicks, start_date, end_date):
    no_dates = (None, None, None)
    if not start_date or not end_date:
        return _empty_fig(), *no_dates

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_p = ex.submit(load_date_range, start_date, end_date)
        fut_t = ex.submit(load_date_range_trh, start_date, end_date)
        fut_r = ex.submit(load_date_range_rain, start_date, end_date)
        fut_a = ex.submit(load_date_range_anem, start_date, end_date)
    ds_pressure = fut_p.result()
    ds_trh = fut_t.result()
    ds_rain = fut_r.result()
    ds_anem = fut_a.result()

    if ds_pressure is None and ds_trh is None and ds_rain is None and ds_anem is None:
        fig = _empty_fig()
        fig.update_layout(title=f"No data for {start_date} \u2013 {end_date}")
        return fig, *no_dates

    start_label = pd.Timestamp(start_date).strftime("%d %b %Y")
    end_label = pd.Timestamp(end_date).strftime("%d %b %Y")
    fig = _make_multi_fig(f"Overview \u2014 {start_label} to {end_label}")

    all_min, all_max = [], []
    for i, key in enumerate(VAR_ORDER, start=1):
        cfg = VARIABLES[key]
        if cfg["source"] == "pressure":
            ds = ds_pressure
        elif cfg["source"] == "trh":
            ds = ds_trh
        elif cfg["source"] == "rain":
            ds = ds_rain
        else:
            ds = ds_anem
        if ds is None or cfg["var"] not in ds:
            continue
        times = pd.to_datetime(ds["time"].values)
        values = ds[cfg["var"]].values
        qc = ds[cfg["qc"]].values if cfg["qc"] and cfg["qc"] in ds else None
        all_min.append(times.min())
        all_max.append(times.max())
        times, values, qc = _thin_arrays(times, values, qc)
        _add_var_traces(fig, i, times, values, qc, use_gl=True)

    if ds_rain is not None and "rainfall_accumulation" in ds_rain:
        rain_row = VAR_ORDER.index("rainfall_rate") + 1
        rain_xaxis = "x" if rain_row == 1 else f"x{rain_row}"
        accum_axis = f"y{len(VAR_ORDER) + 2}"
        r_times = pd.to_datetime(ds_rain["time"].values)
        r_accum = ds_rain["rainfall_accumulation"].values
        r_times_thin, r_accum_thin, _ = _thin_arrays(r_times, r_accum, None)
        fig.add_trace(go.Scatter(
            x=r_times_thin, y=r_accum_thin,
            xaxis=rain_xaxis, yaxis=accum_axis,
            mode="lines", line=dict(color="darkorange", width=1, shape="hv"),
            marker=dict(size=0, opacity=0),
            name="Accumulation (mm)", showlegend=True,
        ))
    _align_rain_axes(fig, ds_rain, len(VAR_ORDER))

    if not all_min:
        return fig, *no_dates

    min_date = min(all_min).date().isoformat()
    max_date = max(all_max).date().isoformat()
    return fig, min_date, max_date, max_date


@callback(
    Output("detail-graph", "figure"),
    Output("day-picker", "date", allow_duplicate=True),
    Output("detail-range-store", "data"),
    Input("overview-graph", "relayoutData"),
    Input("overview-graph", "clickData"),
    Input("day-picker", "date"),
    State("start-date-input", "value"),
    State("end-date-input", "value"),
    prevent_initial_call=True,
)
def update_detail(relayout_data, click_data, date_str, start_date, end_date):
    triggered_prop = dash.ctx.triggered[0]["prop_id"] if dash.ctx.triggered else None

    # Click on a point → show that day at full resolution
    if triggered_prop == "overview-graph.clickData":
        if not click_data:
            return dash.no_update, dash.no_update, dash.no_update
        point_x = click_data["points"][0].get("x", "")
        if not point_x:
            return dash.no_update, dash.no_update, dash.no_update
        clicked_date_str = str(point_x)[:10]
        date = datetime.date.fromisoformat(clicked_date_str)
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_p = ex.submit(load_day, date.year, date.month, date.day)
            fut_t = ex.submit(load_day_trh, date.year, date.month, date.day)
            fut_r = ex.submit(load_day_rain, date.year, date.month, date.day)
            fut_a = ex.submit(load_day_anem, date.year, date.month, date.day)
        ds_pressure, ds_trh, ds_rain, ds_anem = fut_p.result(), fut_t.result(), fut_r.result(), fut_a.result()
        store = {"start": clicked_date_str, "end": clicked_date_str}
        return _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date), clicked_date_str, store

    # Day picker changed manually → show that day
    if triggered_prop == "day-picker.date":
        if not date_str:
            return _empty_fig(), dash.no_update, dash.no_update
        date = datetime.date.fromisoformat(str(date_str)[:10])
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_p = ex.submit(load_day, date.year, date.month, date.day)
            fut_t = ex.submit(load_day_trh, date.year, date.month, date.day)
            fut_r = ex.submit(load_day_rain, date.year, date.month, date.day)
            fut_a = ex.submit(load_day_anem, date.year, date.month, date.day)
        ds_pressure, ds_trh, ds_rain, ds_anem = fut_p.result(), fut_t.result(), fut_r.result(), fut_a.result()
        day_str = date.isoformat()
        store = {"start": day_str, "end": day_str}
        return _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date), dash.no_update, store

    # Zoom / pan on overview → show selected range at full resolution
    if triggered_prop == "overview-graph.relayoutData" and relayout_data:
        if "xaxis.range[0]" in relayout_data:
            detail_start = pd.Timestamp(relayout_data["xaxis.range[0]"])
            detail_end = pd.Timestamp(relayout_data["xaxis.range[1]"])
            with ThreadPoolExecutor(max_workers=4) as ex:
                fut_p = ex.submit(load_date_range, detail_start, detail_end)
                fut_t = ex.submit(load_date_range_trh, detail_start, detail_end)
                fut_r = ex.submit(load_date_range_rain, detail_start, detail_end)
                fut_a = ex.submit(load_date_range_anem, detail_start, detail_end)
            ds_pressure, ds_trh, ds_rain, ds_anem = fut_p.result(), fut_t.result(), fut_r.result(), fut_a.result()
            store = {"start": detail_start.isoformat(), "end": detail_end.isoformat()}
            return _build_multi_range_fig(ds_pressure, ds_trh, ds_rain, ds_anem, detail_start, detail_end), dash.no_update, store
        # User hit the Reset Axes button
        if "xaxis.autorange" in relayout_data:
            return _empty_fig(), dash.no_update, None

    return dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("detail-graph", "figure", allow_duplicate=True),
    Input("detail-graph", "relayoutData"),
    State("detail-graph", "figure"),
    prevent_initial_call=True,
)
def update_detail_mode(relayout_data, current_figure):
    """Switch detail traces between markers and lines+markers based on zoom level."""
    if not relayout_data or current_figure is None:
        return dash.no_update
    n_traces = len(current_figure.get("data", []))
    if n_traces == 0:
        return dash.no_update

    if "xaxis.range[0]" in relayout_data:
        zoom_start = pd.Timestamp(relayout_data["xaxis.range[0]"])
        zoom_end = pd.Timestamp(relayout_data["xaxis.range[1]"])
        use_lines = (zoom_end - zoom_start) < pd.Timedelta(hours=1)
        new_mode = "lines+markers" if use_lines else "markers"
    elif "xaxis.autorange" in relayout_data:
        new_mode = "markers"
    else:
        return dash.no_update

    patched = Patch()
    for i in range(n_traces):
        trace = current_figure["data"][i]
        # Leave accumulation (orange step-line) and compass dummy traces untouched
        if trace.get("name") == "Accumulation (mm)" or trace.get("hoverinfo") == "skip":
            continue
        patched["data"][i]["mode"] = new_mode
    return patched


@callback(
    Output("overview-graph", "figure", allow_duplicate=True),
    Output("detail-graph", "figure", allow_duplicate=True),
    Input("autorange-radio", "value"),
    prevent_initial_call=True,
)
def toggle_autorange(mode):
    """Patch y-axis autorange on both graphs without reloading data."""
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
    """Build per-instrument NetCDF files (full metadata preserved) and send as a zip."""
    import io
    import zipfile

    if not store:
        return dash.no_update

    start = pd.Timestamp(store["start"])
    end = pd.Timestamp(store["end"])

    instrument_names = {
        "pressure": "ncas-pressure-1",
        "trh": "ncas-temperature-rh-1",
        "rain": "ncas-rain-gauge-1",
        "anem": "ncas-anemometer-2",
    }

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

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


@callback(
    Output("reobs-graph", "figure"),
    Output("reobs-years-store", "data"),
    Input("reobs-btn", "n_clicks"),
    State("reobs-variable", "value"),
    State("reobs-year-min", "value"),
    State("reobs-year-max", "value"),
    prevent_initial_call=True,
)
def update_reobs(n_clicks, variable_key, year_min, year_max):
    print(f"[DEBUG] update_reobs called: variable={variable_key!r}, year_min={year_min!r}, year_max={year_max!r}", flush=True)
    if not variable_key:
        return _empty_fig(), dash.no_update
    # Discover the years used so the click callback can decode row → year.
    source_map = {
        "pressure": PRESSURE_ROOTS,
        "trh":      TRH_ROOTS,
        "rain":     RAIN_ROOTS,
        "anem":     ANEM_ROOTS,
    }
    roots = source_map.get(VARIABLES[variable_key]["source"], PRESSURE_ROOTS)
    all_years = set()
    for root_entry in roots:
        root = root_entry[0]
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(root, d)):
                all_years.add(int(d))
    years = sorted(all_years)
    if year_min is not None:
        years = [y for y in years if y >= year_min]
    if year_max is not None:
        years = [y for y in years if y <= year_max]
    y_per_year = 25 if variable_key == "rainfall_rate" else 49
    store = {"years": years, "y_per_year": y_per_year}
    try:
        fig = _build_reobs_fig(variable_key, year_min=year_min, year_max=year_max)
        print(f"[DEBUG] _build_reobs_fig succeeded for {variable_key!r}, title={fig.layout.title.text!r}", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        fig = _empty_fig()
        fig.update_layout(title=f"ReObs error [{type(exc).__name__}]: {exc}")
    return fig, store


@callback(
    Output("reobs-detail-graph", "figure"),
    Output("reobs-detail-container", "style"),
    Input("reobs-graph", "clickData"),
    State("reobs-years-store", "data"),
    prevent_initial_call=True,
)
def update_reobs_detail(click_data, store):
    visible = {"display": "block"}
    if not click_data or not store:
        return _empty_fig(), visible
    point = click_data["points"][0]
    doy = int(point["x"])          # 1-indexed day of year
    y_row = int(point["y"])        # heatmap row index
    years = store.get("years", [])
    y_per_year = store.get("y_per_year", 49)
    yi = y_row // y_per_year
    if not years or yi >= len(years):
        return _empty_fig(), visible
    year = years[yi]
    try:
        date = (datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1))
    except (ValueError, OverflowError):
        return _empty_fig(), visible
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_p = ex.submit(load_day, date.year, date.month, date.day)
        fut_t = ex.submit(load_day_trh, date.year, date.month, date.day)
        fut_r = ex.submit(load_day_rain, date.year, date.month, date.day)
        fut_a = ex.submit(load_day_anem, date.year, date.month, date.day)
    ds_pressure = fut_p.result()
    ds_trh = fut_t.result()
    ds_rain = fut_r.result()
    ds_anem = fut_a.result()
    return _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date), visible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chilbolton Surface Meteorology Explorer — wx2026 machine."
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to serve on (default: 8050)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for local port forwarding)",
    )
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
