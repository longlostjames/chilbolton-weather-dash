#!/usr/bin/env python3
"""
Interactive Dash app for exploring Chilbolton PTB110 air pressure data.

Run with:
    python app.py [--port 8050] [--host 127.0.0.1]

Access via SSH local port forwarding:
    ssh -L 8050:localhost:8050 <username>@<jasmin-host>
Then open http://localhost:8050 in your browser.
"""

import argparse
import datetime
import glob
import os
import re

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from dash import Input, Output, Patch, State, callback, dcc, html

# ---------------------------------------------------------------------------
# Data roots — override with environment variables
# Each entry is (root_dir, layout) where layout is "yearly" or "monthly".
# "yearly"  → root/<year>/*.nc
# "monthly" → root/<year>/<yearmonth>/*.nc
# When multiple roots exist, the file with the highest version number
# (_vX.Y in the filename) is preferred; newest mtime breaks ties.
# ---------------------------------------------------------------------------
PRESSURE_ROOTS = [
    # GWS (level1a, 2019–present, yearly).  BADC copy not readable.
    (os.environ.get(
        "PRESSURE_DATA_ROOT",
        "/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-pressure-1/data/long-term/level1a",
    ), "yearly"),
]
TRH_ROOTS = [
    # BADC archive (2015–2024, yearly, v1.1)
    (os.environ.get(
        "TRH_DATA_ROOT",
        "/badc/ncas-cao/data/ncas-temperature-rh-1/20150415_longterm/v1.1",
    ), "yearly"),
    # GWS long-term archive (2024-04-01 onwards, yearly)
    (os.environ.get(
        "TRH_GWS_LONGTERM_ROOT",
        "/gws/pww/j07/ncas_obs_vol2/cao/processing/ncas-temperature-rh-1/20150415_longterm",
    ), "yearly", "20240401"),
]
RAIN_ROOTS = [
    # GWS (2020–present, monthly).  BADC copy uses incompatible old format.
    (os.environ.get(
        "RAIN_DATA_ROOT",
        "/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-rain-gauge-1/data/long-term",
    ), "monthly"),
]
ANEM_ROOTS = [
    # GWS level1a (2020–2024, yearly)
    (os.environ.get(
        "ANEM_DATA_ROOT",
        "/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-anemometer-2/data/long-term/level1a",
    ), "yearly"),
    # BADC archive (2001–2020, monthly, old cfarr format — wind_direction variable)
    (os.environ.get(
        "ANEM_BADC_ROOT",
        "/badc/ncas-cao/data/ncas-anemometer-2/20010101_longterm/previous_v1",
    ), "monthly"),
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
        "var": "rainfall_rate",       # synthetic — computed from thickness_of_rainfall_amount
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
    Each root entry is (root, layout) or (root, layout, min_date) where
    min_date is a string like "20240401" to restrict the root to dates on or after.
    """
    date_str = f"{year}{month:02d}{day:02d}"
    month_str = f"{year}{month:02d}"
    candidates = []
    for root_entry in roots:
        root, layout = root_entry[0], root_entry[1]
        min_date = root_entry[2] if len(root_entry) > 2 else None
        if min_date and date_str < min_date:
            continue
        if not os.path.isdir(root):
            continue
        if layout == "yearly":
            d = os.path.join(root, str(year))
        else:
            d = os.path.join(root, str(year), month_str)
        candidates += glob.glob(os.path.join(d, f"*{date_str}*.nc"))
    return _best_file(candidates)


def _glob_month_files(roots, year, month):
    """
    Collect all candidate files for a month from multiple (root, layout) pairs.
    For each calendar date found, keep only the best-version file.
    Each root entry is (root, layout) or (root, layout, min_date) where
    min_date is a string like "20240401" to restrict the root to dates on or after.
    Returns a sorted list of file paths.
    """
    month_str = f"{year}{month:02d}"
    by_date = {}
    for root_entry in roots:
        root, layout = root_entry[0], root_entry[1]
        min_date = root_entry[2] if len(root_entry) > 2 else None
        if not os.path.isdir(root):
            continue
        if layout == "yearly":
            d = os.path.join(root, str(year))
            files = glob.glob(os.path.join(d, f"*{month_str}*.nc"))
        else:
            d = os.path.join(root, str(year), month_str)
            files = glob.glob(os.path.join(d, "*.nc"))
        for f in files:
            m = re.search(r'(\d{8})', os.path.basename(f))
            if m and m.group(1).startswith(month_str):
                if min_date and m.group(1) < min_date:
                    continue
                by_date.setdefault(m.group(1), []).append(f)
    return [_best_file(v) for _, v in sorted(by_date.items()) if v]


def available_years():
    years = set()
    for root, _ in PRESSURE_ROOTS:
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(root, d)):
                years.add(int(d))
    return sorted(years)


def available_months(year):
    months = set()
    for root, layout in PRESSURE_ROOTS:
        if layout == "yearly":
            year_dir = os.path.join(root, str(year))
            if not os.path.isdir(year_dir):
                continue
            for fname in os.listdir(year_dir):
                m = re.search(r'(\d{8})', fname)
                if m and m.group(1)[:4] == str(year):
                    months.add(int(m.group(1)[4:6]))
        else:
            year_dir = os.path.join(root, str(year))
            if not os.path.isdir(year_dir):
                continue
            for d in os.listdir(year_dir):
                if d.isdigit() and len(d) == 6 and d[:4] == str(year):
                    months.add(int(d[4:6]))
    return sorted(months)


def _open_nc(path):
    """Open a NetCDF file and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "air_pressure" not in ds:
            ds.close()
            return None
        ds.load()   # pull all data into numpy arrays
        ds.close()  # release file handle (data stays in RAM)
        return ds
    except Exception:
        return None


def _open_nc_trh(path):
    """Open a temp/RH NetCDF file and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "air_temperature" not in ds and "relative_humidity" not in ds:
            ds.close()
            return None
        ds.load()   # pull all data into numpy arrays
        ds.close()  # release file handle (data stays in RAM)
        # Convert temperature from Kelvin to Celsius if stored in K
        if "air_temperature" in ds:
            if ds["air_temperature"].attrs.get("units", "").upper() in ("K", "KELVIN") \
                    or ds["air_temperature"].values.mean() > 200:
                ds["air_temperature"].values[:] -= 273.15
                ds["air_temperature"].attrs["units"] = "degC"
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
    months = []
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        months.append((cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    datasets = [ds for ds in (load_month(y, m) for y, m in months) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    combined = combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )
    # Each individual month was already loaded into memory, so no further .load() needed
    return combined


def load_date_range_trh(start_date, end_date):
    """Load temp/RH data for a date range, searching all TRH roots."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    datasets = []
    while cur <= end:
        for f in _glob_month_files(TRH_ROOTS, cur.year, cur.month):
            ds = _open_nc_trh(f)
            if ds is not None:
                datasets.append(ds)
        cur += pd.DateOffset(months=1)
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
        ds = xr.open_dataset(path)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "thickness_of_rainfall_amount" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        # Derive rainfall rate: accumulation (mm) per 10-s interval → mm/hr
        times = pd.to_datetime(ds["time"].values)
        dt_s = pd.Series(times).diff().dt.total_seconds().fillna(10).values
        rate = ds["thickness_of_rainfall_amount"].values / dt_s * 3600.0
        qc = ds["qc_flag"].values if "qc_flag" in ds else None
        result = xr.Dataset(
            {
                "rainfall_rate": ("time", rate.astype("float32"),
                                  {"long_name": "Rainfall rate", "units": "mm hr-1"}),
            },
            coords={"time": ds["time"]},
        )
        if qc is not None:
            result["qc_flag"] = ("time", qc)
        return result
    except Exception:
        return None


def load_day_rain(year, month, day):
    """Return an in-memory rain Dataset for a single day, or None."""
    f = _glob_day_files(RAIN_ROOTS, year, month, day)
    return _open_nc_rain(f) if f else None


def load_date_range_rain(start_date, end_date):
    """Load rain data for a date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    datasets = []
    while cur <= end:
        for f in _glob_month_files(RAIN_ROOTS, cur.year, cur.month):
            ds = _open_nc_rain(f)
            if ds is not None:
                datasets.append(ds)
        cur += pd.DateOffset(months=1)
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    return combined.sel(
        time=slice(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )


def _open_nc_anem(path):
    """Open an anemometer NetCDF and return an in-memory Dataset, or None."""
    try:
        ds = xr.open_dataset(path)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        if "wind_speed" not in ds and "wind_from_direction" not in ds and "wind_direction" not in ds:
            ds.close()
            return None
        ds.load()
        ds.close()
        # Old cfarr-format files use wind_direction; rename to match AMOF standard
        if "wind_direction" in ds and "wind_from_direction" not in ds:
            ds = ds.rename({"wind_direction": "wind_from_direction"})
        # Drop scalar lat/lon dimensions so concat works cleanly
        squeeze_dims = [d for d in ["latitude", "longitude"] if d in ds.sizes and ds.sizes[d] == 1]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims, drop=True)
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
    datasets = []
    while cur <= end:
        for f in _glob_month_files(ANEM_ROOTS, cur.year, cur.month):
            ds = _open_nc_anem(f)
            if ds is not None:
                datasets.append(ds)
        cur += pd.DateOffset(months=1)
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
    fig = make_subplots(
        rows=n, cols=1,
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
                row=i, col=1,
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
    """Add good/bad traces for one variable to a subplot row."""
    Cls = go.Scattergl if use_gl else go.Scatter
    mode = "lines+markers" if use_lines else "markers"
    msize = 2 if use_gl else 3
    if qc is not None:
        good = qc == 1
        bad_mask = qc == 2
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
    fig.update_xaxes(range=[day_start, day_end])
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
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

years = available_years()
default_end = datetime.date.today()
default_start = (default_end - datetime.timedelta(days=30)).isoformat()
default_end = default_end.isoformat()

app = dash.Dash(__name__, title="Chilbolton PTB110 Air Pressure")

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
                        dcc.Graph(id="overview-graph", style={"height": "900px"}),
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
                        dcc.Graph(id="detail-graph", style={"height": "900px"}),
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
)
def update_overview(n_clicks, start_date, end_date):
    no_dates = (None, None, None)
    if not start_date or not end_date:
        return _empty_fig(), *no_dates

    ds_pressure = load_date_range(start_date, end_date)
    ds_trh = load_date_range_trh(start_date, end_date)
    ds_rain = load_date_range_rain(start_date, end_date)
    ds_anem = load_date_range_anem(start_date, end_date)

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
        ds_pressure = load_day(date.year, date.month, date.day)
        ds_trh = load_day_trh(date.year, date.month, date.day)
        ds_rain = load_day_rain(date.year, date.month, date.day)
        ds_anem = load_day_anem(date.year, date.month, date.day)
        store = {"start": clicked_date_str, "end": clicked_date_str}
        return _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date), clicked_date_str, store

    # Day picker changed manually → show that day
    if triggered_prop == "day-picker.date":
        if not date_str:
            return _empty_fig(), dash.no_update, dash.no_update
        date = datetime.date.fromisoformat(str(date_str)[:10])
        ds_pressure = load_day(date.year, date.month, date.day)
        ds_trh = load_day_trh(date.year, date.month, date.day)
        ds_rain = load_day_rain(date.year, date.month, date.day)
        ds_anem = load_day_anem(date.year, date.month, date.day)
        day_str = date.isoformat()
        store = {"start": day_str, "end": day_str}
        return _build_multi_day_fig(ds_pressure, ds_trh, ds_rain, ds_anem, date), dash.no_update, store

    # Zoom / pan on overview → show selected range at full resolution
    if triggered_prop == "overview-graph.relayoutData" and relayout_data:
        if "xaxis.range[0]" in relayout_data:
            detail_start = pd.Timestamp(relayout_data["xaxis.range[0]"])
            detail_end = pd.Timestamp(relayout_data["xaxis.range[1]"])
            ds_pressure = load_date_range(detail_start, detail_end)
            ds_trh = load_date_range_trh(detail_start, detail_end)
            ds_rain = load_date_range_rain(detail_start, detail_end)
            ds_anem = load_date_range_anem(detail_start, detail_end)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chilbolton PTB110 air pressure interactive explorer."
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to serve on (default: 8050)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for local port forwarding)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the NetCDF data root directory",
    )
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = args.data_root

    app.run(debug=False, host=args.host, port=args.port)
