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

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from dash import Input, Output, callback, dcc, html

# ---------------------------------------------------------------------------
# Data root — override with the PRESSURE_DATA_ROOT environment variable
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "PRESSURE_DATA_ROOT",
    "/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-pressure-1/data/long-term/level1a",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def available_years():
    if not os.path.isdir(DATA_ROOT):
        return []
    return sorted(
        int(d)
        for d in os.listdir(DATA_ROOT)
        if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(DATA_ROOT, d))
    )


def available_months(year):
    year_dir = os.path.join(DATA_ROOT, str(year))
    if not os.path.isdir(year_dir):
        return []
    return sorted(
        int(d[4:6])
        for d in os.listdir(year_dir)
        if d.isdigit() and len(d) == 6 and os.path.isdir(os.path.join(year_dir, d))
    )


def _open_nc(path):
    """Open a NetCDF file; return None if it has no usable time/air_pressure."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
        if "time" not in ds or ds.time.size == 0:
            ds.close()
            return None
        ds = xr.decode_cf(ds)
        if "air_pressure" not in ds:
            ds.close()
            return None
        return ds
    except Exception:
        return None


def load_month(year, month):
    """Return an xarray Dataset covering the whole month, or None."""
    month_str = f"{year}{month:02d}"
    month_dir = os.path.join(DATA_ROOT, str(year), month_str)
    files = sorted(glob.glob(os.path.join(month_dir, "*.nc")))
    datasets = [ds for ds in (_open_nc(f) for f in files) if ds is not None]
    if not datasets:
        return None
    combined = xr.concat(datasets, dim="time")
    for ds in datasets:
        ds.close()
    return combined


def load_day(year, month, day):
    """Return the xarray Dataset for a single day, or None."""
    date_str = f"{year}{month:02d}{day:02d}"
    month_str = f"{year}{month:02d}"
    month_dir = os.path.join(DATA_ROOT, str(year), month_str)
    files = glob.glob(os.path.join(month_dir, f"*{date_str}*.nc"))
    if not files:
        return None
    return _open_nc(sorted(files)[0])


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


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

years = available_years()
default_year = years[-1] if years else datetime.date.today().year
default_months = available_months(default_year)
default_month = default_months[-1] if default_months else 1

app = dash.Dash(__name__, title="Chilbolton PTB110 Air Pressure")

app.layout = html.Div(
    [
        html.H2(
            "Chilbolton PTB110 Air Pressure Explorer",
            style={"fontFamily": "sans-serif", "margin": "16px 16px 8px"},
        ),
        # ── Controls ────────────────────────────────────────────────────────
        html.Div(
            [
                html.Label("Year:", style={"marginRight": "6px"}),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": str(y), "value": y} for y in years],
                    value=default_year,
                    clearable=False,
                    style={"width": "100px"},
                ),
                html.Label("Month:", style={"marginLeft": "16px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="month-dropdown",
                    options=[
                        {
                            "label": datetime.date(2000, m, 1).strftime("%B"),
                            "value": m,
                        }
                        for m in default_months
                    ],
                    value=default_month,
                    clearable=False,
                    style={"width": "130px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "fontFamily": "sans-serif",
                "margin": "0 16px 8px",
            },
        ),
        # ── Monthly overview ─────────────────────────────────────────────────
        html.H3(
            "Monthly overview",
            style={"fontFamily": "sans-serif", "margin": "8px 16px 4px"},
        ),
        html.P(
            "Click a point to load it in the daily detail view below.",
            style={"fontFamily": "sans-serif", "color": "#666", "margin": "0 16px 4px",
                   "fontSize": "0.9em"},
        ),
        dcc.Graph(id="overview-graph", style={"height": "320px"}),
        # ── Daily detail ─────────────────────────────────────────────────────
        html.H3(
            "Daily detail",
            style={"fontFamily": "sans-serif", "margin": "8px 16px 4px"},
        ),
        html.Div(
            [
                html.Label(
                    "Date:",
                    style={"marginRight": "6px", "fontFamily": "sans-serif"},
                ),
                dcc.DatePickerSingle(id="day-picker", display_format="YYYY-MM-DD"),
            ],
            style={"margin": "0 16px 8px"},
        ),
        dcc.Graph(id="detail-graph", style={"height": "320px"}),
    ],
    style={"maxWidth": "1300px", "margin": "0 auto"},
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("month-dropdown", "options"),
    Output("month-dropdown", "value"),
    Input("year-dropdown", "value"),
)
def update_months(year):
    months = available_months(year) if year else []
    options = [
        {"label": datetime.date(2000, m, 1).strftime("%B"), "value": m}
        for m in months
    ]
    return options, (months[-1] if months else None)


@callback(
    Output("overview-graph", "figure"),
    Output("day-picker", "min_date_allowed"),
    Output("day-picker", "max_date_allowed"),
    Output("day-picker", "date"),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
)
def update_overview(year, month):
    no_dates = (None, None, None)
    if not year or not month:
        return _empty_fig(), *no_dates

    ds = load_month(year, month)
    if ds is None:
        fig = _empty_fig()
        fig.update_layout(title=f"No data for {year}-{month:02d}")
        return fig, *no_dates

    times = pd.to_datetime(ds["time"].values)
    pressure = ds["air_pressure"].values
    has_qc = "qc_flag_air_pressure" in ds
    qc = ds["qc_flag_air_pressure"].values if has_qc else None

    fig = go.Figure()

    if has_qc:
        good = qc == 1
        bad = qc == 2
        fig.add_trace(
            go.Scattergl(
                x=times[good],
                y=pressure[good],
                mode="markers",
                marker=dict(color="steelblue", size=2),
                name="Good (flag=1)",
            )
        )
        if bad.any():
            fig.add_trace(
                go.Scattergl(
                    x=times[bad],
                    y=pressure[bad],
                    mode="markers",
                    marker=dict(color="crimson", size=2),
                    name="Bad (flag=2)",
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=pressure,
                mode="markers",
                marker=dict(color="steelblue", size=2),
                name="Air pressure",
            )
        )

    month_name = datetime.date(year, month, 1).strftime("%B %Y")
    fig.update_layout(
        title=f"Air pressure — {month_name}",
        xaxis_title="Date",
        yaxis_title="Air pressure (hPa)",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        margin=dict(l=70, r=20, t=50, b=50),
        hovermode="x unified",
        clickmode="event+select",
    )

    min_date = times.min().date().isoformat()
    max_date = times.max().date().isoformat()
    default_day = times.max().date().isoformat()

    ds.close()
    return fig, min_date, max_date, default_day


@callback(
    Output("day-picker", "date", allow_duplicate=True),
    Input("overview-graph", "clickData"),
    prevent_initial_call=True,
)
def overview_click_to_day(click_data):
    """When the user clicks a point in the overview, set the day picker."""
    if not click_data:
        return dash.no_update
    point_x = click_data["points"][0].get("x", "")
    if not point_x:
        return dash.no_update
    # point_x is an ISO datetime string; take the date portion
    return str(point_x)[:10]


@callback(
    Output("detail-graph", "figure"),
    Input("day-picker", "date"),
)
def update_detail(date_str):
    if not date_str:
        return _empty_fig()

    date = datetime.date.fromisoformat(str(date_str)[:10])
    ds = load_day(date.year, date.month, date.day)
    if ds is None:
        fig = _empty_fig()
        fig.update_layout(title=f"No data for {date}")
        return fig

    times = pd.to_datetime(ds["time"].values)
    pressure = ds["air_pressure"].values
    has_qc = "qc_flag_air_pressure" in ds
    qc = ds["qc_flag_air_pressure"].values if has_qc else None

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    tick_vals = pd.date_range(day_start, day_end, freq="3h")

    fig = go.Figure()

    # Grey shading for bad intervals
    shapes = []
    if has_qc:
        for t0, t1 in bad_intervals(qc, times):
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=t0,
                    x1=t1,
                    y0=0,
                    y1=1,
                    fillcolor="grey",
                    opacity=0.3,
                    line_width=0,
                )
            )

    # Main pressure trace
    fig.add_trace(
        go.Scatter(
            x=times,
            y=pressure,
            mode="lines+markers",
            marker=dict(size=3, color="steelblue"),
            line=dict(color="steelblue"),
            name="Air pressure",
        )
    )

    # Overlay bad-flagged points in red
    if has_qc:
        bad = qc == 2
        if bad.any():
            fig.add_trace(
                go.Scatter(
                    x=times[bad],
                    y=pressure[bad],
                    mode="markers",
                    marker=dict(color="crimson", size=5, symbol="x"),
                    name="Bad (flag=2)",
                )
            )

    fig.update_layout(
        title=f"Air pressure — {date}",
        xaxis=dict(
            title="Time (UTC)",
            range=[day_start, day_end],
            tickvals=tick_vals,
            tickformat="%H:%M",
        ),
        yaxis_title="Air pressure (hPa)",
        shapes=shapes,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        margin=dict(l=70, r=20, t=50, b=50),
        hovermode="x unified",
    )

    ds.close()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
