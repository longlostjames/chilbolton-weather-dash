# chilbolton-pressure-dash

Interactive Plotly/Dash explorer for Chilbolton PTB110 barometric pressure NetCDF data.

## Features

- **Monthly overview** — scatter plot of the full month coloured by QC flag (good = blue, bad = red). Click any point to load that day in the detail view.
- **Daily detail** — line plot with 3-hourly time-axis ticks and grey shading over bad-data intervals. Date can also be set with the date picker.

## Installation

Install dependencies into your conda/Python environment:

```bash
pip install -r requirements.txt
```

Dash and Plotly are not included in the default `cao_3_11` environment, so run this once:

```bash
source /home/users/cjwalden/miniforge3/etc/profile.d/conda.sh
conda activate cao_3_11
pip install dash plotly
```

## Running on JASMIN via local port forwarding

### 1 — Start an SSH tunnel from your laptop

```bash
ssh -L 8050:localhost:8050 <username>@login2.jasmin.ac.uk
```

(Replace `login2` with whichever login node you use.)

### 2 — Start the app on the JASMIN server

On the JASMIN shell (or in a SLURM interactive session):

```bash
source /home/users/cjwalden/miniforge3/etc/profile.d/conda.sh
conda activate cao_3_11
cd /home/users/cjwalden/git/chilbolton-pressure-dash
python app.py --port 8050
```

### 3 — Open in your browser

Navigate to **http://localhost:8050**.

## Options

| Flag | Default | Description |
|---|---|---|
| `--port` | `8050` | Port to serve on |
| `--host` | `127.0.0.1` | Bind address (keep as `127.0.0.1` for local port forwarding) |
| `--data-root` | (compiled in) | Override the NetCDF data root directory |

The data root can also be set with the environment variable `PRESSURE_DATA_ROOT`:

```bash
PRESSURE_DATA_ROOT=/path/to/level1a python app.py
```

## Data root layout expected

```
<DATA_ROOT>/
  YYYY/
    YYYYMM/
      ncas-pressure-1_chilbolton_YYYYMMDD_air-pressure_v1.0.nc
      ...
```
