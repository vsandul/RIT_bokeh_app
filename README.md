# Radio-Immune Therapy Interactive Dashboard

This repository contains an interactive Bokeh app for exploring tumor-immune dynamics in combined radio-immunotherapy (RIT).

## Disclaimer

This software is for research and educational use only.
It is not clinically validated and must not be used for patient treatment decisions.

## Reference Model

The implementation is based on:

T. Friedrich, M. Scholz, M. Durante
A Predictive Biophysical Model of the Combined Action of Radiation Therapy and Immunotherapy of Cancer
International Journal of Radiation Oncology, Biology, Physics (2022)
DOI: https://doi.org/10.1016/j.ijrobp.2022.03.030

## What the App Does

The dashboard runs a forward simulation of primary and abscopal tumor-immune dynamics and updates plots live when you change controls.

Plotted outputs:

- Primary tumor burden (`T + TM`)
- Moribund primary tumor (`TM`)
- Abscopal tumor (`T2`)
- Immune signal (`A`)
- Lymphocytes in TME (`L + LM`)
- Moribund lymphocytes in TME (`LM`)
- Lymphocytes in abscopal site (`LG`)
- Lymphocyte production rate (`d`)
- Integrated primary tumor growth coefficient (`imuteff`, the time integral of μ₁)

## Main Controls

Plan constants and numerics:

- Growth/interaction constants: `a1`, `b1`, `a2`, `b2`, `amplification`
- Immune schedule levels: `fc1base`, `fc1high`, `fc4base`, `fc4high`
- Time controls: `stepsize`, `mintime`, `maxtime`

Initial conditions and immune-signal setup:

- `T1start`, `T2start`, `Lstart`, `LGstart`
- `d_max` cap for `d(t)`
- `Astart` override with `Auto Astart` toggle:
  - `Auto Astart` ON (default): `Astart=None`, `A[0]` is computed from model formula
  - `Auto Astart` OFF: `A[0]` is set to the numeric `Astart` spinner value

Immune signal parameters:

- `rho`, `lambda`, `psi`, `muL`

Treatment and radiosensitivity:

- Fractionation scheme (`fx`, supports comma-separated values and ranges)
- Lymphocyte dose in TME (`d_L`)
- IT periods (`start-stop`, multiple intervals allowed)
- LN dose fraction (`fr`)
- TME lymphocyte fraction (`g`)
- Radiation lymphocyte depletion term (`kRad`)
- Optional `L2 exhaustion` toggle (off by default): adds `-fc1 * amplification * LG` to the abscopal lymphocyte rate
- Lymphocyte radiosensitivity (`a_L`, `b_L`)
- Optional LQ override for tumor survival (`Use LQ model for ST`, `a_T`, `b_T`, dose)
- RT mode selector (`Photon` / `Carbon`)

Plot view controls:

- `Individual plots` shows a 3×3 grid with a linked time axis and crosshair. `Selected plots` combines the main series with a clickable legend.
- Toggle `Log X scale` and `Log Y scale`
- Enter manual axis ranges as `min,max` in `X range` or `Y range`.

## Run Locally

1. Clone repository:

```bash
git clone https://github.com/vsandul/RIT_bokeh_app.git
cd RIT_bokeh_app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Bokeh app:

```bash
bokeh serve --show rit_bokeh_app.py
```

Then open:

`http://localhost:5006/rit_bokeh_app`

## Run on Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vsandul/RIT_bokeh_app/main?urlpath=proxy/5006/rit_bokeh_app)

## Repository Layout

- `rit_bokeh_app.py`: Bokeh widgets, plots, and callbacks
- `rit_dashboard.html`: dashboard layout and parameter help
- `simulationFunctions.py`: numerical kernels and `rit_simulation`
- `requirements.txt`: runtime dependencies

## Requirements

- Python 3.10+
- bokeh
- numpy
- scipy
- matplotlib

## License

MIT License (see `LICENSE`).
