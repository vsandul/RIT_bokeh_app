#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import time

from simulationFunctions import *
print(f"korr = {korr}")


from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Slider, RangeSlider, TextInput, Spinner, Button,
    CheckboxGroup, Div
)
from bokeh.layouts import column, row, gridplot
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter, DataRange1d, Range1d
from bokeh.models import Tabs, Panel


from bokeh.themes import Theme
curdoc().theme = Theme(json={
  "attrs": {
    "Button":   {"height": 28},
    "Slider":   {"height": 36},
    "RangeSlider": {"height": 36},
    "TextInput": {"width": 200, "height": 20},
    "Spinner":  {"width": 80, "height": 50},
    "CheckboxGroup": {"labels": []},
    "Axis": {"major_label_text_font_size": "9pt", "axis_label_text_font_size": "10pt"},
    "Title": {"text_font_size": "9pt"}
  }
})


def LQmodel(d, a, b):
    return np.exp(-(a*d + b*d*d))


ST_default = 0.5
aL_default, bL_default = 0.2, 0.14


# ========= Controls =========
# Plan constants
pc_defaults = dict(
    a1=2.5, b1=0.45, a2=1.2, b2=0.4, amplification=0.5,
    fc1base=0.02, fc1high=0.6, fc4base=0.1254, fc4high=0.1254,
    stepsize=0.1, mintime=0.0, maxtime=60.0, maxvol=1000.0, maxvol2=1000.0,
    T1start=1e5, T2start=1e5, Lstart=100.0, LGstart=100.0,
    d_max=np.inf, rho=0.15, lam=0.15, psi=7.0, muL=-0.15
)

def spin(label, value, step, low=None, high=None, fmt="0.00"):
    return Spinner(title=label, value=value, step=step, low=low, high=high, width=80, format=fmt)

sp_a1 = spin("a1", pc_defaults["a1"], 0.01, fmt="0.00")
sp_b1 = spin("b1", pc_defaults["b1"], 0.01, fmt="0.00")
sp_a2 = spin("a2", pc_defaults["a2"], 0.01, fmt="0.00")
sp_b2 = spin("b2", pc_defaults["b2"], 0.01, fmt="0.00")
sp_amp = spin("amplification", pc_defaults["amplification"], 0.01, 0, 5, fmt="0.00")

sp_fc1b = spin("fc1base", pc_defaults["fc1base"], 0.001, 0, 10, fmt="0.000")
sp_fc1h = spin("fc1high", pc_defaults["fc1high"], 0.001, 0, 10, fmt="0.000")
sp_fc4b = spin("fc4base", pc_defaults["fc4base"], 1e-4, 0, 10, fmt="0.0000")
sp_fc4h = spin("fc4high", pc_defaults["fc4high"], 1e-4, 0, 10, fmt="0.0000")

sp_dt   = spin("stepsize (dt)", pc_defaults["stepsize"], 0.001, 0.0001, 3, fmt="0.000")
sp_tmin = spin("mintime", pc_defaults["mintime"], 0.5, 0, 100, fmt="0.0")
sp_tmax = spin("maxtime", pc_defaults["maxtime"], 0.5, 1, np.inf, fmt="0.0")
sp_mv   = spin("maxvol", pc_defaults["maxvol"], 100, 0, 1e6, fmt="0")
sp_mv2  = spin("maxvol2", pc_defaults["maxvol2"], 100, 0, 1e6, fmt="0")

# initial conditions and d_max
sp_T1start = spin("T1start", pc_defaults["T1start"], 1e8, 0, np.inf, fmt="0.2e")
sp_T2start = spin("T2start", pc_defaults["T2start"], 1e8, 0, np.inf, fmt="0.2e")
sp_Lstart  = spin("Lstart",  pc_defaults["Lstart"],  100,   0, 1e6, fmt="0.2e")
sp_LGstart = spin("LGstart", pc_defaults["LGstart"], 100,   0, 1e6, fmt="0.2e")
sp_dmax    = spin("d_max",   pc_defaults["d_max"],   1e7, -np.inf, np.inf, fmt="0.2e")

# Signal parameters
sp_rho = spin("rho",   pc_defaults["rho"],   0.001, -100, 100, fmt="0.000")
sp_lam = spin("lambda",   pc_defaults["lam"],   0.001, -100, 100, fmt="0.000")
sp_psi = spin("psi",   pc_defaults["psi"],   0.01, -100, 100, fmt="0.00")

# Lymphocyte natural depletion
sp_muL = spin("muL",   pc_defaults["muL"],   0.001, -100, 100, fmt="0.000")

# Treatment plan inputs
fx_input = TextInput(title="Fx scheme", value="1,2,3,4,5,6,7,8,9",placeholder="e.g. 1,2,3,4,5")
dL_slider = Slider(title="d_L (Gy to L in TME)", start=0, end=20, step=0.1, value=2.0, width=200)

it_input = TextInput(
    title="IT periods (days)",
    value="1-20",
    width=200,
    placeholder="e.g. 1-20"
)

ST_slider = Slider(title="ST (tumor survival / Fx)", start=0.0, end=1.0, step=0.01, value=ST_default, width=200)
fr_slider = Slider(title="fr (LN dose fraction)", start=0.0, end=1.0, step=0.01, value=0.2, width=200)
g_slider  = Slider(title="g (fraction of L in TME)", start=0.0, end=1.0, step=0.01, value=0.63, width=200)
kRad_slider = Slider(title="kRad, fx-1", start=0.0, end=5.0, step=0.01, value=0.10, width=200)

# Optional LQ override for ST
use_lq = CheckboxGroup(labels=["Use LQ model for ST"], active=[])
radType_box = CheckboxGroup(labels=["Carbon RT mode"], active=[])
dose_for_ST = Slider(title="Dose for LQ ST (Gy)", start=0, end=10, step=0.1, value=2.0, width=200)
aT_spin = spin("a_T", 0.1, 0.01, 0, 2, fmt="0.00")
bT_spin = spin("b_T", 0.05, 0.01, 0, 2, fmt="0.00")

# Lymphocyte radiosensitivity
aL_spin = spin("a_L", aL_default, 0.01, 0, 2, fmt="0.00")
bL_spin = spin("b_L", bL_default, 0.01, 0, 2, fmt="0.00")

# Update / reset buttons
btn = Button(label="Update simulation", button_type="primary", width=140)
reset_btn = Button(label="Reset to defaults", button_type="warning", width=140)

# ========= Data sources for 9 panels =========
src_Tsum = ColumnDataSource(data=dict(x=[], y=[]))
src_TM   = ColumnDataSource(data=dict(x=[], y=[]))
src_T2   = ColumnDataSource(data=dict(x=[], y=[]))
src_A    = ColumnDataSource(data=dict(x=[], y=[]))
src_L    = ColumnDataSource(data=dict(x=[], y=[]))
src_LM   = ColumnDataSource(data=dict(x=[], y=[]))
src_LG   = ColumnDataSource(data=dict(x=[], y=[]))
src_d    = ColumnDataSource(data=dict(x=[], y=[]))
src_imu  = ColumnDataSource(data=dict(x=[], y=[]))

# ========= Plot-view controls =========
log_y_box = CheckboxGroup(labels=["Log Y scale"], active=[])
log_x_box = CheckboxGroup(labels=["Log X scale"], active=[])

xrange_input = TextInput(title="X range (min,max)", value="", width=160,
                         placeholder="e.g. 0,60")
yrange_input = TextInput(title="Y range (min,max)", value="", width=160,
                         placeholder="e.g. 1e2,1e6")

def _parse_range(text):
    """Return (lo, hi) floats, or None if blank/invalid."""
    text = text.strip()
    if not text:
        return None
    try:
        parts = text.replace(";", ",").split(",")
        lo, hi = float(parts[0]), float(parts[1])
        if lo >= hi:
            return None
        return (lo, hi)
    except Exception:
        return None

# ========= Figures (nxm grid) =========
def make_fig(title, source, renderer="line",
             y_axis_type="linear", x_axis_type="linear",
             x_range=None, y_range=None):
    p = figure(width=250, height=220, title=title,
               y_axis_type=y_axis_type, x_axis_type=x_axis_type)
    p.x_range = Range1d(*x_range) if x_range else DataRange1d(range_padding=0.05, only_visible=True)
    p.y_range = Range1d(*y_range) if y_range else DataRange1d(range_padding=0.05, only_visible=True)
    p.xaxis.axis_label = "Time (d)"
    p.yaxis.axis_label = "Value"
    p.yaxis.formatter = PrintfTickFormatter(format="%.1e")
    if renderer == "step":
        p.step("x", "y", source=source, line_width=2, mode="center")
    else:
        p.line("x", "y", source=source, line_width=2)
    return p

# Panel specs: (title, source, renderer)
_panel_specs = [
    ("Primary tumor, cells",                  src_Tsum, "line"),
    ("Moribund tumor, cells",                  src_TM,   "line"),
    ("Abscopal tumor, cells",                  src_T2,   "line"),
    ("Immune signal, a.u.",                    src_A,    "line"),
    ("Lymphocytes in TME, cells",              src_L,    "line"),
    ("Moribund lymph. in TME, cells",          src_LM,   "line"),
    ("Lymphocyte in abscopal site, cells",     src_LG,   "line"),
    ("Lymphocyte production rate, cells/day",  src_d,    "step"),
    ("Integrated lymphocyte production",       src_imu,  "line"),
]

# grid_container holds the current gridplot so we can swap it on toggle
grid_container = column()

def build_grid(attr, old, new):
    y_type = "log" if 0 in log_y_box.active else "linear"
    x_type = "log" if 0 in log_x_box.active else "linear"
    xr = _parse_range(xrange_input.value)
    yr = _parse_range(yrange_input.value)
    figs = [make_fig(title, src, rend,
                     y_axis_type=y_type, x_axis_type=x_type,
                     x_range=xr, y_range=yr)
            for title, src, rend in _panel_specs]
    new_grid = gridplot([figs[:4], figs[4:]])
    grid_container.children = [new_grid]

build_grid(None, None, None)

log_y_box.on_change("active", build_grid)
log_x_box.on_change("active", build_grid)
xrange_input.on_change("value", build_grid)
yrange_input.on_change("value", build_grid)


# ========= Helpers =========
def parse_fx(text):
    out = []
    if not text:
        return [np.inf]
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            a, b = int(float(a)), int(float(b))
            out.extend(list(range(min(a,b), max(a,b)+1)))
        else:
            out.append(float(token))
    out = sorted(set(out))
    return out

def parse_ITperiods(text):
    text = text.strip()
    if not text:
        return None
    periods = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid IT token '{token}'. Use 'start-stop'.")
        start, stop = token.split("-")
        periods.append((float(start), float(stop)))
    return periods


def gather_parameters():
    return dict(
        a1=float(sp_a1.value), b1=float(sp_b1.value),
        a2=float(sp_a2.value), b2=float(sp_b2.value),
        amplification=float(sp_amp.value),
        fc1base=float(sp_fc1b.value), fc1high=float(sp_fc1h.value),
        fc4base=float(sp_fc4b.value), fc4high=float(sp_fc4h.value),
        stepsize=float(sp_dt.value),
        mintime=float(sp_tmin.value),
        maxtime=float(sp_tmax.value),
        maxvol=float(sp_mv.value),
        maxvol2=float(sp_mv2.value),
        T1start=float(sp_T1start.value),
        T2start=float(sp_T2start.value),
        Lstart=float(sp_Lstart.value),
        LGstart=float(sp_LGstart.value),
        rho=float(sp_rho.value),
        lam=float(sp_lam.value),
        psi=float(sp_psi.value),
        muL=float(sp_muL.value)
    )


def reset_all():
    sp_a1.value = pc_defaults["a1"]
    sp_b1.value = pc_defaults["b1"]
    sp_a2.value = pc_defaults["a2"]
    sp_b2.value = pc_defaults["b2"]
    sp_amp.value = pc_defaults["amplification"]
    sp_fc1b.value = pc_defaults["fc1base"]
    sp_fc1h.value = pc_defaults["fc1high"]
    sp_fc4b.value = pc_defaults["fc4base"]
    sp_fc4h.value = pc_defaults["fc4high"]
    sp_dt.value   = pc_defaults["stepsize"]
    sp_tmin.value = pc_defaults["mintime"]
    sp_tmax.value = pc_defaults["maxtime"]
    sp_mv.value   = pc_defaults["maxvol"]
    sp_mv2.value  = pc_defaults["maxvol2"]
    sp_T1start.value = pc_defaults["T1start"]
    sp_T2start.value = pc_defaults["T2start"]
    sp_Lstart.value  = pc_defaults["Lstart"]
    sp_LGstart.value = pc_defaults["LGstart"]
    sp_dmax.value    = pc_defaults["d_max"]
    sp_rho.value = pc_defaults["rho"]
    sp_lam.value = pc_defaults["lam"]
    sp_psi.value = pc_defaults["psi"]
    sp_muL.value = pc_defaults["muL"]

    fx_input.value = "1,2,3,4,5,6,7,8,9"
    dL_slider.value = 2.0
    it_input.value = "1-20"
    ST_slider.value = ST_default
    fr_slider.value = 0.2
    g_slider.value  = 0.63
    kRad_slider.value = 0.10

    use_lq.active = []
    radType_box.active = []
    dose_for_ST.value = 2.0
    aT_spin.value = 0.1
    bT_spin.value = 0.05
    aL_spin.value = aL_default
    bL_spin.value = bL_default

    # Reset plot-view controls
    log_y_box.active = []
    log_x_box.active = []
    xrange_input.value = ""
    yrange_input.value = ""

    update()


# ========= Main update =========
status = Div(text="", width=600)

def update(_=None):
    try:
        fx = parse_fx(fx_input.value)
        if not fx:
            raise ValueError("fx is empty. Provide at least one fraction index.")

        params = gather_parameters()
        d_L = float(dL_slider.value)
        ITperiods = parse_ITperiods(it_input.value)
        fr = float(fr_slider.value)
        g  = float(g_slider.value)
        kRad = float(kRad_slider.value)
        aL = float(aL_spin.value)
        bL = float(bL_spin.value)
        d_max = float(sp_dmax.value) if sp_dmax.value >= 0 else np.inf

        rho = float(sp_rho.value)
        lam = float(sp_lam.value)
        psi = float(sp_psi.value)
        muL = float(sp_muL.value)

        if 0 in use_lq.active:
            ST = float(LQmodel(dose_for_ST.value, float(aT_spin.value), float(bT_spin.value)))
            ST_slider.value = round(float(ST), 4)
        else:
            ST = float(ST_slider.value)

        radType = "carbon" if 0 in radType_box.active else "photon"

        Tarr, TMarr, T2arr, Aarr, Larr, LMarr, LGarr, imuteff, timearr, darr = rit_simulation(
            fx=fx,
            dose=d_L,
            ITperiods=ITperiods,
            ST=ST,
            fr=fr,
            parameters=params,
            g=g,
            kRadL=kRad,
            radType=radType,
            aL=aL,
            bL=bL,
            d_max=d_max
        )

        src_Tsum.data = dict(x=timearr, y=np.asarray(Tarr) + np.asarray(TMarr))
        src_TM.data   = dict(x=timearr, y=np.asarray(TMarr))
        src_T2.data   = dict(x=timearr, y=np.asarray(T2arr))
        src_A.data    = dict(x=timearr, y=np.asarray(Aarr))
        src_L.data    = dict(x=timearr, y=np.asarray(Larr) + np.asarray(LMarr))
        src_LM.data   = dict(x=timearr, y=np.asarray(LMarr))
        src_LG.data   = dict(x=timearr, y=np.asarray(LGarr))
        src_d.data    = dict(x=timearr, y=np.asarray(darr))
        src_imu.data  = dict(x=timearr, y=np.asarray(imuteff))

        status.text = "<b>OK.</b> Simulation updated."
    except Exception as e:
        status.text = f"<b style='color:red'>Error:</b> {e}"

# Wire updates (live)
def _cb(attr, old, new):
    update()

value_widgets = [
    sp_a1, sp_b1, sp_a2, sp_b2, sp_amp, sp_fc1b, sp_fc1h, sp_fc4b, sp_fc4h,
    sp_dt, sp_tmin, sp_tmax, sp_mv, sp_mv2,
    sp_T1start, sp_T2start, sp_Lstart, sp_LGstart, sp_dmax,
    sp_rho, sp_lam, sp_psi, sp_muL,
    fx_input, dL_slider, it_input, ST_slider, fr_slider, g_slider, kRad_slider,
    dose_for_ST, aT_spin, bT_spin, aL_spin, bL_spin
]
for w in value_widgets:
    w.on_change("value", _cb)

use_lq.on_change("active", _cb)
radType_box.on_change("active", _cb)

btn.on_click(lambda: update())
reset_btn.on_click(reset_all)

update()

# ========= Layout =========
pc_col1 = column(
    Div(text="<b>Plan constants</b> (you can type numbers)"),
    row(
        sp_a1, sp_b1, sp_a2, sp_b2, sp_amp,
        sp_fc1b, sp_fc1h, sp_fc4b, sp_fc4h,
        sp_dt, sp_tmin, sp_tmax, sp_mv, sp_mv2
    ),
    row(
        sp_T1start, sp_T2start, sp_Lstart, sp_LGstart, sp_dmax,
        sp_rho, sp_lam, sp_psi, sp_muL
    ),
    sizing_mode="stretch_width"
)

plan_section = column(
    Div(text="<b>Treatment plan</b>"),
    row(dL_slider, fx_input, g_slider, aL_spin, bL_spin, ST_slider, column(radType_box, use_lq)),
    row(
        fr_slider,
        it_input,
        kRad_slider,
        aT_spin, bT_spin,
        dose_for_ST,
        column(btn, reset_btn)
    ),
    sizing_mode="stretch_width"
)

plot_controls = row(
    Div(text="<b>Plot view:</b>"),
    column(log_x_box, log_y_box),
    xrange_input,
    yrange_input,
)

curdoc().add_root(column(pc_col1, plan_section, plot_controls, grid_container, status))

curdoc().title = "RIT Interactive (Bokeh)"