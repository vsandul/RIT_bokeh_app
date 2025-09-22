#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time

from simulationFunctions import *
from simulationFunctions_Fast import *
print(f"korr = {korr}")


from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Slider, RangeSlider, TextInput, Spinner, Button,
    CheckboxGroup, Div
)
from bokeh.layouts import column, row, gridplot
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter
from bokeh.models import Tabs, Panel


from bokeh.themes import Theme
curdoc().theme = Theme(json={
  "attrs": {
    "Button":   {"height": 28},
    "Slider":   {"height": 36},
    "RangeSlider": {"height": 36},
    "TextInput": {"width": 200, "height": 20},
    "Spinner":  {"width": 80, "height": 50},
    "CheckboxGroup": {"labels": []},  # labels controlled via CSS below if needed
    "Axis": {"major_label_text_font_size": "9pt", "axis_label_text_font_size": "10pt"},
    "Title": {"text_font_size": "11pt"}
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
    stepsize=0.1, mintime=0.0, maxtime=60.0, maxvol=1000.0, maxvol2=1000.0
)

def spin(label, value, step, low=None, high=None, fmt="0.00"):
    return Spinner(title=label, value=value, step=step, low=low, high=high, width=80, format=fmt)

sp_a1 = spin("a1", pc_defaults["a1"], 0.01, fmt="0.00")
sp_b1 = spin("b1", pc_defaults["b1"], 0.01, fmt="0.00")
sp_a2 = spin("a2", pc_defaults["a2"], 0.01, fmt="0.00")
sp_b2 = spin("b2", pc_defaults["b2"], 0.01, fmt="0.00")
sp_amp = spin("amplification", pc_defaults["amplification"], 0.01, 0, 5, fmt="0.00")

sp_fc1b = spin("fc1base", pc_defaults["fc1base"], 0.001, 0, 2, fmt="0.000")
sp_fc1h = spin("fc1high", pc_defaults["fc1high"], 0.001, 0, 2, fmt="0.000")
sp_fc4b = spin("fc4base", pc_defaults["fc4base"], 1e-4, 0, 5, fmt="0.0000")
sp_fc4h = spin("fc4high", pc_defaults["fc4high"], 1e-4, 0, 5, fmt="0.0000")

sp_dt   = spin("stepsize (dt)", pc_defaults["stepsize"], 0.001, 0.001, 3,fmt="0.000")
sp_tmin = spin("mintime", pc_defaults["mintime"], 0.5, 0, 100, fmt="0.0")
sp_tmax = spin("maxtime", pc_defaults["maxtime"], 0.5, 1, 365, fmt="0.0")
sp_mv   = spin("maxvol", pc_defaults["maxvol"], 100, 0, 1e6, fmt="0")
sp_mv2  = spin("maxvol2", pc_defaults["maxvol2"], 100, 0, 1e6, fmt="0")

# Treatment plan inputs
fx_input = TextInput(title="Fx scheme; e.g. 1,2,3,..", value="1,2,3,4,5,6,7,8,9")
dL_slider = Slider(title="d_L (Gy to L in TME)", start=0, end=20, step=0.1, value=2.0, width=200)
it_slider = RangeSlider(title="[startIT, stopIT], d", start=0, end=120, step=0.5, value=(1, 20),width=200)
ST_slider = Slider(title="ST (tumor survival / Fx)", start=0.0, end=1.0, step=0.01, value=ST_default, width=200)
fr_slider = Slider(title="fr (LN dose fraction)", start=0.0, end=1.0, step=0.01, value=0.2, width=200)
g_slider  = Slider(title="g (fraction of L in TME)", start=0.0, end=1.0, step=0.01, value=0.63, width=200)
kRad_slider = Slider(title="kRad, d-1", start=0.0, end=5.0, step=0.01, value=0.10, width=200)

# Optional LQ override for ST
use_lq = CheckboxGroup(labels=["Use LQ model for ST"], active=[])
radType_box = CheckboxGroup(labels=["Carbon RT mode"], active=[])
dose_for_ST = Slider(title="Dose for LQ ST (Gy)", start=0, end=10, step=0.1, value=2.0,width=200)
aT_spin = spin("a_T", 0.1, 0.01, 0, 2, fmt="0.00")
bT_spin = spin("b_T", 0.05, 0.01, 0, 2, fmt="0.00")

# Lymphocyte radiosensetivity
aL_spin = spin("a_L", aL_default, 0.01, 0, 2, fmt="0.00")
bL_spin = spin("b_L", bL_default, 0.01, 0, 2, fmt="0.00")

# Update button (also live-on-change; the button is handy after big edits)
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

# ========= Figures (nxm grid) =========
def fig(title, y_axis_type="linear", y_range=None):
    p = figure(width=250, height=220, title=title, y_axis_type=y_axis_type)
    if y_range is not None:
        p.y_range.start = y_range[0]
        p.y_range.end   = y_range[1]
    p.xaxis.axis_label = "Time (d)"
    p.yaxis.axis_label = "Value"
    p.yaxis.formatter = PrintfTickFormatter(format="%.1e")
    return p

p_Tsum = fig("Tarr + TMarr", y_range=(0, 5e8))
p_TM   = fig("TMarr")
p_T2   = fig("T2arr", y_range=(0, 5e8))
p_A    = fig("Aarr")
p_L    = fig("Larr + LMarr")
p_LM   = fig("LMarr")
p_LG   = fig("LGarr")
p_d    = fig("darr")
p_imu  = fig("imuteffarr")

p_Tsum.line("x", "y", source=src_Tsum, line_width=2)
p_TM.line("x", "y", source=src_TM, line_width=2)
p_T2.line("x", "y", source=src_T2, line_width=2)
p_A.line("x", "y", source=src_A, line_width=2)
p_L.line("x", "y", source=src_L, line_width=2)
p_LM.line("x", "y", source=src_LM, line_width=2)
p_LG.line("x", "y", source=src_LG, line_width=2)
p_d.step("x", "y", source=src_d, line_width=2, mode="center")
p_imu.line("x", "y", source=src_imu, line_width=2)

grid = gridplot([[p_Tsum, p_TM, p_T2, p_A,],
                 [p_L, p_LM, p_LG,   p_d,  p_imu]])


# ========= Helpers =========
def parse_fx(text):
    # Allow ranges like 1-5 and individual values like 7,9
    out = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            a, b = int(float(a)), int(float(b))
            out.extend(list(range(min(a,b), max(a,b)+1)))
        else:
            out.append(int(float(token)))
    # Keep unique, sorted
    out = sorted(set(out))
    return out

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
        maxvol2=float(sp_mv2.value)
    )


def reset_all():
    # --- Plan constants ---
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

    # --- Treatment plan ---
    fx_input.value = "1,2,3,4,5,6,7,8,9"
    dL_slider.value = 2.0
    it_slider.value = (1, 20)
    ST_slider.value = ST_default
    fr_slider.value = 0.2
    g_slider.value  = 0.63
    kRad_slider.value = 0.10

    # --- LQ / radType ---
    use_lq.active = []
    radType_box.active = []   # or [0] if you want carbon as default
    dose_for_ST.value = 2.0
    aT_spin.value = 0.1
    bT_spin.value = 0.05
    aL_spin.value = aL_default
    bL_spin.value = bL_default

    # Refresh plots
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
        startIT, stopIT = it_slider.value
        fr = float(fr_slider.value)
        g  = float(g_slider.value)
        kRad = float(kRad_slider.value)
        aL = float(aL_spin.value)
        bL = float(bL_spin.value)

        # ST from slider or LQ model
        if 0 in use_lq.active:
            ST = float(LQmodel(dose_for_ST.value, float(aT_spin.value), float(bT_spin.value)))
            ST_slider.value = round(float(ST), 4)
        else:
            ST = float(ST_slider.value)
            
        radType = "carbon" if 0 in radType_box.active else "photon"
        # Compute model
        Tarr, TMarr, T2arr, Aarr, Larr, LMarr, LGarr, imuteff, timearr, darr = rit2_modified_fast(
            fx, d_L, startIT, stopIT, ST, fr, params, g, kRad, radType=radType, aL=aL, bL=bL
        )

        # Update sources
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

# Widgets that change on "value"
value_widgets = [
    sp_a1, sp_b1, sp_a2, sp_b2, sp_amp, sp_fc1b, sp_fc1h, sp_fc4b, sp_fc4h,
    sp_dt, sp_tmin, sp_tmax, sp_mv, sp_mv2,
    fx_input, dL_slider, it_slider, ST_slider, fr_slider, g_slider, kRad_slider,
    dose_for_ST, aT_spin, bT_spin, aL_spin, bL_spin
]
for w in value_widgets:
    w.on_change("value", _cb)

# CheckboxGroup changes on "active"
use_lq.on_change("active", _cb)
radType_box.on_change("active", _cb)

# Button uses on_click (no args)
btn.on_click(lambda: update())

reset_btn.on_click(reset_all)

# Initial draw
update()

# ========= Layout =========
pc_col1 = column(  # Section: Plan constants 
    Div(text="<b>Plan constants</b> (you can type numbers)"),
    row(sp_a1, sp_b1, sp_a2, sp_b2, sp_amp, sp_fc1b, sp_fc1h, sp_fc4b, sp_fc4h, sp_dt, sp_tmin, sp_tmax, sp_mv, sp_mv2),
    sizing_mode="stretch_width"
)

plan_section = column(  # Section: Treatment plan
    Div(text="<b>Treatment plan</b>"),
    row(dL_slider, fx_input,  g_slider,  aL_spin, bL_spin, ST_slider, column(radType_box, use_lq) ),
    row(
        fr_slider,
        it_slider,
        kRad_slider,
        aT_spin, bT_spin,
        dose_for_ST, 
        column(btn, reset_btn)              
    ),
    sizing_mode="stretch_width"
)


curdoc().add_root(column(pc_col1, plan_section, grid))

curdoc().title = "RIT Interactive (Bokeh)"




