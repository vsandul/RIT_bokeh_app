from base64 import b64encode
from html import escape
from pathlib import Path

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    Button, CheckboxGroup, ColumnDataSource, CrosshairTool, CustomJSHover,
    CustomJSTickFormatter, DataRange1d, Div, HoverTool, Range1d, Select,
    Slider, Span, Spinner, Switch, TabPanel, Tabs, TextInput,
)
from bokeh.plotting import figure

from simulationFunctions import rit_simulation


TEMPLATE = Path(__file__).with_name("rit_dashboard.html")
COLORS = {
    "primary": "#087d69",
    "moribund": "#ac5a30",
    "abscopal": "#3e64ae",
    "tme": "#7550b0",
    "moribund_tme": "#7f5536",
    "abscopal_lymphocytes": "#536921",
    "signal": "#9c4766",
    "production": "#257085",
    "integrated": "#53646f",
}
WIDGET_STYLE = """
:host { font-family: system-ui, sans-serif; color: #18302b; }
.bk-input {
    background: #f6f9f8;
    border-color: #d6e4df;
    border-radius: 6px;
    color: #18302b;
    font: 12px system-ui, sans-serif;
    min-height: 28px;
}
.bk-input:focus { border-color: #087d69; }
.bk-input:disabled { color: #60746e; }
.noUi-connect { background: #087d69; }
"""
SWITCH_STYLE = """
:host(.active) .bar { background-color: #8bcbb9; }
:host(.active) .knob { background-color: #087d69; }
"""
TABS_STYLE = """
.bk-tabs-header { margin-bottom: 6px; border-color: #dae6e2; }
.bk-tab { color: #536a63; font: 600 13px system-ui, sans-serif; }
.bk-tab.bk-active { color: #095f50; background: #e9f4ef; border-radius: 7px 7px 0 0; }
"""


def spin(value, step, low=None, high=None, fmt="0.00"):
    return Spinner(value=value, step=step, low=low, high=high, format=fmt)


def slider(value, end, step, fmt="0.00"):
    return Slider(
        value=value, start=0, end=end, step=step, format=fmt,
        title="", show_value=True, bar_color="#a5d6c8",
    )


# A field's label, widget, and help text are used together in the page template.
sections = []


def section(title, *field_rows):
    sections.append({"title": title, "rows": field_rows})


fx_input = TextInput(value="1,2,3,4,5,6,7,8,9", placeholder="e.g. 1,3,5-9")
it_input = TextInput(value="1-20", placeholder="e.g. 1-20,25-30")
rad_type = Select(value="Photon", options=["Photon", "Carbon"])
d_l = slider(2.0, 20, 0.1, "0.0")
st = slider(0.5, 1, 0.01)
section(
    "Treatment plan",
    (("RT mode", rad_type, "Photon or Carbon. Selects the model for the delay in radiation-induced cell death."),),
    (("Fractionation (Fx)", fx_input, "Radiation days as a comma-separated list or inclusive ranges, e.g. 1,3,5-9. Blank means no fractions."),),
    (("Immunotherapy (days)", it_input, "Use start-stop windows separated by commas. Each applies at start < t ≤ stop. Blank means no immunotherapy."),),
    (("Dose d<sub>L</sub> (Gy)", d_l, "Dose per fraction to lymphocytes in the tumor microenvironment (TME). Range: 0–20 Gy."),
     ("Survival S<sub>T</sub>", st, "Fraction of tumor cells surviving one fraction. Range: 0–1. Calculated when the LQ switch is on.")),
)

fr = slider(0.2, 1, 0.01)
g = slider(0.63, 1, 0.01)
k_rad = slider(0.10, 5, 0.01)
exhaustion = Switch()
a_l = spin(0.2, 0.01, 0, 2)
b_l = spin(0.14, 0.01, 0, 2)
section(
    "Radiosensitivity & depletion",
    (("f<sub>r</sub>", fr, "Dose to draining lymph nodes as a fraction of the TME lymphocyte dose. Range: 0–1."),
     ("g", g, "Share of recruited lymphocytes allocated to the primary TME. The remainder go to the abscopal site. Range: 0–1.")),
    (("k<sub>Rad</sub> (fx⁻¹)", k_rad, "Abscopal lymphocyte depletion coefficient per radiation fraction. Input range: 0–5 per fraction."),
     ("L<sub>2</sub> exhaustion", exhaustion, "On adds an abscopal lymphocyte loss term scaled by the tumor killing and abscopal action coefficients.")),
    (("α<sub>L</sub> (Gy⁻¹)", a_l, "Linear coefficient for lymphocyte survival in the TME and draining lymph nodes. Range: 0–2 Gy⁻¹."),
     ("β<sub>L</sub> (Gy⁻²)", b_l, "Quadratic coefficient for lymphocyte survival in the TME and draining lymph nodes. Range: 0–2 Gy⁻².")),
)

use_lq = Switch()
lq_dose = slider(2.0, 10, 0.1, "0.0")
a_t = spin(0.1, 0.01, 0, 2)
b_t = spin(0.05, 0.01, 0, 2)
section(
    "Tumor survival (LQ)",
    (("Use LQ for S<sub>T</sub>", use_lq, "Off uses the entered survival fraction. On calculates it as exp(−αₜD − βₜD²)."),
     ("Dose for LQ S<sub>T</sub>", lq_dose, "Dose per fraction used to calculate tumor survival in LQ mode. Range: 0–10 Gy.")),
    (("α<sub>T</sub> (Gy⁻¹)", a_t, "Linear coefficient for tumor survival in LQ mode. Range: 0–2 Gy⁻¹."),
     ("β<sub>T</sub> (Gy⁻²)", b_t, "Quadratic coefficient for tumor survival in LQ mode. Range: 0–2 Gy⁻².")),
)

t1_start = spin(1e5, 1e8, 0, np.inf, "0.2e")
t2_start = spin(1e5, 1e8, 0, np.inf, "0.2e")
l_start = spin(100, 100, 0, 1e6, "0.2e")
lg_start = spin(100, 100, 0, 1e6, "0.2e")
auto_a = Switch(active=True)
a_start = spin(0.0, 100, -np.inf, np.inf, "0.2e")
d_max = spin(np.inf, 1e7, -np.inf, np.inf, "0.2e")
section(
    "Initial conditions",
    (("T<sub>1,start</sub>", t1_start, "Starting number of viable primary tumor cells. Range: 0 or greater."),
     ("T<sub>2,start</sub>", t2_start, "Starting number of abscopal tumor cells. Range: 0 or greater.")),
    (("L<sub>1,start</sub>", l_start, "Starting number of viable lymphocytes in the primary TME. Range: 0–1e6 cells."),
     ("L<sub>2,start</sub>", lg_start, "Starting number of lymphocytes in the abscopal site. Range: 0–1e6 cells.")),
    (("Auto A<sub>start</sub>", auto_a, "On computes the initial signal from the model. Off uses the number entered in the adjacent field."),
     ("A<sub>start</sub>", a_start, "Manual starting level of the attraction signal. Any finite number. Used when the automatic setting is off.")),
    (("d<sub>max</sub> (cells/day)", d_max, "Cap on the lymphocyte production rate. Zero or positive values cap the rate. Infinity or a negative value leaves it uncapped."),),
)

a1, b1, a2, b2 = (spin(value, 0.001, fmt="0.000") for value in (2.5, 0.45, 1.2, 0.4))
amplification = spin(0.5, 0.01, 0, 5)
fc1_base = spin(0.02, 0.001, 0, 10, "0.000")
fc1_high = spin(0.6, 0.001, 0, 10, "0.000")
fc4_base = spin(0.1254, 1e-4, 0, 10, "0.0000")
fc4_high = spin(0.1254, 1e-4, 0, 10, "0.0000")
section(
    "Tumor growth & checkpoints",
    (("a<sub>1</sub>", a1, "Primary tumor growth scale. Fit to the untreated growth curve. No numeric bounds in the app."),
     ("b<sub>1</sub>", b1, "Exponent relating primary tumor size to growth rate. Fit to untreated growth. No numeric bounds in the app.")),
    (("a<sub>2</sub>", a2, "Abscopal tumor growth scale. Fit to the untreated growth curve. No numeric bounds in the app."),
     ("b<sub>2</sub>", b2, "Exponent relating abscopal tumor size to growth rate. Fit to untreated growth. No numeric bounds in the app.")),
    (("Abscopal amplification", amplification, "Scales lymphocyte action on the abscopal tumor relative to the primary tumor. Range: 0–5."),),
    (("f<sub>P,base</sub>", fc1_base, "Lymphocyte-mediated tumor killing coefficient outside immunotherapy windows. Range: 0–10."),
     ("f<sub>P,IT</sub>", fc1_high, "Lymphocyte-mediated tumor killing coefficient during immunotherapy windows. Range: 0–10.")),
    (("f<sub>C,base</sub>", fc4_base, "Lymphocyte replenishment coefficient outside immunotherapy windows. Range: 0–10."),
     ("f<sub>C,IT</sub>", fc4_high, "Lymphocyte replenishment coefficient during immunotherapy windows. Range: 0–10.")),
)

rho = spin(0.30, 0.001, -100, 100, "0.000")
lam = spin(0.30, 0.001, -100, 100, "0.000")
psi = spin(14.0, 0.01, -100, 100)
mu_l = spin(-0.15, 0.001, -100, 100, "0.000")
section(
    "Immune signal",
    (("ρ", rho, "Baseline attraction signal produced per viable tumor cell and unit time. Input range: −100 to 100."),
     ("λ", lam, "Decay coefficient of the attraction signal. Positive values reduce the signal. Input range: −100 to 100.")),
    (("ψ", psi, "Multiplier for attraction signals released when irradiated tumor cells die. Input range: −100 to 100."),
     ("μ<sub>L</sub> (day⁻¹)", mu_l, "Net lymphocyte growth or decay coefficient. Negative values represent loss. Input range: −100 to 100 day⁻¹.")),
)

dt = spin(0.1, 0.001, 0.0001, 3, "0.0000")
t_min = spin(0, 0.5, 0, 100, "0.0")
t_max = spin(60, 0.5, 1, np.inf, "0.0")
section(
    "Simulation time",
    (("Time step Δt (days)", dt, "Time between simulation samples and integration steps. Range: 0.0001–3 days."),),
    (("t<sub>min</sub> (days)", t_min, "The field accepts 0–100 days. The current simulation always starts at day 0."),
     ("t<sub>max</sub> (days)", t_max, "Simulation horizon. Range: 1 day or greater.")),
)

plan_inputs = {
    "a1": a1, "b1": b1, "a2": a2, "b2": b2, "amplification": amplification,
    "fc1base": fc1_base, "fc1high": fc1_high, "fc4base": fc4_base, "fc4high": fc4_high,
    "stepsize": dt, "mintime": t_min, "maxtime": t_max, "T1start": t1_start, "T2start": t2_start,
    "Lstart": l_start, "LGstart": lg_start, "rho": rho, "lam": lam, "psi": psi, "muL": mu_l,
}

# The order here defines the rows in the individual-plots view.
plot_specs = [
    ("primary", "Primary tumor", "Cells", "Primary", "solid"),
    ("moribund", "Moribund tumor", "Cells", "Moribund", "dashed"),
    ("abscopal", "Abscopal tumor", "Cells", "Abscopal", "solid"),
    ("tme", "TME lymphocytes", "Cells", "L in TME", "solid"),
    ("moribund_tme", "Moribund TME lymphocytes", "Cells", "Moribund L in TME", "dotted"),
    ("abscopal_lymphocytes", "Abscopal lymphocytes", "Cells", "Abscopal L", "dashed"),
    ("signal", "Immune signal", "Signal (a.u.)", "Immune signal", "solid"),
    ("production", "Lymphocyte production", "Cells/day", "L production", "dotted"),
    ("integrated", "Integrated primary growth rate", "∫ μ₁ dt (dimensionless)", "", "solid"),
]
plot_title_notes = {
    "primary": "Includes moribund tumor cells (T₁ + Tₘ).",
    "moribund": "Lethally irradiated primary tumor cells awaiting clearance.",
    "abscopal": "Tumor cells at the unirradiated site.",
    "tme": "Includes moribund lymphocytes (L₁ + Lₘ).",
    "moribund_tme": "Lethally irradiated lymphocytes in the primary tumor.",
    "abscopal_lymphocytes": "Lymphocytes at the unirradiated tumor site.",
    "signal": "Tumor-derived signal driving lymphocyte production.",
    "production": "New lymphocytes produced per day.",
    "integrated": "Primary tumor growth coefficient (μ₁) integrated over time.",
}
sources = {key: ColumnDataSource(data={"x": [], "y": []}) for key, *_ in plot_specs}
combined_renderers = {}
# Bokeh 3.4 listens for changes to a layout's children, but not to TabPanel.child.
individual_plots = column(sizing_mode="stretch_both")
combined_plot = column(sizing_mode="stretch_both")
results = Tabs(
    tabs=[
        TabPanel(title="Individual plots", child=individual_plots),
        TabPanel(title="Selected plots", child=combined_plot),
    ],
    sizing_mode="stretch_both", name="results",
    stylesheets=[TABS_STYLE],
)
log_x = CheckboxGroup(labels=["Log X"], width=53, margin=0)
log_y = CheckboxGroup(labels=["Log Y"], width=53, margin=0)
x_range_input = TextInput(title="X range", placeholder="Auto", width=100, margin=0, stylesheets=[WIDGET_STYLE])
y_range_input = TextInput(title="Y range", placeholder="Auto", width=100, margin=0, stylesheets=[WIDGET_STYLE])
for control in (log_x, log_y, x_range_input, y_range_input):
    control.align = "end"
plot_controls = row(log_x, log_y, x_range_input, y_range_input, spacing=12, name="plot_controls")
status = Div(name="status", sizing_mode="stretch_width")

# Use plain numbers near unity and scientific notation for large or tiny values.
NUMBER_FORMAT = """
const value = Number(VALUE.toPrecision(4))
if (value === 0 || (Math.abs(value) >= 0.01 && Math.abs(value) < 1000))
    return String(value)
return value.toExponential().replace("e+", "e")
"""


def parse_range(text, logarithmic):
    try:
        low, high = map(float, text.replace(";", ",").split(","))
    except ValueError:
        return None
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        return None
    return None if logarithmic and low <= 0 else (low, high)


def axis_range(bounds):
    # Pan, zoom, and auto-ranging happen in the browser. The server only needs
    # the range fields, so stale range updates cannot race with plot replacement.
    if bounds is not None:
        return Range1d(*bounds, syncable=False)
    return DataRange1d(range_padding=0.04, only_visible=True, syncable=False)


def make_plot(title, y_label, x_range, y_bounds, *, combined=False):
    plot = figure(
        title=title, x_axis_label="Time (days)", y_axis_label=y_label,
        x_axis_type="log" if log_x.active else "linear",
        y_axis_type="log" if log_y.active else "linear",
        x_range=x_range, y_range=axis_range(y_bounds),
        syncable=False,  # Canvas dimensions are only used in the browser.
        sizing_mode="stretch_both", min_border=7, min_border_left=60,
        background_fill_color="#ffffff", border_fill_color="#ffffff",
        outline_line_color=None, toolbar_location="above",
        tools="pan,box_zoom,wheel_zoom,reset,save", active_scroll=None,
        styles={"border": "1px solid #dae6e2", "border-radius": "12px"},
    )
    plot.toolbar.logo = None
    for tool in plot.tools:
        tool.syncable = False
    plot.title.text_font = "system-ui"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "13px"
    plot.title.text_color = "#18302b"
    plot.title.standoff = 6
    plot.grid.grid_line_color = "#e8efec"
    plot.grid.minor_grid_line_color = None
    for axis in (plot.xaxis[0], plot.yaxis[0]):
        axis.formatter = CustomJSTickFormatter(code=NUMBER_FORMAT.replace("VALUE", "tick"))
        axis.ticker.desired_num_ticks = 6 if combined else 4
        axis.axis_label_text_font = axis.major_label_text_font = "system-ui"
        axis.axis_label_text_font_style = "normal"
        axis.axis_label_text_font_size = "11px"
        axis.major_label_text_font_size = "10px"
        axis.axis_label_text_color = axis.major_label_text_color = "#49605a"
        axis.axis_line_color = axis.major_tick_line_color = "#a7bbb4"
        axis.minor_tick_line_color = None
    return plot


def add_line(plot, spec, *, combined=False):
    key, title, _, legend, dash = spec
    kwargs = {"legend_label": legend} if combined else {}
    return plot.line(
        "x", "y", source=sources[key], name=legend or title,
        line_color=COLORS[key], line_dash=dash, line_width=2, **kwargs,
    )


def make_hover(renderers="auto", *, combined=False):
    number = CustomJSHover(code=NUMBER_FORMAT.replace("VALUE", "value"))
    tooltips = [("Time (days)", "@x{custom}"), ("Value", "@y{custom}")]
    if combined:
        tooltips.insert(0, ("Series", "$name"))
    return HoverTool(
        renderers=renderers, mode="vline", tooltips=tooltips,
        formatters={"@x": number, "@y": number},
        syncable=False,
    )


def build_plots():
    previous = {key: glyph.visible for key, glyph in combined_renderers.items()}
    x_bounds = parse_range(x_range_input.value, log_x.active)
    y_bounds = parse_range(y_range_input.value, log_y.active)
    shared_x = axis_range(x_bounds)
    cursor = Span(
        dimension="height", line_color="#71958a", line_dash="dashed",
        line_alpha=0.85, syncable=False,
    )
    individual = []
    hover = make_hover()
    for spec in plot_specs:
        key, title, y_label, _, _ = spec
        plot = make_plot(title, y_label, shared_x, y_bounds)
        add_line(plot, spec)
        plot.add_tools(hover, CrosshairTool(overlay=cursor, syncable=False))
        note = plot_title_notes[key]
        header = Div(
            text=(
                f'<strong>{escape(title)}</strong> '
                f'<span tabindex="0" aria-label="{escape(note)}">ⓘ'
                f'<span class="note" role="tooltip">{escape(note)}</span></span>'
            ),
            margin=0, height=24, sizing_mode="stretch_width",
            stylesheets=["""
                :host { z-index: 1; overflow: visible; }
                .bk-clearfix { padding: 4px 0 0 60px; font: 13px system-ui; color: #18302b; }
                [tabindex] { position: relative; color: #087d69; cursor: help; }
                .note { display: none; position: absolute; top: 100%; left: 0;
                        width: 150px; transform: translateX(-65%); padding: 7px 9px;
                        border-radius: 6px; background: #18302b; color: white;
                        font: 12px/1.4 system-ui; }
                [tabindex]:hover .note, [tabindex]:focus .note { display: block; }
            """],
        )
        plot.title.visible = False
        card_styles = plot.styles
        plot.styles = {}
        individual.append(column(header, plot, spacing=0, sizing_mode="stretch_both",
                                 styles={**card_styles, "background": "white"}))


    grid = gridplot(
        individual, ncols=3, sizing_mode="stretch_both", toolbar_location="above",
        toolbar_options={"logo": None, "autohide": False},
    )
    grid.spacing = 8
    grid.rows = grid.cols = "1fr"
    for tool in grid.toolbar.tools:
        tool.syncable = False
    individual_plots.children = [grid]

    overlay = make_plot("Selected plots", "Model value", axis_range(x_bounds), y_bounds, combined=True)
    combined_renderers.clear()
    for spec in plot_specs:
        key, _, _, legend, _ = spec
        if legend:
            glyph = add_line(overlay, spec, combined=True)
            glyph.visible = previous.get(key, True)
            combined_renderers[key] = glyph
    legend = overlay.legend[0]
    legend.click_policy = "hide"
    legend.label_text_font_size = "11px"
    legend.border_line_color = None
    legend.background_fill_color = "#f6f9f8"
    legend.background_fill_alpha = 1
    overlay.center.remove(legend)
    overlay.add_layout(legend, "right")
    overlay.add_tools(make_hover(list(combined_renderers.values()), combined=True))
    combined_plot.children = [overlay]


def parse_fx(text):
    if not text:
        return []
    out = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            a, b = (int(float(endpoint)) for endpoint in token.split("-"))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(float(token))
    if not out:
        raise ValueError("Enter at least one fraction day or leave the field blank.")
    return sorted(set(out))


def parse_it_periods(text):
    if not text.strip():
        return None
    periods = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid IT window '{token}'. Use 'start-stop'.")
        start, stop = token.split("-")
        periods.append((float(start), float(stop)))
    return periods


def update():
    try:
        params = {key: float(widget.value) for key, widget in plan_inputs.items()}
        params["Astart"] = None if auto_a.active else float(a_start.value)
        a_start.disabled = auto_a.active
        st.disabled = use_lq.active
        survival = float(st.value)
        if use_lq.active:
            dose = lq_dose.value
            survival = float(np.exp(-(a_t.value * dose + b_t.value * dose**2)))
            st.value = round(survival, 4)

        result = rit_simulation(
            fx=parse_fx(fx_input.value), dose=float(d_l.value), ITperiods=parse_it_periods(it_input.value),
            ST=survival, fr=float(fr.value), parameters=params, g=float(g.value),
            kRadL=float(k_rad.value), radType=rad_type.value.lower(),
            aL=float(a_l.value), bL=float(b_l.value),
            d_max=float(d_max.value) if d_max.value >= 0 else np.inf,
            use_L2_exhaustion=exhaustion.active, return_dict=True,
        )
        values = {
            "primary": result["Tarr"] + result["TMarr"],
            "moribund": result["TMarr"], "abscopal": result["T2arr"],
            "tme": result["Larr"] + result["LMarr"], "moribund_tme": result["LMarr"],
            "abscopal_lymphocytes": result["LGarr"], "signal": result["Aarr"],
            "production": result["darr"], "integrated": result["imuteffarr"],
        }
        for key, source in sources.items():
            source.data = {"x": result["timearr"], "y": values[key]}
        status.text = '<span class="rit-success" role="status">Simulation updated</span>'
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        status.text = f'<span role="alert" style="color:#a43842">{escape(str(error))}</span>'


def on_input(attr, old, new):
    if not resetting:
        update()


def on_survival(attr, old, new):
    if not use_lq.active:
        on_input(attr, old, new)


def on_plot_option(attr, old, new):
    if not resetting:
        build_plots()


def reset():
    global resetting
    resetting = True
    try:
        for widget, prop, value in defaults:
            setattr(widget, prop, value)
        for glyph in combined_renderers.values():
            glyph.visible = True
    finally:
        resetting = False
    update()
    build_plots()


document = curdoc()
template_sections = []
defaults = []
resetting = False
for section_index, group in enumerate(sections):
    template_rows = []
    for row_index, fields in enumerate(group["rows"]):
        template_fields = []
        for field_index, (label, widget, help_text) in enumerate(fields):
            name = f"field_{section_index}_{row_index}_{field_index}"
            widget.name = name
            widget.margin = 0
            widget.stylesheets = [WIDGET_STYLE]
            if isinstance(widget, Switch):
                widget.stylesheets = [SWITCH_STYLE]
            else:
                widget.sizing_mode = "stretch_width"
            prop = "active" if isinstance(widget, Switch) else "value"
            defaults.append((widget, prop, getattr(widget, prop)))
            widget.on_change(prop, on_survival if widget is st else on_input)
            document.add_root(widget)
            template_fields.append({"name": name, "label": label, "help": help_text, "switch": prop == "active"})
        template_rows.append(template_fields)
    template_sections.append({"title": group["title"], "rows": template_rows})

for widget, prop in ((log_x, "active"), (log_y, "active"), (x_range_input, "value"), (y_range_input, "value")):
    defaults.append((widget, prop, getattr(widget, prop)))
    widget.on_change(prop, on_plot_option)

run_button = Button(
    label="Update simulation", button_type="success", sizing_mode="stretch_width", margin=0,
    stylesheets=["""
        :host .bk-btn-success { background: #087d69; border-color: #087d69; }
        :host .bk-btn-success:hover { background: #096457; }
    """],
)
reset_button = Button(label="Reset to defaults", sizing_mode="stretch_width", margin=0)
run_button.on_click(update)
reset_button.on_click(reset)
actions = column(run_button, reset_button, spacing=6, sizing_mode="stretch_width", name="actions")

update()
build_plots()
for layout in (actions, status, plot_controls, results):
    document.add_root(layout)
logo_path = TEMPLATE.parent / "assets" / "rit-simulator.png"
document.template_variables["logo_png"] = b64encode(logo_path.read_bytes()).decode("ascii")
document.template = TEMPLATE.read_text(encoding="utf-8")
document.template_variables["parameter_sections"] = template_sections
document.title = "RIT Simulator"
