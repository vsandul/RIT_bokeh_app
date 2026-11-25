# RIT Simulator

An interactive Bokeh dashboard for exploring a biophysical model of combined radiotherapy and immunotherapy (RIT). Change the treatment schedule or model parameters to simulate the response of a primary tumor and an abscopal tumor.

**For research and education only.** The simulator is not clinically validated and must not be used to make treatment decisions.

## Run locally

Use Python 3.10 or later. The commands below use a virtual environment on macOS or Linux.

```bash
git clone https://github.com/vsandul/RIT_bokeh_app.git
cd RIT_bokeh_app
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m bokeh serve --show rit_bokeh_app.py
```

If your browser does not open automatically, go to [localhost:5006/rit_bokeh_app](http://localhost:5006/rit_bokeh_app). Stop the server with **Ctrl+C**. Using `python -m bokeh` runs Bokeh from the same environment as the installed dependencies. The requirements file pins Bokeh 3.5.1.

## Use the dashboard

The left panel groups simulation inputs by purpose. Hover over **ⓘ** beside a field for its meaning and accepted values. Changes recalculate the simulation as you enter them. Press **Enter** to apply a text field, use **Update simulation** to rerun explicitly, or choose **Reset to defaults** to restore the original inputs and plot settings.

### Set up a simulation

| Section | What to enter |
| --- | --- |
| Treatment plan | Enter radiation days as a comma-separated list with optional inclusive integer ranges, such as `1,3,5-9`. Enter immunotherapy windows as `start-stop` pairs separated by commas, such as `1-20,25-30`. Immunotherapy applies for `start < t ≤ stop`. Leave either schedule blank to omit that treatment. Choose Photon or Carbon, the lymphocyte dose per fraction, and the tumor survival fraction. |
| Radiosensitivity & depletion | Adjust lymphocyte survival coefficients, the dose fraction reaching draining lymph nodes, the share of recruited lymphocytes allocated to the primary tumor microenvironment (TME), and abscopal lymphocyte depletion. The optional **L₂ exhaustion** switch adds a separate abscopal loss term. |
| Tumor survival (LQ) | Turn on **Use LQ for Sₜ** to calculate tumor survival from the linear-quadratic coefficients and the adjacent dose field. With the switch off, the treatment plan supplies the survival fraction directly. |
| Initial conditions | Set starting cell counts for both sites. **Auto Aₛₜₐᵣₜ** computes the initial immune signal by default. Turn it off to set the initial signal manually. The optional production cap applies to `d(t)`. |
| Tumor growth & checkpoints | Set the growth coefficients for both tumors and the baseline and on-treatment levels for tumor killing and lymphocyte replenishment. |
| Immune signal | Adjust signal production, decay, and amplification, along with baseline lymphocyte growth or loss. |
| Simulation time | Set the time step and simulation horizon. The kernel currently starts at day 0 regardless of the displayed minimum-time field. |

### Read the plots

**Individual plots** shows nine outputs in a 3×3 grid with a linked time axis and crosshair.

| Row | Left | Center | Right |
| --- | --- | --- | --- |
| Tumor cells | Primary tumor, T₁ + Tₘ | Moribund tumor, Tₘ | Abscopal tumor, T₂ |
| Lymphocyte counts | TME lymphocytes, L₁ + Lₘ | Moribund TME lymphocytes, Lₘ | Abscopal lymphocytes, L₂ |
| Signal and rates | Immune signal, A | Lymphocyte production rate, d(t) | Integrated primary growth rate, the dimensionless time integral of μ₁ |

**Selected plots** overlays eight series with a clickable legend. The integrated growth coefficient is available in **Individual plots**.

Use **Log X** and **Log Y** to change the scale. Enter a manual axis range as `min,max`, such as `1,60` or `1e2,1e8`, then press **Enter**. Clear a field to restore automatic scaling. Logarithmic ranges require two positive bounds.

## Reference model

Friedrich, T., Scholz, M., and Durante, M. (2022). [A Predictive Biophysical Model of the Combined Action of Radiation Therapy and Immunotherapy of Cancer](https://doi.org/10.1016/j.ijrobp.2022.03.030). *International Journal of Radiation Oncology, Biology, Physics*, 113(4), 872–884.

The last plot reports the accumulated integral of the primary tumor growth coefficient μ₁ from the start of the simulation. Differences in that integral give the exponent used for post-fraction growth in equations (2) and (4) of the paper. Equation (8) defines the separate lymphocyte production rate d(t).

## Repository layout

- [`rit_bokeh_app.py`](rit_bokeh_app.py) configures Bokeh widgets, plots, and callbacks.
- [`rit_dashboard.html`](rit_dashboard.html) defines the dashboard layout and parameter help.
- [`simulationFunctions.py`](simulationFunctions.py) contains the numerical model and the `rit_simulation` function.
- [`requirements.txt`](requirements.txt) lists the Python dependencies.

## License

[MIT](LICENSE).
