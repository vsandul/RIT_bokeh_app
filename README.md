# Radio-Immune Therapy Interactive Plots

This repository contains an interactive **Bokeh dashboard** for exploring tumor–immune dynamics in combined radio-immunotherapy (RIT).  
It allows you to change simulation parameters (tumor/immune kinetics, RT fractionation, immunotherapy timing, etc.) and immediately see their effect on model outputs.

---

## ⚠️ Disclaimer
This application is intended **for research and educational purposes only**.  
It is **not validated for clinical use** and must not be used to guide patient treatment decisions.  

---

## 🧩 Model Background

This dashboard implements and extends the **predictive biophysical model of combined radiotherapy and immunotherapy** developed by  
**T. Friedrich, M. Scholz, and M. Durante**:  

> *A Predictive Biophysical Model of the Combined Action of Radiation Therapy and Immunotherapy of Cancer*  
> International Journal of Radiation Oncology, Biology, Physics (IJROBP), 2022.  
> DOI: [10.1016/j.ijrobp.2022.03.030](https://doi.org/10.1016/j.ijrobp.2022.03.030)

### 🔧 Modifications in this implementation:
- The fraction of antigen-specific lymphocytes migrating to the **abscopal tumor** vs. the **primary tumor** is no longer fixed (previously assumed equal).  
  → Controlled by a new parameter **`g`**.  
- **Radiation-induced lymphocyte depletion** is now explicitly modeled for lymphocytes migrating to the abscopal site.  
  → Controlled by a new parameter **`kRad`**.  

These extensions enable more flexible exploration of abscopal dynamics and immune–RT interactions.

---

## 📊 Features
- Interactive **grid of plots** showing:
  - Tumor dynamics and signal (`Tarr`, `TMarr`, `T2arr`, `Aarr`)
  - Lymphocyte dynamics (`Larr`, `LMarr`, `LGarr`, `darr`, `imuteffarr`)
- Adjustable **plan constants** (α/β, amplification, fc1/4 parameters, etc.)
- Adjustable **treatment plan** (fractions, dose to LN, IT start/stop, fr, g, kRad)
- Option to compute tumor survival `ST` via **LQ model** or direct slider
- Switch between **Photon / Carbon RT mode**
- Reset button to restore defaults

---

## 🚀 Run Online (Binder)

Click the badge below to launch the interactive app directly in your browser (no installation required):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vsandul/RIT_bokeh_app/main?urlpath=proxy/5006/rit_bokeh_app)

⚠️ First launch may take several minutes while Binder builds the environment.

---

## 💻 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/vsandul/RIT_bokeh_app.git
   cd RIT_bokeh_app
   ```

2. Install requirements (preferably in a virtualenv or conda env):
   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:
   ```bash
   bokeh serve --show rit_bokeh_app.py
   ```
   This will open your browser at [http://localhost:5006/rit_bokeh_app](http://localhost:5006/rit_bokeh_app).

---

## 📂 Repository structure
```
.
├── rit_bokeh_app.py             # Main Bokeh dashboard application
├── simulationFunctions_Fast.py  # file with RIT model (fast vectorized calculations)
├── simulationFunctions.py       # file with RIT model (old, slow calculations)
├── requirements.txt             # Dependencies for Binder/local run
└── README.md                    # This file
```

---

## ⚙️ Requirements
- Python 3.8+
- Bokeh ≥ 3.0
- NumPy, SciPy

---

## 🧑‍🤝‍🧑 Credits
Developed as part of the **PhD project on radiation-induced lymphopenia and combined radio-immunotherapy modeling** at GSI Helmholtz Centre for Heavy Ion Research.  
Model based on **Friedrich et al. (2022)**, with modifications to include abscopal lymphocyte distribution (`g`) and radiation-induced lymphocyte depletion (`kRad`).

---

## 📜 License
Project is held and distributed under MIT License.
