# Radio-Immune Therapy Interactive Plots

This repository contains an interactive **Bokeh dashboard** for exploring tumor–immune dynamics in combined radio-immunotherapy (RIT).  
It allows you to change simulation parameters (tumor/immune kinetics, RT fractionation, immunotherapy timing, etc.) and immediately see their effect on model outputs.

---

## 📊 Features
- Interactive **grid of plots** showing:
  - Tumor dynamics and signal (`Tarr`, `TMarr`, `T2arr`,`Aarr`)
  - Lymphocyte dynamics (, `Larr`, `LMarr`, `LGarr`, `darr`, `imuteffarr`)
- Adjustable **plan constants** (α/β, amplification, fc1/4 parameters, etc.)
- Adjustable **treatment plan** (fractions, dose to LN, IT start/stop, fr, g, kRad)
- Option to compute tumor survival `ST` via **LQ model** or direct slider
- Switch between **Photon / Carbon RT mode**
- Reset button to restore defaults

---

## 🚀 Run Online (Binder)

Click the badge below to launch the interactive app directly in your browser (no installation required):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vsandul/RIT_bokeh-app/main?urlpath=proxy/5006/rit_bokeh_app)

⚠️ First launch may take several minutes while Binder builds the environment.

---

## 💻 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/vsandul/RIT-bokeh-app.git
   cd RIT-bokeh-app
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

---

## 📜 License
MIT License (or specify your own).
