# Radio-Immune Therapy Interactive Dashboard

This repository contains an interactive **Bokeh application** for exploring tumor–immune dynamics in **combined radio-immunotherapy (RIT)**.  
The tool allows you to modify radiotherapy fractionation, immunotherapy schedules, and biophysical parameters — and instantly visualize model behavior.

---

## ⚠️ Disclaimer

This software is intended **only for research and educational purposes**.  
It is **not validated clinically** and must **not** be used to guide patient treatment decisions.

---

## 🧩 Scientific Background

This dashboard is based on the predictive biophysical model of combined radiotherapy and immunotherapy developed by:

> **T. Friedrich, M. Scholz, M. Durante**  
> *A Predictive Biophysical Model of the Combined Action of Radiation Therapy and Immunotherapy of Cancer*  
> *International Journal of Radiation Oncology, Biology, Physics* (IJROBP), 2022  
> DOI: https://doi.org/10.1016/j.ijrobp.2022.03.030

## 📊 Features

### Interactive visualization panels
Nine dynamic plots:

- Tumor populations: `Tarr`, `TMarr`, `T2arr`
- Antigen signal: `Aarr`
- Lymphocyte populations: `Larr`, `LMarr`, `LGarr`
- Damage signal: `darr`
- Immunity effectiveness: `imuteffarr`

### Fully interactive treatment design
- Radiotherapy fractionation scheme (`fx`)
- Dose to lymphocytes in TME (`d_L`)
- **Multiple immunotherapy intervals**
- Lymph node dose fraction (`fr`)
- Lymphocyte distribution parameter (`g`)
- Radiation-induced lymphocyte depletion (`kRad`)
- Optional **LQ model** for tumor survival `ST`
- Select **Photon** or **Carbon** RT mode
- Adjustable `a_L`, `b_L` for lymphocyte radiosensitivity

### Adjustable plan constants
- Tumor growth parameters (`a1`, `b1`, `a2`, `b2`)
- Antigen amplification
- fc1/fc4 immune activation parameters (PD-1/CTLA-4 parameters)
- Numerical ODE settings (`stepsize`, `maxtime`)
- **Initial values**: `T1start`, `T2start`, `Lstart`, `LGstart`
- **Antitumor lymphocyte production cap** `d_max` (0 = ∞)

---

## 🚀 Run Online (Binder)

Launch directly in your browser—no installation required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vsandul/RIT_bokeh_app/main?urlpath=proxy/5006/rit_bokeh_app)

> ⚠️ Binder builds may take several minutes on first launch.

---

## 💻 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/vsandul/RIT_bokeh_app.git
cd RIT_bokeh_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Bokeh server
```bash
bokeh serve --show rit_bokeh_app.py
```

Your browser will open automatically at:

**http://localhost:5006/rit_bokeh_app**

---

## 📂 Repository Structure

```
.
├── rit_bokeh_app.py            # Main Bokeh dashboard
├── simulationFunctions.py      # RIT model (reference version)
├── requirements.txt            # Dependencies for Binder/local execution
└── README.md                   # Documentation
```

---

## ⚙️ Requirements

- Python 3.8+
- Bokeh ≥ 3.0
- NumPy, SciPy

---

## 🧑‍🔬 Author & Context

Developed as part of a **PhD research project on radiation-induced lymphopenia and combined radio-immunotherapy modeling** at  
**GSI Helmholtz Centre for Heavy Ion Research**.

Model based on **Friedrich et al. (2022)** with added mechanisms for abscopal lymphocyte flow (`g`) and radiation-induced depletion (`kRad`).

---

## 📜 License

This project is released under the **MIT License**.
