import numpy as np
from scipy.special import gammainc, gammaincc, gammaln   # regularized lower/upper, and log Γ
from scipy.integrate import quad

from typing import Dict, Tuple, Sequence, Union, Optional


########################################
########### HELPER FUNCTIONS ###########
########################################

# Stable upper incomplete gamma Γ(a, x) = gammaincc(a, x) * Γ(a)
def Ginc(a, x):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.exp(gammaln(a)) * gammaincc(a, x)

def _Gamma(a):
    # Γ(a) = exp(gammaln(a)) (stable)
    return np.exp(gammaln(a))

def df(k1, k2, k3, t):
    """
    Vectorized df. Works with scalar or array t.
    Original piecewise:
      if t < 1/k1:  t1 / (t2 + t3)
      else: 0
    """
    # Cast to arrays for vectorization (scalars will still work)
    t  = np.asarray(t, dtype=float)
    k1 = float(k1); k2 = float(k2); k3 = float(k3)

    invk1 = 1.0 / k1
    ratio = k2 / k1

    # Precompute constants (independent of t)
    # t2
    t2 = invk1 + (np.exp(-ratio) - 1.0) / k2 - 1.0 / (k1 * (1.0 + k3))
    # t3 = (k1/k2)^k3 * (Γ(1+k3) - Ginc(1+k3, ratio)) / k2  = (k1/k2)^k3 * γ(1+k3, ratio) / k2
    Gamma_1k3   = _Gamma(1.0 + k3)
    Gupper_rat  = np.exp(gammaln(1.0 + k3)) * gammaincc(1.0 + k3, ratio)
    Glower_rat  = Gamma_1k3 - Gupper_rat  # lower incomplete gamma γ(a,x)
    t3 = (k1 / k2) ** k3 * Glower_rat / k2

    denom = t2 + t3  # constant

    out = np.zeros_like(t, dtype=float)
    mask = t < invk1
    if np.any(mask):
        tm = t[mask]
        # t1 = (1 - (k1*t)^k3) * (1 - exp(-k2*t))
        # use -expm1 for better small-argument accuracy: 1 - exp(-x) = -expm1(-x)
        t1 = (1.0 - (k1 * tm) ** k3) * (-np.expm1(-k2 * tm))
        out[mask] = t1 / denom
    return out if out.shape else float(out)  # return scalar when scalar in

def idf(k1, k2, k3, t):
    """
    Vectorized idf. Works with scalar or array t.
    Original piecewise:
      t == 0      -> 1
      0 < t < 1/k1: (t1 + t2 - t3 + t4) / (k2*(1+k3)*(t5 + t6/k2))
      t >= 1/k1   -> 0
    """
    t  = np.asarray(t, dtype=float)
    k1 = float(k1); k2 = float(k2); k3 = float(k3)

    invk1 = 1.0 / k1
    ratio = k2 / k1
    c_pow = (k1 / k2) ** k3

    # constants (no t)
    Gamma_1k3  = _Gamma(1.0 + k3)
    Gupper_rat = np.exp(gammaln(1.0 + k3)) * gammaincc(1.0 + k3, ratio)
    Glower_rat = Gamma_1k3 - Gupper_rat

    # t5 = 1/k1 + (exp(-k2/k1) - 1)/k2 - 1/(k1*(1+k3))
    t5 = invk1 + (np.exp(-ratio) - 1.0) / k2 - 1.0 / (k1 * (1.0 + k3))
    # t6 = (k1/k2)^k3 * (Γ(1+k3) - Ginc(1+k3, ratio)) = (k1/k2)^k3 * γ(1+k3, ratio)
    t6 = c_pow * Glower_rat

    denom_const = k2 * (1.0 + k3) * (t5 + t6 / k2)  # constant denominator

    out = np.zeros_like(t, dtype=float)

    # t == 0 -> 1
    mask0 = (t == 0.0)
    out[mask0] = 1.0

    # 0 < t < 1/k1
    mask = (t > 0.0) & (t < invk1)
    if np.any(mask):
        tm = t[mask]
        exp_neg_ratio = np.exp(-ratio)
        exp_neg_k2t   = np.exp(-k2 * tm)

        # Simplifications:
        # t1 = (1+k3)*(exp(-ratio) - exp(-k2*t))
        t1 = (1.0 + k3) * (exp_neg_ratio - exp_neg_k2t)

        # t2 = k2*(k3/k1 - t*(1+k3) + t*(k1*t)^k3)
        t2 = k2 * ( (k3 / k1) - tm*(1.0 + k3) + tm * (k1 * tm) ** k3 )

        # t3 = (k1/k2)^k3 * (1+k3) * Ginc(1+k3, ratio)   (upper incomplete gamma at ratio)  [constant]
        t3 = c_pow * (1.0 + k3) * Gupper_rat

        # t4 = (1+k3) * (k1/k2)^k3 * Ginc(1+k3, k2*t)    (upper incomplete gamma at k2*t)
        Gupper_k2t = np.exp(gammaln(1.0 + k3)) * gammaincc(1.0 + k3, k2 * tm)
        t4 = (1.0 + k3) * c_pow * Gupper_k2t

        num = t1 + t2 - t3 + t4
        out[mask] = num / denom_const

    # t >= 1/k1 remains 0
    return out if out.shape else float(out)


korr, err = quad(lambda tau: float(df(0.5, 0.2, 20, tau)), 0.0, 2.0)


########################################
######### SIMULATION FUNCTIONS #########
########################################

# Expect these to exist in your environment:
# - df(k1, k2, k3, t)  : kernel (must accept vector dt or be safe with np.vectorize)
# - idf(k1, k2, k3, t) : its integral (same note)
# - optionally a global 'korr' scalar; if absent we'll default to 1.0 here.

from typing import Sequence, Tuple, Dict, Optional, Union
import numpy as np

# Helper: normalize ITperiods into a list of (start, stop) tuples
def _normalize_it_periods(ITperiods):
    """
    Normalize ITperiods into a list of (start, stop) float tuples.

    Allowed forms:
      - None, [] or ()                      -> []
      - (start, stop) or [start, stop]      -> [(start, stop)]
      - [(start1, stop1), (start2, stop2), ...]
    """
    if ITperiods is None:
        return []

    if isinstance(ITperiods, (list, tuple)):
        if len(ITperiods) == 0:
            return []

        # Single interval as (start, stop) or [start, stop]
        if len(ITperiods) == 2 and not isinstance(ITperiods[0], (list, tuple)):
            return [(float(ITperiods[0]), float(ITperiods[1]))]

        # Otherwise: iterable of pairs
        intervals = []
        for item in ITperiods:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each IT period must be a pair (start, stop).")
            intervals.append((float(item[0]), float(item[1])))
        return intervals

    raise TypeError("ITperiods must be None, a pair (start, stop), or a sequence of such pairs.")

# Primary + abscopal tumor (optional), modified to use kRad (damage to abscopal lymphocytes)
# Dose is the biological dose to lyphocytes in the TME; fr = D_LN / D_TME (bio, for lymphocytes)
def rit_simulation(
    fx: Sequence[float],
    dose: float,
    ITperiods = None,
    ST: float = 0.0,
    fr: float = 1.0,
    parameters: Dict[str, float] = None,
    g: float = 0.5,
    kRadL: float = 0.0,
    radType: str = "photon",
    aL: float = 0.2,
    bL: float = 0.14,
    d_max: float = np.inf,
    *,
    return_dict: bool = False,
    korr_override: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, np.ndarray]
]:
    """
    General RIT simulator (primary + abscopal tumor) with:
      - Multiple immunotherapy (IT) windows via ITperiods.
      - Optional custom initial values T1start, T2start, Lstart, LGstart from `parameters`.
      - Cap on d(t) via d_max.
      - kRadL term for lymphocyte damage in abscopal compartment.

    ITperiods:
      * None or []                        -> no IT
      * (start, stop) or [start, stop]    -> single window
      * [(start1, stop1), (start2, stop2), ...] -> multiple windows

    One IT mask is used for both fc1 and fc4:
      it_mask = OR over all (start < t <= stop) or [start, stop] (see below).
    """
    if parameters is None:
        raise ValueError("`parameters` dict must be provided.")

    # ----- unpack parameters -----
    a1 = float(parameters["a1"])
    b1 = float(parameters["b1"])
    a2 = float(parameters["a2"])
    b2 = float(parameters["b2"])
    amplification = float(parameters["amplification"]) #Attenuation of lymphocyte infiltration in secondary tumor compared with primary
    fc1base = float(parameters["fc1base"])
    fc1high = float(parameters["fc1high"])
    fc4base = float(parameters["fc4base"])
    fc4high = float(parameters["fc4high"])
    stepsize = float(parameters["stepsize"])
    maxtime = float(parameters["maxtime"])
    
    # Signal parameters (rho, lambda, psi)
    lam = float(parameters.get("lam", 0.15))
    rho = float(parameters.get("rho", 0.15))
    psi = float(parameters.get("psi", 7.0))
    
    # Lymphocyte natural decay
    mul = float(parameters.get("muL", -0.15))
    

    # Optional starts
    T1start = float(parameters.get("T1start", 1e5))
    T2start = float(parameters.get("T2start", 1e5))
    Lstart  = float(parameters.get("Lstart", 100.0))
    LGstart = float(parameters.get("LGstart", 100.0))

    steps = int(np.ceil(maxtime / stepsize))
    t_grid = np.arange(steps, dtype=np.float64) * stepsize

    # ----- arrays -----
    Tarr = np.zeros(steps, dtype=np.float64)
    TMarr = np.zeros(steps, dtype=np.float64)
    T2arr = np.zeros(steps, dtype=np.float64)
    Aarr = np.zeros(steps, dtype=np.float64)
    Larr = np.zeros(steps, dtype=np.float64)
    LGarr = np.zeros(steps, dtype=np.float64)
    LMarr = np.zeros(steps, dtype=np.float64)
    imuteffarr = np.zeros(steps, dtype=np.float64)
    timearr = t_grid.copy()
    darr = np.zeros(steps, dtype=np.float64)

    # ----- constants / initial conditions -----


    # k1 conditional on radType
    k1 = 0.5 if radType.lower() == "carbon" else min(max(0.04 * dose, 1.0 / 7.0), 0.5)
    k2 = 0.2
    k3 = 2.8

    # Lymphocyte survival in TME/LN
    SL  = max(np.exp(-aL * dose        - bL * dose**2       ), 0.001)
    SLn = max(np.exp(-aL * (fr * dose) - bL * (fr * dose)**2), 0.001)

    # initial values
    Tarr[0]  = T1start if T1start > 0 else 0.
    T2arr[0] = T2start if T2start > 0 else 0.
    TMarr[0] = 0.0
    Larr[0]  = Lstart if Lstart > 0 else 0.
    LGarr[0] = LGstart if LGstart > 0 else 0.
    LMarr[0] = 0.0

    mut1 = a1 / ((T1start/1e6) ** b1) if T1start > 0 else 0.
    mut2 = a2 / ((T2start/1e6) ** b2) if T2start > 0 else 0.
    Aarr[0] = rho/(lam+mut1)*T1start + rho/(lam+mut2)*T2start
    darr[0] = 0.0

    # korr (fall back to 1.0 if not present)
    if korr_override is not None:
        _korr = float(korr_override)
    else:
        _korr = float(globals().get("korr", 1.0))

    # ----- IT schedules on the grid -----
    it_intervals = _normalize_it_periods(ITperiods)

    if not it_intervals:
        # No IT: schedules are constant
        fc1_sched = np.full(steps, fc1base, dtype=np.float64)
        fc4_sched = np.full(steps, fc4base, dtype=np.float64)
    else:
        intervals_arr = np.array(it_intervals, dtype=np.float64)  # (N, 2)
        starts = intervals_arr[:, 0][:, None]  # (N, 1)
        stops  = intervals_arr[:, 1][:, None]  # (N, 1)
        t2d    = t_grid[None, :]               # (1, steps)

        # Single mask: you can choose (start < t <= stop) or [start, stop].
        # Here: (start < t <= stop):
        it_mask_all = (t2d > starts) & (t2d <= stops)
        it_mask = np.any(it_mask_all, axis=0)

        fc1_sched = np.where(it_mask, fc1high, fc1base)
        fc4_sched = np.where(it_mask, fc4high, fc4base)

    # ----- fraction indices and counts -----
    fx = np.asarray(fx, dtype=np.float64)
    fx.sort()
    if fx.size:
        fx_idx = np.floor(fx / stepsize).astype(int)
        mask = (fx_idx >= 0) & (fx_idx < steps)
        fx_idx = fx_idx[mask]
    else:
        fx_idx = np.array([], dtype=int)

    frac_hits = np.zeros(steps, dtype=np.int32)
    if fx_idx.size:
        np.add.at(frac_hits, fx_idx, 1)
    nfx_until = np.cumsum(frac_hits)  # inclusive

    # ----- df kernels on the grid -----
    df_fixed = df(0.5, 0.2, 20, t_grid)  # for d(t)
    df_k     = df(k1,  k2,  k3, t_grid)
    idf_k    = idf(k1, k2,  k3, t_grid)

    # ----- main loop -----
    for i in range(1, steps):
        timeeval_idx = i - 2

        # fc1 at this evaluation time (use base if timeeval < 0)
        fc1 = fc1_sched[timeeval_idx] if timeeval_idx >= 0 else fc1base

        # ---- d[i] via vectorized dot (convolution-like) ----
        if i >= 2:
            past_len = i - 1
            j_slice = slice(0, i-1)
            weights = (fc4_sched[j_slice] / _korr) * (SLn ** nfx_until[j_slice]) * Aarr[j_slice]
            d_sum = np.dot(df_fixed[:past_len], weights[::-1])
            d_val = stepsize * d_sum
        else:
            d_val = 0.0

        # Cap d(t)
        if d_val > d_max:
            d_val = d_max
        darr[i] = d_val

        # ---- muteff terms ----
        total_T = Tarr[i-1] + TMarr[i-1]
        if total_T <= 0.0:
            muteff = 1e-12
        else:
            muteff = a1 / ((total_T / 1e6) ** b1)

        if T2arr[i-1] <= 0.0:
            muteff2 = 1e-12
        else:
            muteff2 = a2 / ((T2arr[i-1] / 1e6) ** b2)

        hits_now = frac_hits[i-1]

        # ---- delT ----
        delT = (stepsize * (muteff * Tarr[i-1] - fc1 * (Larr[i-1] + LMarr[i-1]))
                - hits_now * (1.0 - ST) * Tarr[i-1])

        # ---- delTM: sum over past fractions (<= timeeval) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs   = fx_idx[mask_eval]
                dt_idx = timeeval_idx - idxs
                Tarr_at = Tarr[idxs]
                exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
                #exp_fac = np.exp(imuteffarr[timeeval_idx] - imuteffarr[idxs])
                dfv  = df_k[dt_idx]
                idfv = idf_k[dt_idx]
                tm_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac * (muteff * idfv - dfv))
                delTM = hits_now * (1.0 - ST) * Tarr[i-1] + stepsize * tm_sum
            else:
                delTM = hits_now * (1.0 - ST) * Tarr[i-1]
        else:
            delTM = hits_now * (1.0 - ST) * Tarr[i-1]

        # ---- delT2 ----
        delT2 = stepsize * (muteff2 * T2arr[i-1] - fc1 * amplification * LGarr[i-1])

        # ---- delA ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs = fx_idx[mask_eval]
                dt_idx = timeeval_idx - idxs
                Tarr_at = Tarr[idxs]
                exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
                #exp_fac = np.exp(imuteffarr[timeeval_idx] - imuteffarr[idxs])
                dfv = df_k[dt_idx]
                a_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac * dfv)
            else:
                a_sum = 0.0
        else:
            a_sum = 0.0

        delA = stepsize * (-lam * Aarr[i-1] + rho * (Tarr[i-1] + T2arr[i-1]) + psi * a_sum)

        # ---- delL ----
        #delL = (stepsize * (mul * Larr[i-1] + g * 2.0 * d_val - fc1 * Larr[i-1]) - hits_now * (1.0 - SL) * Larr[i-1])
        delL = (stepsize * ((mul - fc1) * (Larr[i-1] - Lstart) + g * 2.0 * d_val ) - hits_now * (1.0 - SL) * Larr[i-1])
                

        # ---- delLM ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs = fx_idx[mask_eval]
                dt_idx = timeeval_idx - idxs
                Larr_at = Larr[idxs]
                exp_mul = np.exp(mul * (dt_idx.astype(np.float64) * stepsize))
                dfv  = df_k[dt_idx]
                idfv = idf_k[dt_idx]
                lm_sum = np.sum((1.0 - SL) * Larr_at * exp_mul * (mul * idfv - dfv))
            else:
                lm_sum = 0.0
        else:
            lm_sum = 0.0

        delLM = hits_now * (1.0 - SL) * Larr[i-1] + stepsize * lm_sum - stepsize * fc1 * LMarr[i-1]

        # ---- delLG ----
        #delLG = stepsize * (mul * LGarr[i-1] + (1.0 - g) * 2.0 * d_val) - hits_now * kRadL * LGarr[i-1]
        delLG = stepsize * (mul * (LGarr[i-1] - LGstart) + (1.0 - g) * 2.0 * d_val) - hits_now * kRadL * LGarr[i-1]

        # ---- update state (non-negativity clamp) ----
        Tarr[i]       = max(0.0, Tarr[i-1] + delT)
        TMarr[i]      = max(0.0, TMarr[i-1] + delTM)
        T2arr[i]      = max(0.0, T2arr[i-1] + delT2)
        Aarr[i]       = max(0.0, Aarr[i-1] + delA)
        Larr[i]       = max(0.0, Larr[i-1] + delL)
        LMarr[i]      = max(0.0, LMarr[i-1] + delLM)
        LGarr[i]      = max(0.0, LGarr[i-1] + delLG)
        imuteffarr[i] = imuteffarr[i-1] + stepsize * muteff

    if return_dict:
        return {
            "Tarr": Tarr,
            "TMarr": TMarr,
            "T2arr": T2arr,
            "Aarr": Aarr,
            "Larr": Larr,
            "LMarr": LMarr,
            "LGarr": LGarr,
            "imuteffarr": imuteffarr,
            "timearr": timearr,
            "darr": darr,
        }
    else:
        return Tarr, TMarr, T2arr, Aarr, Larr, LMarr, LGarr, imuteffarr, timearr, darr

