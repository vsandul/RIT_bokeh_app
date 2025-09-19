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


# 1. Primary + abscopal tumor, no kRad (damage to abscopal lymphocytes)
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
import numpy as np
from typing import Dict, Tuple, Sequence, Union, Optional

def rit2_fast(
    fx: Sequence[float],
    dose: float,
    startIT: float,
    stopIT: float,
    ST: float,
    fr: float,
    parameters: Dict[str, float],
    g: float = 0.5,
    *,
    return_dict: bool = False,
    korr_override: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, np.ndarray]
]:
    """
    Vectorized/optimized version of rit2.
    Assumes df(...) and idf(...) accept NumPy arrays for t (or are wrapped/vectorized).
    Returns the same 10 arrays as rit2, or a dict when return_dict=True.
    """

    # ----- unpack parameters -----
    a1 = float(parameters["a1"])
    b1 = float(parameters["b1"])
    a2 = float(parameters["a2"])
    b2 = float(parameters["b2"])
    amplification = float(parameters["amplification"])
    fc1base = float(parameters["fc1base"])
    fc1high = float(parameters["fc1high"])
    fc4base = float(parameters["fc4base"])
    fc4high = float(parameters["fc4high"])
    stepsize = float(parameters["stepsize"])
    maxtime  = float(parameters["maxtime"])

    steps  = int(np.ceil(maxtime / stepsize))
    t_grid = np.arange(steps, dtype=np.float64) * stepsize

    # ----- state arrays -----
    Tarr = np.zeros(steps, dtype=np.float64)
    TMarr = np.zeros(steps, dtype=np.float64)
    T2arr = np.zeros(steps, dtype=np.float64)
    Aarr  = np.zeros(steps, dtype=np.float64)
    Larr  = np.zeros(steps, dtype=np.float64)
    LGarr = np.zeros(steps, dtype=np.float64)
    LMarr = np.zeros(steps, dtype=np.float64)
    imuteffarr = np.zeros(steps, dtype=np.float64)
    timearr = t_grid.copy()
    darr = np.zeros(steps, dtype=np.float64)

    # ----- constants / initials -----
    Tstart = 1e5
    LStart = 100.0
    LGStart = 100.0

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7.0

    k1 = min(max(0.04 * dose, 1.0/7.0), 0.5)
    k2 = 0.2
    k3 = 2.8

    # lymphocyte survival (fixed α=0.2, β=0.14 here to match your original)
    SL  = max(np.exp(-0.2 * dose              - 0.14 * dose**2          ), 0.001)
    SLn = max(np.exp(-0.2 * (fr * dose)       - 0.14 * (fr**2)*dose**2  ), 0.001)

    Tarr[0]  = Tstart
    T2arr[0] = Tstart
    TMarr[0] = 0.0
    Larr[0]  = LStart
    LGarr[0] = LGStart
    LMarr[0] = 0.0

    mut1 = a1 / ((Tstart/1e6) ** b1)
    mut2 = a2 / ((Tstart/1e6) ** b2)
    Aarr[0] = rho * (1.0/(lam + mut1) + 1.0/(lam + mut2)) * Tstart
    darr[0] = 0.0

    # korr (fall back to 1.0 if not present)
    _korr = float(korr_override) if korr_override is not None else float(globals().get("korr", 1.0))

    # ----- schedules on the grid -----
    # fc1 uses (startIT < t <= stopIT) at evaluation time (i-2)*stepsize
    fc1_sched = np.where((t_grid > startIT) & (t_grid <= stopIT), fc1high, fc1base)
    # fc4 inside d[i] uses inclusive [startIT, stopIT] at sample times j*stepsize
    fc4_sched = np.where((t_grid >= startIT) & (t_grid <= stopIT), fc4high, fc4base)

    # ----- fractions to grid -----
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

    # ----- precompute kernels (df/idf must be vectorized) -----
    df_fixed = df(0.5, 0.2, 20, t_grid)   # for d(t)
    df_k     = df(k1,  k2,  k3, t_grid)   # for memory terms
    idf_k    = idf(k1, k2,  k3, t_grid)

    # ----- main loop -----
    for i in range(1, steps):
        timeeval_idx = i - 2
        fc1 = fc1_sched[timeeval_idx] if timeeval_idx >= 0 else fc1base

        # ---- d[i] via dot product (align indices 1..i-1 with df_fixed[:i-1]) ----
        if i >= 2:
            # weights built from j = 1..i-1 (Aarr, fc4, SLn^nfx at those j)
            weights = (fc4_sched[1:i] / _korr) * (SLn ** nfx_until[1:i]) * Aarr[1:i]
            d_val = stepsize * np.dot(df_fixed[:i-1], weights[::-1])
        else:
            d_val = 0.0
        darr[i] = d_val

        # ---- mutational effects ----
        total_T = Tarr[i-1] + TMarr[i-1]
        muteff  = 1e-12 if total_T <= 0.0 else a1 / ((total_T / 1e6) ** b1)
        muteff2 = 1e-12 if T2arr[i-1] <= 0.0 else a2 / ((T2arr[i-1] / 1e6) ** b2)

        hits_now = frac_hits[i-1]

        # ---- delT ----
        delT = (stepsize * (muteff * Tarr[i-1] - fc1 * (Larr[i-1] + LMarr[i-1]))
                - hits_now * (1.0 - ST) * Tarr[i-1])

        # ---- past fractions (<= timeeval) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            have_past = np.any(mask_eval)
        else:
            have_past = False

        # ---- delTM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            tm_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac
                            * (muteff * idf_k[dt_idx] - df_k[dt_idx]))
            delTM = hits_now * (1.0 - ST) * Tarr[i-1] + stepsize * tm_sum
        else:
            delTM = hits_now * (1.0 - ST) * Tarr[i-1]

        # ---- delT2 ----
        delT2 = stepsize * (muteff2 * T2arr[i-1] - fc1 * amplification * LGarr[i-1])

        # ---- delA ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            a_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac * df_k[dt_idx])
        else:
            a_sum = 0.0
        delA = stepsize * (-lam * Aarr[i-1] + rho * (Tarr[i-1] + T2arr[i-1]) + psi * a_sum)

        # ---- delL ----
        delL = (stepsize * (mul * Larr[i-1] + g * 2.0 * d_val - fc1 * Larr[i-1])
                - hits_now * (1.0 - SL) * Larr[i-1])

        # ---- delLM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Larr_at = Larr[idxs]
            exp_mul = np.exp(mul * (dt_idx.astype(np.float64) * stepsize))
            lm_sum  = np.sum((1.0 - SL) * Larr_at * exp_mul
                             * (mul * idf_k[dt_idx] - df_k[dt_idx]))
        else:
            lm_sum = 0.0
        delLM = hits_now * (1.0 - SL) * Larr[i-1] + stepsize * lm_sum - stepsize * fc1 * LMarr[i-1]

        # ---- delLG (no kRadL term in rit2) ----
        delLG = stepsize * (mul * LGarr[i-1] + (1.0 - g) * 2.0 * d_val)

        # ---- update (non-negative clamp) ----
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

    
# 2. Only primary tumor, no abscopal site
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=
def rit1_fast(
    fx: Sequence[float],
    dose: float,
    startIT: float,
    stopIT: float,
    ST: float,
    fr: float,
    parameters: Dict[str, float],
    g: float = 0.5,
    *,
    return_dict: bool = False,
    korr_override: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, np.ndarray]
]:
    """
    Vectorized/optimized version of rit1.
    Assumes df(...) and idf(...) accept NumPy arrays for t (or are wrapped/vectorized).

    Returns the same 8 arrays as rit1 (or a dict when return_dict=True):
      Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr
    """
    # ----- unpack parameters -----
    a1 = float(parameters["a1"])
    b1 = float(parameters["b1"])
    fc1base = float(parameters["fc1base"])
    fc1high = float(parameters["fc1high"])
    fc4base = float(parameters["fc4base"])
    fc4high = float(parameters["fc4high"])
    stepsize = float(parameters["stepsize"])
    maxtime  = float(parameters["maxtime"])

    steps  = int(np.ceil(maxtime / stepsize))
    t_grid = np.arange(steps, dtype=np.float64) * stepsize

    # ----- state arrays -----
    Tarr = np.zeros(steps, dtype=np.float64)
    TMarr = np.zeros(steps, dtype=np.float64)
    Aarr  = np.zeros(steps, dtype=np.float64)
    Larr  = np.zeros(steps, dtype=np.float64)
    LMarr = np.zeros(steps, dtype=np.float64)
    imuteffarr = np.zeros(steps, dtype=np.float64)
    timearr = t_grid.copy()
    darr = np.zeros(steps, dtype=np.float64)

    # ----- constants / initials -----
    Tstart = 1e5
    LStart = 100.0

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7.0

    k1 = min(max(0.04 * dose, 1.0/7.0), 0.5)
    k2 = 0.2
    k3 = 2.8

    # lymphocyte survival (fixed α=0.2, β=0.14 to match your original)
    SL  = max(np.exp(-0.2 * dose              - 0.14 * dose**2          ), 0.001)
    SLn = max(np.exp(-0.2 * (fr * dose)       - 0.14 * (fr**2)*dose**2  ), 0.001)

    Tarr[0]  = Tstart
    TMarr[0] = 0.0
    Larr[0]  = LStart
    LMarr[0] = 0.0

    mut = a1 / ((Tstart/1e6) ** b1)
    Aarr[0] = rho / (lam + mut) * Tstart
    darr[0] = 0.0

    # korr (fall back to 1.0 if not present)
    _korr = float(korr_override) if korr_override is not None else float(globals().get("korr", 1.0))

    # ----- schedules on the grid -----
    # fc1 uses (startIT < t ≤ stopIT) at evaluation time (i-2)*stepsize
    fc1_sched = np.where((t_grid > startIT) & (t_grid <= stopIT), fc1high, fc1base)
    # fc4 inside d[i] uses inclusive [startIT, stopIT] at sample times j*stepsize
    fc4_sched = np.where((t_grid >= startIT) & (t_grid <= stopIT), fc4high, fc4base)

    # ----- fractions to grid -----
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

    # ----- precompute kernels (df/idf must be vectorized) -----
    df_fixed = df(0.5, 0.2, 20, t_grid)  # for d(t)
    df_k     = df(k1,  k2,  k3, t_grid)  # for memory terms
    idf_k    = idf(k1, k2,  k3, t_grid)

    # ----- main loop -----
    for i in range(1, steps):
        timeeval_idx = i - 2
        fc1 = fc1_sched[timeeval_idx] if timeeval_idx >= 0 else fc1base

        # ---- d[i] via dot product (align indices 1..i-1 with df_fixed[:i-1]) ----
        if i >= 2:
            weights = (fc4_sched[1:i] / _korr) * (SLn ** nfx_until[1:i]) * Aarr[1:i]
            d_val = stepsize * np.dot(df_fixed[:i-1], weights[::-1])
        else:
            d_val = 0.0
        darr[i] = d_val

        # ---- mutational effect ----
        total_T = Tarr[i-1] + TMarr[i-1]
        muteff  = 1e-12 if total_T <= 0.0 else a1 / ((total_T / 1e6) ** b1)

        hits_now = frac_hits[i-1]

        # ---- delT ----
        delT = (stepsize * (muteff * Tarr[i-1] - fc1 * (Larr[i-1] + LMarr[i-1]))
                - hits_now * (1.0 - ST) * Tarr[i-1])

        # ---- past fractions (≤ timeeval) mask ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            have_past = np.any(mask_eval)
        else:
            have_past = False

        # ---- delTM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            tm_sum  = np.sum((1.0 - ST) * Tarr_at * exp_fac
                             * (muteff * idf_k[dt_idx] - df_k[dt_idx]))
            delTM = hits_now * (1.0 - ST) * Tarr[i-1] + stepsize * tm_sum
        else:
            delTM = hits_now * (1.0 - ST) * Tarr[i-1]

        # ---- delA ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            a_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac * df_k[dt_idx])
        else:
            a_sum = 0.0
        delA = stepsize * (-lam * Aarr[i-1] + rho * Tarr[i-1] + psi * a_sum)

        # ---- delL ----
        delL = (stepsize * (mul * Larr[i-1] + g * 2.0 * d_val - fc1 * Larr[i-1])
                - hits_now * (1.0 - SL) * Larr[i-1])

        # ---- delLM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Larr_at = Larr[idxs]
            exp_mul = np.exp(mul * (dt_idx.astype(np.float64) * stepsize))
            lm_sum  = np.sum((1.0 - SL) * Larr_at * exp_mul
                             * (mul * idf_k[dt_idx] - df_k[dt_idx]))
        else:
            lm_sum = 0.0
        delLM = hits_now * (1.0 - SL) * Larr[i-1] + stepsize * lm_sum - stepsize * fc1 * LMarr[i-1]

        # ---- update (non-negative clamp) ----
        Tarr[i]       = max(0.0, Tarr[i-1] + delT)
        TMarr[i]      = max(0.0, TMarr[i-1] + delTM)
        Aarr[i]       = max(0.0, Aarr[i-1] + delA)
        Larr[i]       = max(0.0, Larr[i-1] + delL)
        LMarr[i]      = max(0.0, LMarr[i-1] + delLM)
        imuteffarr[i] = imuteffarr[i-1] + stepsize * muteff

    if return_dict:
        return {
            "Tarr": Tarr,
            "TMarr": TMarr,
            "Aarr": Aarr,
            "Larr": Larr,
            "LMarr": LMarr,
            "imuteffarr": imuteffarr,
            "timearr": timearr,
            "darr": darr,
        }
    else:
        return Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr

# 3. Only primary tumor, no abscopal site; 2 periods of immunotherapy
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def rit1moore_fast(
    fx: Sequence[float],
    dose: float,
    startIT: float,
    stopIT: float,
    startIT2: float,
    stopIT2: float,
    ST: float,
    fr: float,
    parameters: Dict[str, float],
    g: float = 0.5,
    *,
    return_dict: bool = False,
    korr_override: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, np.ndarray]
]:
    """
    Vectorized/optimized version of rit1moore (two IT windows).
    Assumes df(...) and idf(...) accept NumPy arrays for t (or are wrapped/vectorized).

    Returns (same order as original):
      Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr
      or a dict when return_dict=True.
    """
    # ----- unpack parameters -----
    a1 = float(parameters["a1"])
    b1 = float(parameters["b1"])
    fc1base = float(parameters["fc1base"])
    fc1high = float(parameters["fc1high"])
    fc4base = float(parameters["fc4base"])
    fc4high = float(parameters["fc4high"])
    stepsize = float(parameters["stepsize"])
    maxtime  = float(parameters["maxtime"])

    steps  = int(np.ceil(maxtime / stepsize))
    t_grid = np.arange(steps, dtype=np.float64) * stepsize

    # ----- state arrays -----
    Tarr = np.zeros(steps, dtype=np.float64)
    TMarr = np.zeros(steps, dtype=np.float64)
    Aarr  = np.zeros(steps, dtype=np.float64)
    Larr  = np.zeros(steps, dtype=np.float64)
    LMarr = np.zeros(steps, dtype=np.float64)
    imuteffarr = np.zeros(steps, dtype=np.float64)
    timearr = t_grid.copy()
    darr = np.zeros(steps, dtype=np.float64)

    # ----- constants / initials -----
    Tstart = 1e5
    LStart = 100.0

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7.0

    k1 = min(max(0.04 * dose, 1.0/7.0), 0.5)
    k2 = 0.2
    k3 = 2.8

    # lymphocyte survival (fixed α=0.2, β=0.14 to match your originals)
    SL  = max(np.exp(-0.2 * dose              - 0.14 * dose**2          ), 0.001)
    SLn = max(np.exp(-0.2 * (fr * dose)       - 0.14 * (fr**2)*dose**2  ), 0.001)

    Tarr[0]  = Tstart
    TMarr[0] = 0.0
    Larr[0]  = LStart
    LMarr[0] = 0.0

    mut = a1 / ((Tstart / 1e6) ** b1)
    Aarr[0] = rho / (lam + mut) * Tstart
    darr[0] = 0.0

    # korr (fall back to 1.0 if not present)
    _korr = float(korr_override) if korr_override is not None else float(globals().get("korr", 1.0))

    # ----- schedules on grid -----
    # fc1 at evaluation time (i-2)*stepsize uses (start < t ≤ stop) for either window
    fc1_mask = ((t_grid > startIT) & (t_grid <= stopIT)) | ((t_grid > startIT2) & (t_grid <= stopIT2))
    fc1_sched = np.where(fc1_mask, fc1high, fc1base)

    # fc4 at sample times j*stepsize uses inclusive [start, stop] for either window (matches your d[] condition)
    fc4_mask = ((t_grid >= startIT) & (t_grid <= stopIT)) | ((t_grid >= startIT2) & (t_grid <= stopIT2))
    fc4_sched = np.where(fc4_mask, fc4high, fc4base)

    # ----- fractions aligned to grid -----
    fx = np.asarray(fx, dtype=np.float64)
    fx.sort()
    if fx.size:
        fx_idx = np.floor(fx / stepsize).astype(int)
        valid  = (fx_idx >= 0) & (fx_idx < steps)
        fx_idx = fx_idx[valid]
    else:
        fx_idx = np.array([], dtype=int)

    frac_hits = np.zeros(steps, dtype=np.int32)
    if fx_idx.size:
        np.add.at(frac_hits, fx_idx, 1)
    nfx_until = np.cumsum(frac_hits)  # inclusive

    # ----- precompute kernels (df/idf must be vectorized) -----
    df_fixed = df(0.5, 0.2, 20, t_grid)  # for d(t)
    df_k     = df(k1,  k2,  k3, t_grid)  # for memory terms
    idf_k    = idf(k1, k2,  k3, t_grid)

    # ----- main loop -----
    for i in range(1, steps):
        timeeval_idx = i - 2
        fc1 = fc1_sched[timeeval_idx] if timeeval_idx >= 0 else fc1base

        # ---- d[i] via dot product (align indices 1..i-1 with df_fixed[:i-1]) ----
        if i >= 2:
            weights = (fc4_sched[1:i] / _korr) * (SLn ** nfx_until[1:i]) * Aarr[1:i]
            d_val = stepsize * np.dot(df_fixed[:i-1], weights[::-1])
        else:
            d_val = 0.0
        darr[i] = d_val

        # ---- mutational effect ----
        total_T = Tarr[i-1] + TMarr[i-1]
        muteff  = 1e-12 if total_T <= 0.0 else a1 / ((total_T / 1e6) ** b1)

        hits_now = frac_hits[i-1]

        # ---- delT ----
        delT = (stepsize * (muteff * Tarr[i-1] - fc1 * (Larr[i-1] + LMarr[i-1]))
                - hits_now * (1.0 - ST) * Tarr[i-1])

        # ---- past fractions (≤ timeeval) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            have_past = np.any(mask_eval)
        else:
            have_past = False

        # ---- delTM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            tm_sum  = np.sum((1.0 - ST) * Tarr_at * exp_fac
                             * (muteff * idf_k[dt_idx] - df_k[dt_idx]))
            delTM = hits_now * (1.0 - ST) * Tarr[i-1] + stepsize * tm_sum
        else:
            delTM = hits_now * (1.0 - ST) * Tarr[i-1]

        # ---- delA ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Tarr_at = Tarr[idxs]
            exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
            a_sum   = np.sum((1.0 - ST) * Tarr_at * exp_fac * df_k[dt_idx])
        else:
            a_sum = 0.0
        delA = stepsize * (-lam * Aarr[i-1] + rho * Tarr[i-1] + psi * a_sum)

        # ---- delL ----
        delL = (stepsize * (mul * Larr[i-1] + g * 2.0 * d_val - fc1 * Larr[i-1])
                - hits_now * (1.0 - SL) * Larr[i-1])

        # ---- delLM ----
        if have_past:
            idxs   = fx_idx[mask_eval]
            dt_idx = timeeval_idx - idxs
            Larr_at = Larr[idxs]
            exp_mul = np.exp(mul * (dt_idx.astype(np.float64) * stepsize))
            lm_sum  = np.sum((1.0 - SL) * Larr_at * exp_mul
                             * (mul * idf_k[dt_idx] - df_k[dt_idx]))
        else:
            lm_sum = 0.0
        delLM = hits_now * (1.0 - SL) * Larr[i-1] + stepsize * lm_sum - stepsize * fc1 * LMarr[i-1]

        # ---- update (non-negative clamp) ----
        Tarr[i]       = max(0.0, Tarr[i-1] + delT)
        TMarr[i]      = max(0.0, TMarr[i-1] + delTM)
        Aarr[i]       = max(0.0, Aarr[i-1] + delA)
        Larr[i]       = max(0.0, Larr[i-1] + delL)
        LMarr[i]      = max(0.0, LMarr[i-1] + delLM)
        imuteffarr[i] = imuteffarr[i-1] + stepsize * muteff

    if return_dict:
        return {
            "Tarr": Tarr,
            "TMarr": TMarr,
            "Aarr": Aarr,
            "Larr": Larr,
            "LMarr": LMarr,
            "imuteffarr": imuteffarr,
            "timearr": timearr,
            "darr": darr,
        }
    else:
        return Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr


# 4. Primary + abscopal tumor, modified to use kRad (damage to abscopal lymphocytes)
# Dose is the biological dose to lyphocytes in the TME; fr = D_LN / D_TME (bio, for lymphocytes)
def rit2_modified_fast(
    fx: Sequence[float],
    dose: float,
    startIT: float,
    stopIT: float,
    ST: float,
    fr: float,
    parameters: Dict[str, float],
    g: float = 0.5,
    kRadL: float = 0.0,
    radType: str = "photon",
    aL: float = 0.2,
    bL: float = 0.14,
    *,
    return_dict: bool = False,
    korr_override: Optional[float] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Dict[str, np.ndarray]
]:
    """
    Faster/cleaner rewrite with vectorized inner sums and precomputations.
    Keeps the original dynamics and outputs.

    Notes:
    - Assumes fraction times in `fx` lie on the same grid as `stepsize`.
    - If df/idf only accept scalars, wrap them with np.vectorize or pre-tabulate on the grid.
    """
    # ----- unpack parameters -----
    a1 = float(parameters["a1"])
    b1 = float(parameters["b1"])
    a2 = float(parameters["a2"])
    b2 = float(parameters["b2"])
    amplification = float(parameters["amplification"])
    fc1base = float(parameters["fc1base"])
    fc1high = float(parameters["fc1high"])
    fc4base = float(parameters["fc4base"])
    fc4high = float(parameters["fc4high"])
    stepsize = float(parameters["stepsize"])
    maxtime = float(parameters["maxtime"])

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
    Tstart = 1e5
    LStart = 100.0
    LGStart = 100.0

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7.0

    # k1 conditional on radType (as you had it)
    k1 = 0.5 if radType == "carbon" else min(max(0.04 * dose, 1.0 / 7.0), 0.5)
    k2 = 0.2
    k3 = 2.8

    # lymphocyte survival in TME/LN
    SL  = max(np.exp(-aL * dose        - bL * dose**2       ), 0.001)
    SLn = max(np.exp(-aL * (fr * dose) - bL * (fr * dose)**2), 0.001)

    # initial values
    Tarr[0] = Tstart
    T2arr[0] = Tstart
    TMarr[0] = 0.0
    Larr[0] = LStart
    LGarr[0] = LGStart
    LMarr[0] = 0.0

    mut1 = a1 / ((Tstart/1e6) ** b1)
    mut2 = a2 / ((Tstart/1e6) ** b2)
    Aarr[0] = rho * (1.0/(lam+mut1) + 1.0/(lam+mut2)) * Tstart
    darr[0] = 0.0

    # korr (fall back to 1.0 if not present)
    if korr_override is not None:
        _korr = float(korr_override)
    else:
        _korr = float(globals().get("korr", 1.0))

    # ----- precompute schedules on the grid -----
    fc1_sched = np.where((t_grid > startIT) & (t_grid <= stopIT), fc1high, fc1base)
    fc4_sched = np.where((t_grid > startIT) & (t_grid <= stopIT), fc4high, fc4base)

    # ----- precompute fraction indices and counts -----
    fx = np.asarray(fx, dtype=np.float64)
    fx.sort()
    if fx.size:
        fx_idx = np.floor(fx / stepsize).astype(int)
        # keep only those in range
        mask = (fx_idx >= 0) & (fx_idx < steps)
        fx_idx = fx_idx[mask]
        fx_in_grid = fx[mask]
    else:
        fx_idx = np.array([], dtype=int)
        fx_in_grid = np.array([], dtype=np.float64)

    # hits per time step and cumulative number of fractions up to each index
    frac_hits = np.zeros(steps, dtype=np.int32)
    if fx_idx.size:
        np.add.at(frac_hits, fx_idx, 1)
    nfx_until = np.cumsum(frac_hits)  # inclusive

    # ----- precompute df kernels on the grid for fast lookups -----
    # d(t): uses df(0.5, 0.2, 20, t) in your code
    # build once for integer multiples of stepsize
    df_fixed = df(0.5, 0.2, 20, t_grid)  # should be vectorized; if not, wrap with np.vectorize

    # kernels used with k1,k2,k3 for "past fraction" sums
    df_k = df(k1, k2, k3, t_grid)
    idf_k = idf(k1, k2, k3, t_grid)

    # ----- main loop -----
    for i in range(1, steps):
        # timeeval corresponds to (i-2)*stepsize in your code
        timeeval_idx = i - 2
        timeeval = timeeval_idx * stepsize

        # fc1 at this evaluation time (use base if timeeval < 0)
        fc1 = fc1_sched[timeeval_idx] if timeeval_idx >= 0 else fc1base

        # ---- compute d[i] via vectorized dot (convolution-like) ----
        # original:
        # sum over k=0..i-2 of df_fixed[k] * [ fc4(t_{i-1-k})/_korr * SLn**(num_fx_up_to(i-1-k)) * Aarr[i-1-k] ]
        if i >= 2:
            past_len = i - 1  # number of terms in the sum
            # values at indices j = 0 .. i-2
            j_slice = slice(0, i-1)
            weights = (fc4_sched[j_slice] / _korr) * (SLn ** nfx_until[j_slice]) * Aarr[j_slice]
            # align reversed weights with df[k]
            d_sum = np.dot(df_fixed[:past_len], weights[::-1])
            d_val = stepsize * d_sum
        else:
            d_val = 0.0
        darr[i] = d_val

        # ---- muteff terms ----
        # avoid division by zero by guarding with tiny floor as you did
        total_T = Tarr[i-1] + TMarr[i-1]
        if total_T <= 0.0:
            muteff = 1e-12
        else:
            muteff = a1 / ((total_T / 1e6) ** b1)

        if T2arr[i-1] <= 0.0:
            muteff2 = 1e-12
        else:
            muteff2 = a2 / ((T2arr[i-1] / 1e6) ** b2)

        # ---- instantaneous hits at current step (vectors instead of per-fx loops) ----
        hits_now = frac_hits[i-1]  # number of fractions at t_{i-1}

        # ---- delT ----
        delT = stepsize * (muteff * Tarr[i-1] - fc1 * (Larr[i-1] + LMarr[i-1])) \
               - hits_now * (1.0 - ST) * Tarr[i-1]

        # ---- delTM second part: sum over past fractions (<= timeeval) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs = fx_idx[mask_eval]                      # integer indices of past fx
                dt_idx = timeeval_idx - idxs                  # integer dt / stepsize
                Tarr_at = Tarr[idxs]
                # exp(imuteffarr[i-1] - imuteffarr[idxs])
                exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
                # kernels at integer dt
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

        # ---- delA (with past fraction sum using df only) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs = fx_idx[mask_eval]
                dt_idx = timeeval_idx - idxs
                Tarr_at = Tarr[idxs]
                exp_fac = np.exp(imuteffarr[i-1] - imuteffarr[idxs])
                dfv = df_k[dt_idx]
                a_sum = np.sum((1.0 - ST) * Tarr_at * exp_fac * dfv)
            else:
                a_sum = 0.0
        else:
            a_sum = 0.0

        delA = stepsize * (-lam * Aarr[i-1] + rho * (Tarr[i-1] + T2arr[i-1]) + psi * a_sum)

        # ---- delL ----
        delL = stepsize * (mul * Larr[i-1] + g * 2.0 * d_val - fc1 * Larr[i-1]) \
               - hits_now * (1.0 - SL) * Larr[i-1]

        # ---- delLM (with past fractions and mul kernel) ----
        if fx_idx.size and timeeval_idx >= 0:
            mask_eval = fx_idx <= timeeval_idx
            if np.any(mask_eval):
                idxs = fx_idx[mask_eval]
                dt_idx = timeeval_idx - idxs
                Larr_at = Larr[idxs]
                # exp(mul * (timeeval - fx)) ; timeeval - fx = dt_idx * stepsize
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
        delLG = stepsize * (mul * LGarr[i-1] + (1.0 - g) * 2.0 * d_val) \
                - hits_now * (kRadL) * LGarr[i-1]

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
