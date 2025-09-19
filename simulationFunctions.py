import numpy as np
from scipy.special import gamma, gammainc, gammaincc
from scipy.integrate import quad


########################################
########### HELPER FUNCTIONS ###########
########################################

def Ginc(a,x):
    return gammaincc(a,x)*gamma(a)
    
def df(k1, k2, k3, t):
    if t < 1/k1:
        t1 = (1 - (k1*t)**k3) * (1 - np.exp(-k2*t))
        t2 = 1/k1 + (-1 + np.exp(-(k2/k1)))/k2 - 1/(k1 + k1*k3)
        t3 = ((k2/k1)**-k3 * (gamma(1 + k3) - Ginc(1 + k3, k2/k1)))/k2
        return t1 / (t2 + t3)#, t1,t2,t3
    else:
        return 0

def idf(k1, k2, k3, t):
    if t == 0:
        return 1
    elif t < 1/k1:
        t1 = np.exp(-(k2/k1)) - np.exp(-k2*t) + np.exp(-(k2/k1))*k3 - np.exp(-k2*t)*k3
        t2 = (k2*k3)/k1 - k2*t - k2*k3*t + k2*t*(k1*t)**k3
        t3 = (k2/k1)**-k3*(1+k3)*Ginc(1+k3, k2/k1)
        t4 = (1+k3)*(k1*t)**k3*(k2*t)**-k3*Ginc(1+k3, k2*t)
        t5 = 1/k1 + (-1 + np.exp(-(k2/k1)))/k2 - 1/(k1 + k1*k3)
        t6 = ((k2/k1)**-k3*(gamma(1+k3) - Ginc(1+k3, k2/k1)))
        return (t1 + t2 - t3 + t4) / (k2*(1+k3)*(t5 + t6/k2))
    else:
        return 0
    
korr, err = quad(lambda t: df(1/2, 0.2, 20, t), 0, 2)


########################################
######### SIMULATION FUNCTIONS #########
########################################
# In total, 4 funtions (see below)

# 1. Primary + abscopal tumor, no kRad (damage to abscopal lymphocytes)
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def rit2(fx, dose, startIT, stopIT, ST, fr, parameters, g=0.5):
    # Extract variables from the dictionary
    a1 = parameters["a1"]
    b1 = parameters["b1"]
    a2 = parameters["a2"]
    b2 = parameters["b2"]
    amplification = parameters["amplification"]
    fc1base = parameters["fc1base"]
    fc1high = parameters["fc1high"]
    fc4base = parameters["fc4base"]
    fc4high = parameters["fc4high"]
    stepsize = parameters["stepsize"]
    maxtime = parameters["maxtime"]
    
    
    steps = int(np.ceil(maxtime / stepsize))
    Tarr = np.zeros(steps)
    TMarr = np.zeros(steps)
    T2arr = np.zeros(steps)
    Aarr = np.zeros(steps)
    Larr = np.zeros(steps)
    LGarr = np.zeros(steps)
    LMarr = np.zeros(steps)
    imuteffarr = np.zeros(steps)
    timearr = np.zeros(steps)
    
    darr = np.zeros(steps)

    Tstart = 1e5
    LStart = 100
    LGStart = 100

    mut1 = a1 / ((Tstart / 1e6) ** b1)
    mut2 = a2 / ((Tstart / 1e6) ** b2)

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7

    k1 = min(max(0.04 * dose, 1 / 7), 1 / 2)
    k3 = 2.8
    k2 = 0.2
    SL = max(np.exp(-0.2 * dose - 0.14 * dose ** 2), 0.001)
    SLn = max(np.exp(-0.2 * fr * dose - 0.14 * fr ** 2 * dose ** 2), 0.001)

    Tarr[0] = Tstart
    T2arr[0] = Tstart
    TMarr[0] = 0
    Larr[0] = LStart
    LGarr[0] = LGStart
    LMarr[0] = 0
    Aarr[0] = rho * (1 / (lam + mut1) + 1 / (lam + mut2)) * Tstart
    
    darr[0] = 0

    dfarr = np.array([df(0.5, 0.2, 20, k * stepsize) for k in range(steps)])
    timearr[0] = 0

    for i in range(1, steps):
        time = stepsize * (i - 1)
        timeeval = stepsize * (i - 2)
        fxeval = [t for t in fx if t <= timeeval]

        fc1 = fc1high if startIT < timeeval <= stopIT else fc1base
        fc4 = fc4high if startIT < timeeval <= stopIT else fc4base

        d = stepsize * sum(
            (
                (fc4base if stepsize * (i - 1 - k) < startIT or stepsize * (i - 1 - k) > stopIT else fc4high) / korr
            ) * 
            (SLn ** len([x for x in fx if x <= stepsize * (i - 1 - k)])) *
            Aarr[i - 1 - k] *
            dfarr[k]
            for k in range(i - 1)
        )
        
        darr[i] = d

        muteff = (
            1e-12 if Tarr[i - 1] == 0 and TMarr[i - 1] == 0 else
            a1 / ((Tarr[i - 1] + TMarr[i - 1]) / 1e6) ** b1
        )

        muteff2 = (
            1e-12 if T2arr[i - 1] == 0 else
            a2 / (T2arr[i - 1] / 1e6) ** b2
        )

        delT = stepsize * (
            muteff * Tarr[i - 1] -
            fc1 * (Larr[i - 1] + LMarr[i - 1])
        ) - sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        delTM = sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                (muteff * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        )

        delT2 = stepsize * (muteff2 * T2arr[i - 1] - fc1 * amplification * LGarr[i - 1])

        delA = stepsize * (
            -lam * Aarr[i - 1] +
            rho * (Tarr[i - 1] + T2arr[i - 1]) +
            psi * (
                sum(
                    (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                    np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                    df(k1, k2, k3, timeeval - fxeval[k])
                    for k in range(len(fxeval))
                ) if len(fxeval) > 0 else 0
            )
        )

        delL = stepsize * (
            mul * Larr[i - 1] + g * 2 * d - fc1 * Larr[i - 1]
        ) - sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        delLM = sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - SL) * Larr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(mul * (timeeval - fxeval[k])) *
                (mul * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        ) - stepsize * fc1 * LMarr[i - 1]

        delLG = stepsize * (mul * LGarr[i - 1] + (1 - g) * 2 * d)

        timearr[i] = time
        Tarr[i] = max(0, Tarr[i - 1] + delT)
        TMarr[i] = max(0, TMarr[i - 1] + delTM)
        T2arr[i] = max(0, T2arr[i - 1] + delT2)
        Aarr[i] = max(0, Aarr[i - 1] + delA)
        Larr[i] = max(0, Larr[i - 1] + delL)
        LMarr[i] = max(0, LMarr[i - 1] + delLM)
        LGarr[i] = max(0, LGarr[i - 1] + delLG)
        imuteffarr[i] = imuteffarr[i - 1] + stepsize * muteff

    return Tarr, TMarr, T2arr, Aarr, Larr, LMarr, LGarr, imuteffarr, timearr, darr



# 2. Only primary tumor, no abscopal site
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=
def rit1(fx, dose, startIT, stopIT, ST, fr, parameters, g=0.5):
    # Extract variables from the dictionary
    a1 = parameters["a1"]
    b1 = parameters["b1"]
    fc1base = parameters["fc1base"]
    fc1high = parameters["fc1high"]
    fc4base = parameters["fc4base"]
    fc4high = parameters["fc4high"]
    stepsize = parameters["stepsize"]
    maxtime = parameters["maxtime"]
    
    steps = int(np.ceil(maxtime / stepsize))

    # Initialize arrays
    Tarr = np.zeros(steps)
    TMarr = np.zeros(steps)
    Aarr = np.zeros(steps)
    Larr = np.zeros(steps)
    LMarr = np.zeros(steps)
    imuteffarr = np.zeros(steps)
    timearr = np.zeros(steps)
    
    darr = np.zeros(steps)

    # Initial conditions and parameters
    Tstart = 1e5
    LStart = 100
    mut = a1 / ((Tstart / 1e6) ** b1)
    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7
    k1 = min(max(0.04 * dose, 1 / 7), 1 / 2)
    k3 = 2.8
    k2 = 0.2
    SL = max(np.exp(-0.2 * dose - 0.14 * dose ** 2), 0.001)
    SLn = max(np.exp(-0.2 * fr * dose - 0.14 * fr ** 2 * dose ** 2), 0.001)

    # Initial values
    Tarr[0] = Tstart
    TMarr[0] = 0
    Larr[0] = LStart
    LMarr[0] = 0
    Aarr[0] = rho / (lam + mut) * Tstart
    dfarr = np.array([df(0.5, 0.2, 20, k * stepsize) for k in range(steps)])
    timearr[0] = 0
    
    darr[0] = 0

    # Main loop
    for i in range(1, steps):
        time = stepsize * (i - 1)
        timeeval = stepsize * (i - 2)
        fxeval = [t for t in fx if t <= timeeval]

        fc1 = fc1high if startIT < timeeval <= stopIT else fc1base
        fc4 = fc4high if startIT < timeeval <= stopIT else fc4base

        # Compute d
        d = stepsize * sum(
            (
                (fc4base if stepsize * (i - 1 - k) < startIT or stepsize * (i - 1 - k) > stopIT else fc4high) / korr
            ) *
            (SLn ** len([x for x in fx if x <= stepsize * (i - 1 - k)])) *
            Aarr[i - 1 - k] *
            dfarr[k]
            for k in range(i - 1)
        )
        darr[i] = d

        # Compute muteff
        muteff = (
            1e-12 if Tarr[i - 1] == 0 and TMarr[i - 1] == 0 else
            a1 / ((Tarr[i - 1] + TMarr[i - 1]) / 1e6) ** b1
        )

        # Compute delT
        delT = stepsize * (
            muteff * Tarr[i - 1] - fc1 * (Larr[i - 1] + LMarr[i - 1])
        ) - sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        # Compute delTM
        delTM = sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                (muteff * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        )

        # Compute delA
        delA = stepsize * (
            -lam * Aarr[i - 1] +
            rho * Tarr[i - 1] +
            psi * (
                sum(
                    (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                    np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                    df(k1, k2, k3, timeeval - fxeval[k])
                    for k in range(len(fxeval))
                ) if len(fxeval) > 0 else 0
            )
        )

        # Compute delL
        delL = stepsize * (
            mul * Larr[i - 1] + g * 2 * d - fc1 * Larr[i - 1]
        ) - sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        # Compute delLM
        delLM = sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - SL) * Larr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(mul * (timeeval - fxeval[k])) *
                (mul * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        ) - stepsize * fc1 * LMarr[i - 1]

        # Update arrays
        timearr[i] = time
        Tarr[i] = max(0, Tarr[i - 1] + delT)
        TMarr[i] = max(0, TMarr[i - 1] + delTM)
        Aarr[i] = max(0, Aarr[i - 1] + delA)
        Larr[i] = max(0, Larr[i - 1] + delL)
        LMarr[i] = max(0, LMarr[i - 1] + delLM)
        imuteffarr[i] = imuteffarr[i - 1] + stepsize * muteff

    return Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr


# 3. Only primary tumor, no abscopal site; 2 periods of immunotherapy
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def rit1moore(fx, dose, startIT, stopIT, startIT2, stopIT2, ST, fr, parameters, g=0.5):
    # Extract variables from the dictionary
    a1 = parameters["a1"]
    b1 = parameters["b1"]
    fc1base = parameters["fc1base"]
    fc1high = parameters["fc1high"]
    fc4base = parameters["fc4base"]
    fc4high = parameters["fc4high"]
    stepsize = parameters["stepsize"]
    maxtime = parameters["maxtime"]
    
    steps = int(np.ceil(maxtime / stepsize))

    # Initialize arrays
    Tarr = np.zeros(steps)
    TMarr = np.zeros(steps)
    Aarr = np.zeros(steps)
    Larr = np.zeros(steps)
    LMarr = np.zeros(steps)
    imuteffarr = np.zeros(steps)
    timearr = np.zeros(steps)
    
    darr = np.zeros(steps)

    # Initial conditions and parameters
    Tstart = 1e5
    LStart = 100
    mut = a1 / ((Tstart / 1e6) ** b1)
    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7
    k1 = min(max(0.04 * dose, 1 / 7), 1 / 2)
    k3 = 2.8
    k2 = 0.2
    SL = max(np.exp(-0.2 * dose - 0.14 * dose ** 2), 0.001)
    SLn = max(np.exp(-0.2 * fr * dose - 0.14 * fr ** 2 * dose ** 2), 0.001)

    # Initial values
    Tarr[0] = Tstart
    TMarr[0] = 0
    Larr[0] = LStart
    LMarr[0] = 0
    Aarr[0] = rho / (lam + mut) * Tstart
    dfarr = np.array([df(0.5, 0.2, 20, k * stepsize) for k in range(steps)])
    timearr[0] = 0
    
    darr[0] = 0

    # Main loop
    for i in range(1, steps):
        time = stepsize * (i - 1)
        timeeval = stepsize * (i - 2)
        fxeval = [t for t in fx if t <= timeeval]

        fc1 = (
            fc1high
            if (startIT < timeeval <= stopIT or startIT2 < timeeval <= stopIT2)
            else fc1base
        )
        fc4 = (
            fc4high
            if (startIT < timeeval <= stopIT or startIT2 < timeeval <= stopIT2)
            else fc4base
        )

        # Compute d
        d = stepsize * sum(
            (
                (fc4high if (startIT <= stepsize * (i - 1 - k) <= stopIT) or
                 (startIT2 <= stepsize * (i - 1 - k) <= stopIT2) else fc4base) / korr
            ) *
            (SLn ** len([x for x in fx if x <= stepsize * (i - 1 - k)])) *
            Aarr[i - 1 - k] *
            dfarr[k]
            for k in range(i - 1)
        )
        
        darr[i] = d

        # Compute muteff
        muteff = (
            1e-12 if Tarr[i - 1] == 0 and TMarr[i - 1] == 0 else
            a1 / ((Tarr[i - 1] + TMarr[i - 1]) / 1e6) ** b1
        )

        # Compute delT
        delT = stepsize * (
            muteff * Tarr[i - 1] - fc1 * (Larr[i - 1] + LMarr[i - 1])
        ) - sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        # Compute delTM
        delTM = sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                (muteff * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        )

        # Compute delA
        delA = stepsize * (
            -lam * Aarr[i - 1] +
            rho * Tarr[i - 1] +
            psi * (
                sum(
                    (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                    np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                    df(k1, k2, k3, timeeval - fxeval[k])
                    for k in range(len(fxeval))
                ) if len(fxeval) > 0 else 0
            )
        )

        # Compute delL
        delL = stepsize * (
            mul * Larr[i - 1] + g * 2 * d - fc1 * Larr[i - 1]
        ) - sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        # Compute delLM
        delLM = sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - SL) * Larr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(mul * (timeeval - fxeval[k])) *
                (mul * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        ) - stepsize * fc1 * LMarr[i - 1]

        # Update arrays
        timearr[i] = time
        Tarr[i] = max(0, Tarr[i - 1] + delT)
        TMarr[i] = max(0, TMarr[i - 1] + delTM)
        Aarr[i] = max(0, Aarr[i - 1] + delA)
        Larr[i] = max(0, Larr[i - 1] + delL)
        LMarr[i] = max(0, LMarr[i - 1] + delLM)
        imuteffarr[i] = imuteffarr[i - 1] + stepsize * muteff

    return Tarr, TMarr, Aarr, Larr, LMarr, imuteffarr, timearr, darr


# 4. Primary + abscopal tumor, modified to use kRad (damage to abscopal lymphocytes)
# Dose is the biological dose to lyphocytes in the TME; fr = D_LN / D_TME (bio, for lymphocytes)
# -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def rit2_modified(fx, dose, startIT, stopIT, ST, fr, parameters, g=0.5, kRadL=0., radType='photon', aL=0.2, bL=0.14):
    # Extract variables from the dictionary
    a1 = parameters["a1"]
    b1 = parameters["b1"]
    a2 = parameters["a2"]
    b2 = parameters["b2"]
    amplification = parameters["amplification"]
    fc1base = parameters["fc1base"]
    fc1high = parameters["fc1high"]
    fc4base = parameters["fc4base"]
    fc4high = parameters["fc4high"]
    stepsize = parameters["stepsize"]
    maxtime = parameters["maxtime"]
    
    
    steps = int(np.ceil(maxtime / stepsize))
    Tarr = np.zeros(steps)
    TMarr = np.zeros(steps)
    T2arr = np.zeros(steps)
    Aarr = np.zeros(steps)
    Larr = np.zeros(steps)
    LGarr = np.zeros(steps)
    LMarr = np.zeros(steps)
    imuteffarr = np.zeros(steps)
    timearr = np.zeros(steps)
    
    darr = np.zeros(steps)

    Tstart = 1e5
    LStart = 100
    LGStart = 100

    mut1 = a1 / ((Tstart / 1e6) ** b1)
    mut2 = a2 / ((Tstart / 1e6) ** b2)

    lam = 0.15
    rho = 0.15
    mul = -0.15
    psi = 7
    

    k1 = 1/2 if radType == 'carbon' else min(max(0.04 * dose, 1 / 7), 1 / 2)
    k2 = 0.2
    k3 = 2.8
    
    #aL = 0.2 # alpha of lymphocytes
    #bL = 0.14 # beta of lymphocytes
    SL = max(np.exp(-aL * dose - bL * dose ** 2), 0.001) # survival of lymphocytes in TME
    SLn = max(np.exp(-aL * (fr*dose) - bL * (fr*dose) ** 2), 0.001) # survival of lymphocytes in LN

    Tarr[0] = Tstart
    T2arr[0] = Tstart
    TMarr[0] = 0
    Larr[0] = LStart
    LGarr[0] = LGStart
    LMarr[0] = 0
    Aarr[0] = rho * (1 / (lam + mut1) + 1 / (lam + mut2)) * Tstart
    
    darr[0] = 0

    dfarr = np.array([df(0.5, 0.2, 20, k * stepsize) for k in range(steps)])
    timearr[0] = 0

    for i in range(1, steps):
        time = stepsize * (i - 1)
        timeeval = stepsize * (i - 2)
        fxeval = [t for t in fx if t <= timeeval]

        fc1 = fc1high if startIT < timeeval <= stopIT else fc1base
        fc4 = fc4high if startIT < timeeval <= stopIT else fc4base

        d = stepsize * sum(
            (
                (fc4base if stepsize * (i - 1 - k) < startIT or stepsize * (i - 1 - k) > stopIT else fc4high) / korr
            ) * 
            (SLn ** len([x for x in fx if x <= stepsize * (i - 1 - k)])) *
            Aarr[i - 1 - k] *
            dfarr[k]
            for k in range(i - 1)
        )
        
        darr[i] = d

        muteff = (
            1e-12 if Tarr[i - 1] == 0 and TMarr[i - 1] == 0 else
            a1 / ((Tarr[i - 1] + TMarr[i - 1]) / 1e6) ** b1
        )

        muteff2 = (
            1e-12 if T2arr[i - 1] == 0 else
            a2 / (T2arr[i - 1] / 1e6) ** b2
        )

        delT = stepsize * (
            muteff * Tarr[i - 1] -
            fc1 * (Larr[i - 1] + LMarr[i - 1])
        ) - sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        delTM = sum(
            (1 - ST) * Tarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                (muteff * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        )

        delT2 = stepsize * (muteff2 * T2arr[i - 1] - fc1 * amplification * LGarr[i - 1])

        delA = stepsize * (
            -lam * Aarr[i - 1] +
            rho * (Tarr[i - 1] + T2arr[i - 1]) +
            psi * (
                sum(
                    (1 - ST) * Tarr[int(np.floor(fxeval[k] / stepsize))] *
                    np.exp(imuteffarr[i - 1] - imuteffarr[int(np.floor(fxeval[k] / stepsize))]) *
                    df(k1, k2, k3, timeeval - fxeval[k])
                    for k in range(len(fxeval))
                ) if len(fxeval) > 0 else 0
            )
        )

        delL = stepsize * (
            mul * Larr[i - 1] + g * 2 * d - fc1 * Larr[i - 1]
        ) - sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        delLM = sum(
            (1 - SL) * Larr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        ) + stepsize * (
            sum(
                (1 - SL) * Larr[int(np.floor(fxeval[k] / stepsize))] *
                np.exp(mul * (timeeval - fxeval[k])) *
                (mul * idf(k1, k2, k3, timeeval - fxeval[k]) - df(k1, k2, k3, timeeval - fxeval[k]))
                for k in range(len(fxeval))
            ) if len(fxeval) > 0 else 0
        ) - stepsize * fc1 * LMarr[i - 1]

        delLG = stepsize * (mul * LGarr[i - 1] + (1 - g) * 2 * d) - sum(
            (kRadL) * LGarr[i - 1] if len(fx) > 0 and np.floor(fx[k] / stepsize) == i - 1 else 0
            for k in range(len(fx))
        )

        timearr[i] = time
        Tarr[i] = max(0, Tarr[i - 1] + delT)
        TMarr[i] = max(0, TMarr[i - 1] + delTM)
        T2arr[i] = max(0, T2arr[i - 1] + delT2)
        Aarr[i] = max(0, Aarr[i - 1] + delA)
        Larr[i] = max(0, Larr[i - 1] + delL)
        LMarr[i] = max(0, LMarr[i - 1] + delLM)
        LGarr[i] = max(0, LGarr[i - 1] + delLG)
        imuteffarr[i] = imuteffarr[i - 1] + stepsize * muteff

    return Tarr, TMarr, T2arr, Aarr, Larr, LMarr, LGarr, imuteffarr, timearr, darr

