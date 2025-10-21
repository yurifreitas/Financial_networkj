import numpy as np
from scipy import stats

def compute(signal, scale_min=5, scale_max=100, scale_res=20):
    N = len(signal)
    if N < scale_max:
        return {
            "scales": [],
            "fluctuations": [],
            "dfa_alpha": None
        }

    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
    F = []

    y = np.cumsum(signal - np.mean(signal))  # perfil integrado

    for s in scales:
        if s >= N:
            continue

        shape = (N // s, s)
        X = np.reshape(y[:shape[0]*s], shape)
        RMS = []

        for segment in X:
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, deg=1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            RMS.append(np.sqrt(np.mean(detrended ** 2)))

        F.append(np.mean(RMS))

    if len(F) < 2:
        return {
            "scales": scales.tolist(),
            "fluctuations": F,
            "dfa_alpha": None
        }

    log_scales = np.log10(scales[:len(F)])
    log_F = np.log10(F)
    alpha, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_F)

    return {
        "scales": scales.tolist(),
        "fluctuations": F,
        "dfa_alpha": alpha
    }
