import numpy as np
from scipy import stats

def compute(signal: np.ndarray, scale_min: int = 5, scale_max: int = 100, scale_res: int = 20):
    """
    Cálculo multifractal por DFA (Detrended Fluctuation Analysis).
    Retorna dicionário com escalas, flutuações e expoente alpha.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if N < scale_max:
        return {"scales": [], "fluctuations": [], "dfa_alpha": np.nan}

    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
    F = []

    # Perfil integrado
    y = np.cumsum(signal - np.mean(signal))

    for s in scales:
        if s >= N:
            continue

        shape = (N // s, s)
        X = np.reshape(y[:shape[0] * s], shape)
        RMS = []

        for segment in X:
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, deg=1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            RMS.append(np.sqrt(np.mean(detrended ** 2)))

        F.append(np.mean(RMS))

    if len(F) < 2:
        return {"scales": scales.tolist(), "fluctuations": F, "dfa_alpha": np.nan}

    log_scales = np.log10(scales[:len(F)])
    log_F = np.log10(F)
    alpha, intercept, _, _, _ = stats.linregress(log_scales, log_F)

    return {"scales": scales.tolist(), "fluctuations": F, "dfa_alpha": alpha}
