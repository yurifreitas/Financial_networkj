# =========================================================
# ⚡ EtherSym Finance — Multifractal DFA (UltraFast Parallel)
# =========================================================
# - Implementa Detrended Fluctuation Analysis (DFA) acelerado com Numba
# - Substitui np.polyfit por regressão linear simbiótica (100% compatível)
# - Usa paralelismo total (CPU multi-core) e cálculo vetorizado
# =========================================================

import numpy as np
from numba import njit, prange
from math import log10


# =========================================================
# 🔹 Núcleo paralelizado
# =========================================================
@njit(parallel=True, fastmath=True)
def _dfa_parallel(y, scales):
    """
    Cálculo paralelo do RMS em cada escala (sem np.polyfit).
    """
    n_scales = len(scales)
    F = np.empty(n_scales, dtype=np.float64)

    for idx in prange(n_scales):
        s = scales[idx]
        N = len(y)
        if s >= N:
            F[idx] = np.nan
            continue

        n_segments = N // s
        rms_accum = 0.0
        count = 0

        for seg_idx in range(n_segments):
            start = seg_idx * s
            end = start + s
            segment = y[start:end]

            # 🔹 Regressão linear manual (sem np.polyfit)
            x = np.arange(s, dtype=np.float64)
            x_mean = np.mean(x)
            y_mean = np.mean(segment)

            cov_xy = np.mean((x - x_mean) * (segment - y_mean))
            var_x = np.mean((x - x_mean) ** 2)

            slope = cov_xy / (var_x + 1e-12)
            intercept = y_mean - slope * x_mean

            trend = slope * x + intercept
            detrended = segment - trend
            rms = np.sqrt(np.mean(detrended * detrended))
            rms_accum += rms
            count += 1

        F[idx] = rms_accum / max(count, 1)

    return F


# =========================================================
# 🔸 Interface simbiótica de alto nível
# =========================================================
def compute(signal: np.ndarray, scale_min: int = 5, scale_max: int = 100, scale_res: int = 20):
    """
    🌌 Multifractal DFA simbiótico ultra-paralelo.
    - Usa Numba JIT com paralelismo total.
    - Ideal para séries longas (N > 10.000).
    - Retorna dicionário com expoente α, escalas e flutuações.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if N < scale_max:
        return {"scales": [], "fluctuations": [], "dfa_alpha": np.nan}

    # Escalas log-distribuídas
    scales = np.logspace(log10(scale_min), log10(scale_max), num=scale_res).astype(np.int64)
    scales = np.unique(scales)

    # Perfil integrado
    y = np.cumsum(signal - np.mean(signal))

    print(f"🌀 Calculando MF-DFA com {len(scales)} escalas e N={N} pontos (modo UltraParallel)")

    # 🚀 Paralelismo total
    F = _dfa_parallel(y, scales)

    # Remove NaNs e zeros
    mask = np.isfinite(F) & (F > 0)
    if np.sum(mask) < 3:
        return {"scales": scales.tolist(), "fluctuations": F.tolist(), "dfa_alpha": np.nan}

    log_scales = np.log10(scales[mask])
    log_F = np.log10(F[mask])

    # 🔮 Regressão linear para extrair o expoente α
    A = np.vstack([log_scales, np.ones_like(log_scales)]).T
    alpha, intercept = np.linalg.lstsq(A, log_F, rcond=None)[0]

    return {
        "scales": scales.tolist(),
        "fluctuations": F.tolist(),
        "dfa_alpha": float(alpha)
    }
