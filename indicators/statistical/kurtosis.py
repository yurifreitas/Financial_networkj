# =========================================================
# ⚡ EtherSym Finance — Kurtosis (UltraFast Parallel)
# =========================================================
# - Curtose móvel paralelizada (Numba + fastmath)
# - Calcula forma da distribuição (achatamento/leptocurtose)
# - Compatível com features_env simbiótico
# =========================================================

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _rolling_kurtosis(x: np.ndarray, window: int) -> np.ndarray:
    N = len(x)
    result = np.full(N, np.nan, dtype=np.float64)

    for i in prange(window - 1, N):
        seg = x[i - window + 1 : i + 1]
        n = len(seg)
        mean = np.mean(seg)
        diff = seg - mean
        var = np.mean(diff ** 2)
        if var < 1e-12:
            result[i] = 0.0
            continue
        m4 = np.mean(diff ** 4)
        # Curtose de Pearson (fisher=False)
        result[i] = m4 / (var ** 2)
    return result


def compute(close: np.ndarray, window: int = 28) -> np.ndarray:
    """
    ⚙️ Curtose (forma da distribuição) — versão simbiótica paralelizada.
    Mede o grau de achatamento ou cauda pesada da série.
    """
    close = np.asarray(close, dtype=np.float64)
    if len(close) < window:
        return np.full(len(close), np.nan)

    print(f"📊 Calculando curtose simbiótica (janela={window}) [modo UltraParallel]")
    return _rolling_kurtosis(close, window)
