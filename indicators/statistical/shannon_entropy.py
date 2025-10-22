# =========================================================
# ⚡ EtherSym Finance — Shannon Entropy (UltraFast Parallel)
# =========================================================
# - Cálculo de entropia de Shannon móvel totalmente paralelo (Numba)
# - Usa histogramas densos vetorizados e precomputação por bins
# - Escala linear com paralelismo total (100% CPU)
# =========================================================

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _entropy_numba(close, window, bins, min_count=1e-10):
    N = len(close)
    result = np.full(N, np.nan, dtype=np.float64)

    # Criação de hist arrays temporários
    for i in prange(window - 1, N):
        segment = close[i - window + 1 : i + 1]
        mn = np.min(segment)
        mx = np.max(segment)

        if mx == mn:
            result[i] = 0.0
            continue

        # 🧩 Binning manual para evitar overhead do np.histogram
        hist = np.zeros(bins, dtype=np.float64)
        bin_width = (mx - mn) / bins

        for j in range(window):
            idx = int((segment[j] - mn) / bin_width)
            if idx >= bins:
                idx = bins - 1
            hist[idx] += 1.0

        hist /= (window + 1e-10)
        hist += min_count  # estabilidade numérica

        # 🔮 Shannon Entropy
        result[i] = -np.sum(hist * np.log2(hist))

    return result


def compute(close: np.ndarray, window: int = 14, bins: int = 20) -> np.ndarray:
    """
    Cálculo simbiótico de entropia de Shannon móvel (100% paralelizado).
    """
    close = np.asarray(close, dtype=np.float64)
    N = len(close)

    if N < window:
        return np.full(N, np.nan, dtype=np.float64)

    print(f"🧠 Calculando Shannon Entropy ({window}) com {bins} bins — modo UltraParallel")

    H = _entropy_numba(close, window, bins)
    H = np.clip(H, 0.0, np.log2(bins))  # limite teórico máximo da entropia
    H = _smooth(H, alpha=0.12)
    return H


def _smooth(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Suavização simbiótica exponencial."""
    y = np.copy(x)
    for i in range(1, len(y)):
        if np.isnan(y[i]) or np.isnan(y[i-1]):
            continue
        y[i] = alpha * y[i] + (1 - alpha) * y[i-1]
    return y
