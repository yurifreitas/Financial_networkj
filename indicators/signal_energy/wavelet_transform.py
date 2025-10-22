# =========================================================
# ⚡ EtherSym Finance — Wavelet Energy (UltraFast Parallel)
# =========================================================
# - Cálculo paralelo da energia média das componentes Wavelet
# - 100% Numba JIT + vetorizado (sem overhead do PyWavelets)
# - Suporte simbiótico a níveis múltiplos e estabilidade fractal
# =========================================================

import numpy as np
from numba import njit, prange
import pywt

# ---------------------------------------------------------
# 🧩 Extrator de energia paralelizado
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _wavelet_energy_numba(signal, lo_d, hi_d, level_max):
    """
    Cálculo simbiótico da energia em múltiplos níveis usando filtros da wavelet.
    """
    N = len(signal)
    energy_total = 0.0

    # buffers temporários
    data = signal.copy()

    for level in range(level_max):
        step = 2 ** (level + 1)
        n = len(data)
        if n < len(lo_d):
            break

        # Convoluções manuais
        approx = np.zeros(n // 2, dtype=np.float64)
        detail = np.zeros(n // 2, dtype=np.float64)

        for i in prange(n // 2):
            s_lo = 0.0
            s_hi = 0.0
            for k in range(len(lo_d)):
                j = (2 * i - k)
                if j < 0 or j >= n:
                    continue
                s_lo += data[j] * lo_d[k]
                s_hi += data[j] * hi_d[k]
            approx[i] = s_lo
            detail[i] = s_hi

        # Energia simbiótica de cada banda
        e_approx = np.mean(approx * approx)
        e_detail = np.mean(detail * detail)
        energy_total += e_approx + e_detail

        data = approx  # próxima iteração na escala coarse

    return energy_total


def compute(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    """
    ⚙️ Energia média das componentes wavelet — versão simbiótica paralelizada.
    """
    try:
        # 🔹 Coeficientes base do filtro da wavelet escolhida
        w = pywt.Wavelet(wavelet)
        lo_d, hi_d = np.array(w.dec_lo), np.array(w.dec_hi)

        series = np.asarray(series, dtype=np.float64)
        if len(series) < len(lo_d) * 2:
            return np.nan

        # 🚀 Paralelismo total
        energy = _wavelet_energy_numba(series, lo_d, hi_d, level)
        return float(energy)

    except Exception as e:
        print(f"[⚠️ WaveletEnergyFast] erro simbiótico: {e}")
        return np.nan
