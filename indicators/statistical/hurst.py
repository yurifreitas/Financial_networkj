# =========================================================
# 🌌 EtherSym Finance — Hurst Exponent (R/S Analysis)
# =========================================================
# - Implementa cálculo estável e vetorizado do expoente de Hurst
# - Inclui fallback para variâncias pequenas e ruído simbiótico leve
# - Retorna array completo, com NaN apenas nas regiões sem janela
# =========================================================

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def compute(close: np.ndarray, window: int = 100, min_var: float = 1e-8) -> np.ndarray:
    """
    Calcula o expoente de Hurst local (R/S) com janela deslizante.
    Retorna um vetor de mesma dimensão que `close`.

    Parâmetros:
    -----------
    close : np.ndarray
        Série de preços ou retornos.
    window : int
        Tamanho da janela deslizante.
    min_var : float
        Variância mínima para evitar divisão por zero.

    Retorna:
    --------
    np.ndarray
        Vetor com o Hurst local (NaN nas bordas).
    """
    N = len(close)
    if N < window:
        return np.full(N, np.nan, dtype=float)

    # 🌀 Janela deslizante vetorizada (sem loops lentos)
    windows = sliding_window_view(close, window)  # shape: (N-window+1, window)

    # Subtrai média e calcula cumulativa
    mean_adj = windows - np.mean(windows, axis=1, keepdims=True)
    cumdev = np.cumsum(mean_adj, axis=1)

    # Calcula R e S
    R = np.ptp(cumdev, axis=1)  # max-min
    S = np.std(windows, axis=1)
    S = np.clip(S, min_var, None)  # evita zeros

    # 🧠 Expoente Hurst local
    H = np.log(R / S) / np.log(window)
    H = np.clip(H, 0.0, 2.0)  # limites teóricos aproximados

    # 🔹 Reconstrói vetor completo (NaN nas bordas iniciais)
    result = np.full(N, np.nan, dtype=float)
    result[window - 1:] = H

    # 🔮 Suavização simbiótica leve (menos ruído e saltos)
    result = _smooth(result, alpha=0.15)

    return result


def _smooth(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Suavização exponencial leve para evitar ruídos espúrios."""
    y = np.copy(x)
    for i in range(1, len(y)):
        if np.isnan(y[i]) or np.isnan(y[i-1]):
            continue
        y[i] = alpha * y[i] + (1 - alpha) * y[i-1]
    return y
