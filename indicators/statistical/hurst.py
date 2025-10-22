import numpy as np

def compute(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Cálculo do expoente de Hurst (R/S) com janela deslizante."""
    N = len(close)
    result = np.full(N, np.nan, dtype=float)
    for i in range(window - 1, N):
        sub = close[i - window + 1:i + 1]
        mean_adj = sub - np.mean(sub)
        cumdev = np.cumsum(mean_adj)
        R = np.max(cumdev) - np.min(cumdev)
        S = np.std(sub)
        result[i] = np.log(R / (S + 1e-10)) / np.log(window)
    return result
