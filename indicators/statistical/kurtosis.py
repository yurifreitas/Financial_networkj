import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

def compute(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Curtose (forma da distribuição) com janela móvel."""
    N = len(close)
    result = np.full(N, np.nan, dtype=float)
    for i in range(window - 1, N):
        result[i] = float(scipy_kurtosis(close[i - window + 1:i + 1], fisher=False))
    return result
