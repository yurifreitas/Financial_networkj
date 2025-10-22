import numpy as np

def compute(close: np.ndarray, window: int = 14, bins: int = 10) -> np.ndarray:
    """Entropia de Shannon móvel."""
    N = len(close)
    result = np.full(N, np.nan, dtype=float)
    for i in range(window - 1, N):
        segment = close[i - window + 1:i + 1]
        hist, _ = np.histogram(segment, bins=bins, density=True)
        hist = hist + 1e-10
        result[i] = -np.sum(hist * np.log2(hist))
    return result
