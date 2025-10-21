import numpy as np
from hurst import compute_Hc

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        H, _, _ = compute_Hc(close[i-window:i])
        D = 2 - H
        result.append(D)
    return result
