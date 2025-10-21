import numpy as np
import hurst

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        H, _, _ = hurst.compute_Hc(close[i-window:i], kind='price', simplified=True)
        result.append(H)
    return result
