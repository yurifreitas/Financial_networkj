import numpy as np
import antropy as ant

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        ApEn = ant.app_entropy(close[i-window:i])
        result.append(ApEn)
    return result
