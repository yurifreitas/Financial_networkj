import numpy as np
from scipy.stats import kurtosis

def compute(close: np.ndarray, window=14):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        result.append(kurtosis(close[i-window:i]))
    return result
