import numpy as np
from statsmodels.tsa.stattools import adfuller

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        adf_stat, pvalue, *_ = adfuller(close[i-window:i])
        result.append(pvalue)
    return result
