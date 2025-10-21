import numpy as np
from statsmodels.tsa.stattools import bds

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        bds_stat, pvalue = bds(close[i-window:i])
        result.append(pvalue[0])  # primeiro valor
    return result
