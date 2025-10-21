import numpy as np
from scipy.stats import skew

def compute(close, window=14):
    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        result.append(skew(close[i-window:i]))
    while len(result) < len(close):
        result.append(None)
    return result
