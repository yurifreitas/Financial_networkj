import numpy as np

def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14):
    wr = [None] * (period - 1)
    for i in range(period - 1, len(close)):
        highest_high = np.max(high[i + 1 - period:i + 1])
        lowest_low = np.min(low[i + 1 - period:i + 1])
        value = -100 * (highest_high - close[i]) / (highest_high - lowest_low + 1e-10)
        wr.append(value)
    return wr
