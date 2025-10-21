import numpy as np

def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=20):
    tp = (high + low + close) / 3
    cci = [None] * (period - 1)

    for i in range(period - 1, len(tp)):
        window = tp[i - period + 1:i + 1]
        sma = np.mean(window)
        mean_dev = np.mean(np.abs(window - sma))
        if mean_dev == 0:
            cci.append(None)
        else:
            cci_value = (tp[i] - sma) / (0.015 * mean_dev)
            cci.append(float(cci_value))

    return cci
