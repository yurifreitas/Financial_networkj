import numpy as np

def compute(close: np.ndarray, window=20) -> list:
    hurst_values = [None] * (window - 1)

    for i in range(window, len(close) + 1):
        subseries = close[i - window:i]
        N = len(subseries)
        ts = np.cumsum(subseries - np.mean(subseries))
        R = np.max(ts) - np.min(ts)
        S = np.std(subseries)
        H = np.log(R / (S + 1e-10)) / np.log(N)
        hurst_values.append(float(H))

    return hurst_values
