import numpy as np

def compute(close: np.ndarray, window=14, bins=10):
    def shannon_entropy(data):
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist += 1e-10
        return -np.sum(hist * np.log2(hist))

    result = [None] * (window - 1)
    for i in range(window, len(close) + 1):
        result.append(shannon_entropy(close[i-window:i]))
    return result
