import numpy as np

def compute(high: np.ndarray, low: np.ndarray, period=10):
    median_price = (high + low) / 2
    fisher = [None] * (period - 1)
    value = np.zeros(len(high))

    for i in range(period - 1, len(median_price)):
        max_h = np.max(median_price[i - period + 1:i + 1])
        min_l = np.min(median_price[i - period + 1:i + 1])
        value[i] = 0.33 * 2 * ((median_price[i] - min_l) / (max_h - min_l + 1e-10) - 0.5) + 0.67 * value[i-1]
        fisher.append(0.5 * np.log((1 + value[i]) / (1 - value[i] + 1e-10)))

    return fisher
