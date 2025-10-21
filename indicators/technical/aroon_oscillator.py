import numpy as np

def compute(high: np.ndarray, low: np.ndarray, period=25):
    aroon_osc = [None] * (period - 1)
    for i in range(period - 1, len(high)):
        high_days = period - np.argmax(high[i - period + 1:i + 1]) - 1
        low_days = period - np.argmin(low[i - period + 1:i + 1]) - 1
        aroon_up = (high_days / period) * 100
        aroon_down = (low_days / period) * 100
        aroon_osc.append(aroon_up - aroon_down)
    return aroon_osc
