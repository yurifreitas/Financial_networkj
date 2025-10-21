import numpy as np

def compute(high: np.ndarray, low: np.ndarray, volume: np.ndarray, period=14):
    midpoint_move = ((high[1:] + low[1:]) / 2) - ((high[:-1] + low[:-1]) / 2)
    box_ratio = volume[1:] / (high[1:] - low[1:] + 1e-10)
    eom = midpoint_move / (box_ratio + 1e-10)
    eom_avg = [None] * period
    for i in range(period, len(eom)):
        eom_avg.append(np.mean(eom[i-period:i]))
    return [None] + eom_avg  # Align to candle length
