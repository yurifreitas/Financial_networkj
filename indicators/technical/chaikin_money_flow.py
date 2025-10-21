import numpy as np

def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period=20):
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
    cmf = [None] * (period - 1)
    for i in range(period - 1, len(mfv)):
        cmf.append(np.sum(mfv[i + 1 - period:i + 1]) / (np.sum(volume[i + 1 - period:i + 1]) + 1e-10))
    return cmf
