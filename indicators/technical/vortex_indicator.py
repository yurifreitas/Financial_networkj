import numpy as np

def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14):
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])

    vi_plus = [None] * period
    for i in range(period, len(tr)):
        vi_plus.append(np.sum(vm_plus[i-period:i]) / (np.sum(tr[i-period:i]) + 1e-10))

    return vi_plus  # Retorna apenas uma lista simples
