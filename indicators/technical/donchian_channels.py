import numpy as np

def compute(high: np.ndarray, low: np.ndarray, period=20):
    upper = [None] * (period - 1)
    for i in range(period - 1, len(high)):
        upper.append(np.max(high[i - period + 1:i + 1]))
    
    return upper  # Retorna somente o canal superior como uma lista simples
