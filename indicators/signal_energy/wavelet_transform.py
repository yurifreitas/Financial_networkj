import numpy as np
import pywt

def compute(series: np.ndarray, wavelet='db4', level=4):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    return coeffs
