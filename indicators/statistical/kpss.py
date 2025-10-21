import numpy as np
import warnings
from statsmodels.tsa.stattools import kpss

def compute(close: np.ndarray, window=100):
    result = [None] * (window - 1)

    for i in range(window, len(close) + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                kpss_stat, pvalue, *_ = kpss(close[i - window:i])
        except Exception:
            pvalue = 1.0  # valor neutro em caso de falha

        result.append(pvalue)

    return result
