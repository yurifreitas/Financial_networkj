import numpy as np

def compute(close: np.ndarray, period=15):
    ema1 = ema(close, period)
    ema2 = ema(ema1[~np.isnan(ema1)], period)
    ema3 = ema(ema2[~np.isnan(ema2)], period)
    trix = np.diff(ema3) / (ema3[:-1] + 1e-10) * 100

    trix_result = [None] * (len(close) - len(trix)) + trix.tolist()

    # Tratamento seguro para NaN sem erro de tipo
    trix_result = [
        None if not isinstance(x, (float, int)) or np.isnan(x) else x
        for x in trix_result
    ]

    return trix_result

def ema(series, period):
    ema_series = np.full_like(series, np.nan)
    alpha = 2 / (period + 1)
    for i in range(len(series)):
        if i == period - 1:
            ema_series[i] = np.mean(series[:period])
        elif i >= period:
            ema_series[i] = alpha * (series[i] - ema_series[i-1]) + ema_series[i-1]
    return ema_series
