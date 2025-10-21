import numpy as np

def ema(series, period):
    alpha = 2 / (period + 1)
    ema_series = np.full_like(series, np.nan)
    for i in range(len(series)):
        if i == period - 1:
            ema_series[i] = np.mean(series[:period])
        elif i >= period:
            ema_series[i] = alpha * series[i] + (1 - alpha) * ema_series[i-1]
    return ema_series

def compute(close: np.ndarray, period_short=23, period_long=50, cycle=10):
    macd = ema(close, period_short) - ema(close, period_long)
    stoch = 100 * (macd - np.min(macd[-cycle:])) / (np.max(macd[-cycle:]) - np.min(macd[-cycle:]) + 1e-10)
    schaff = ema(stoch, cycle // 2)
    return [None if np.isnan(x) else x for x in schaff]
