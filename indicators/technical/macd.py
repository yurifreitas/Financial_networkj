import numpy as np

def compute(close: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line (9-period EMA of MACD line)
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal)

    # Garantindo o alinhamento correto
    macd_line = macd_line[-len(signal_line):]
    macd_histogram = macd_line - signal_line

    # Prefixo inicial None até começar o MACD válido
    prefix_length = len(series) - len(macd_histogram)
    prefix = [None] * prefix_length

    # Evitar NaN no retorno (converter para None)
    macd_histogram_list = [None if np.isnan(x) else float(x) for x in macd_histogram]

    return prefix + macd_histogram_list

def ema(series, period):
    ema_series = np.full_like(series, fill_value=np.nan, dtype=float)
    alpha = 2 / (period + 1)

    for i in range(len(series)):
        if i == period - 1:
            ema_series[i] = np.mean(series[:period])
        elif i >= period:
            ema_series[i] = alpha * (series[i] - ema_series[i - 1]) + ema_series[i - 1]

    return ema_series
