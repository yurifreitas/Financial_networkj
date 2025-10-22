import numpy as np

def ema(series: np.ndarray, period: int) -> np.ndarray:
    ema_series = np.full_like(series, fill_value=np.nan, dtype=float)
    alpha = 2 / (period + 1)
    for i in range(len(series)):
        if i == period - 1:
            ema_series[i] = np.mean(series[:period])
        elif i > period - 1:
            ema_series[i] = alpha * (series[i] - ema_series[i - 1]) + ema_series[i - 1]
    return ema_series

def compute(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Retorna (macd_line, signal_line, histograma)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal)

    # Realinha o tamanho das séries
    valid_len = len(signal_line)
    macd_line = macd_line[-valid_len:]
    macd_hist = macd_line - signal_line

    # Completa com NaN para alinhar com close
    full_hist = np.full(len(close), np.nan)
    full_hist[-len(macd_hist):] = macd_hist
    return macd_line, signal_line, full_hist
