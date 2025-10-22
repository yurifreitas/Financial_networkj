import numpy as np

def compute(close: np.ndarray, period: int = 15) -> np.ndarray:
    """TRIX — variação percentual da média tripla exponencial."""
    close = np.asarray(close, dtype=float)
    ema1 = ema(close, period)
    ema2 = ema(np.nan_to_num(ema1), period)
    ema3 = ema(np.nan_to_num(ema2), period)

    trix = np.diff(ema3) / (ema3[:-1] + 1e-10) * 100
    result = np.concatenate([np.full(len(close) - len(trix), np.nan), trix])
    return result

def ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA simples (sem pandas, totalmente vetorial)."""
    ema_series = np.full_like(series, np.nan, dtype=float)
    alpha = 2 / (period + 1)
    for i in range(len(series)):
        if i == period - 1:
            ema_series[i] = np.mean(series[:period])
        elif i > period - 1:
            ema_series[i] = alpha * (series[i] - ema_series[i - 1]) + ema_series[i - 1]
    return ema_series
