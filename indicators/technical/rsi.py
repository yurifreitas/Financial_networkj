import numpy as np

def compute(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI normalizado (0–100) com preenchimento temporal completo."""
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.convolve(gain, np.ones(period), 'valid') / period
    avg_loss = np.convolve(loss, np.ones(period), 'valid') / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # alinhar com o tamanho original
    pad_len = len(close) - len(rsi)
    result = np.concatenate([np.full(pad_len, np.nan), rsi])
    return result
