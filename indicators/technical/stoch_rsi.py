import numpy as np

def compute(close: np.ndarray, period=14) -> list:
    rsi_series = np.array(compute_rsi(close, period), dtype=float)  # dtype float para aceitar np.nan
    rsi_series[np.equal(rsi_series, None)] = np.nan  # substituir None por np.nan

    stoch_rsi = []

    for i in range(len(rsi_series)):
        if i < period or np.isnan(rsi_series[i]):
            stoch_rsi.append(None)
        else:
            window = rsi_series[i - period:i]

            # agora a janela é garantidamente numérica ou np.nan
            if np.all(np.isnan(window)):
                stoch_rsi.append(None)
                continue

            min_rsi, max_rsi = np.nanmin(window), np.nanmax(window)
            denominator = max_rsi - min_rsi

            if np.isnan(denominator) or denominator == 0:
                stoch_rsi.append(None)
            else:
                stoch = (rsi_series[i] - min_rsi) / denominator
                stoch_rsi.append(float(stoch))

    return stoch_rsi

def compute_rsi(series, period):
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.convolve(gain, np.ones(period), 'valid') / period
    avg_loss = np.convolve(loss, np.ones(period), 'valid') / period
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return [None]*period + rsi.tolist()
