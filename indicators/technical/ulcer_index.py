import numpy as np

def compute(close: np.ndarray, period=14):
    ui = [None] * (period - 1)
    for i in range(period - 1, len(close)):
        max_close = np.max(close[i - period + 1:i + 1])
        squared_drawdown = ((close[i - period + 1:i + 1] - max_close) / (max_close + 1e-10)) ** 2
        ui.append(np.sqrt(np.mean(squared_drawdown)))
    return ui
