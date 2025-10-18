# =========================================================
# 📈 Coletor de dados — Binance (modo síncrono)
# =========================================================

from binance import BinanceSync
import pandas as pd
import numpy as np

def get_recent_candles(symbol="BTC/USDT", limit=120, timeframe="1m"):
    ex = BinanceSync({})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    return df
