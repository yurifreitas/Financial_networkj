from binance import BinanceSync
from datetime import datetime, timezone
import pandas as pd

def get_recent_candles(symbol="BTC/USDT", limit=120, timeframe="1m"):
    ex = BinanceSync({})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df
