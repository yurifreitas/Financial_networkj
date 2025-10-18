# =========================================================
# 🧩 EtherSym Finance — features_env.py
# =========================================================
import numpy as np
import pandas as pd

def make_feats(df: pd.DataFrame):
    df.columns = [c.lower() for c in df.columns]
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rsi = 100 - (100 / (1 + up/(down+1e-9)))
    df["rsi_n"] = (rsi - 50)/50
    df["range"] = (df["high"]-df["low"])/(df["close"]+1e-9)
    df["body"] = (df["close"]-df["open"])/((df["high"]-df["low"])+1e-9)
    df["volume_z"] = ((df["volume"]-df["volume"].rolling(48).mean())/
                      (df["volume"].rolling(48).std()+1e-9)).fillna(0.0)
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"]-roll.mean())/(roll.std()+1e-9)).fillna(0.0)
    df = df.dropna().reset_index(drop=True)
    base = df[["ret","vol_ret","range","body","ema_diff","rsi_n","z","volume_z"]].astype(np.float32).values
    price = df["close"].astype(np.float32).values
    return base, price
