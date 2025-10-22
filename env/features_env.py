# =========================================================
# 🌌 EtherSym Finance — features_env.py (v2.0 simbiótico)
# =========================================================
# - Combina análise estatística, fractal, energética e técnica
# - Normaliza coerentemente o vetor de estado simbiótico
# - Compatível com o ambiente Env do treino principal
# =========================================================

import numpy as np
import pandas as pd

# === Indicadores internos ===
from indicators.statistical.hurst import compute as hurst
from indicators.statistical.shannon_entropy import compute as shannon_entropy
from indicators.statistical.kurtosis import compute as kurtosis
from indicators.signal_energy.wavelet_transform import compute as wavelet_energy
from indicators.technical.macd import compute as macd
from indicators.technical.rsi import compute as rsi
from indicators.technical.trix import compute as trix
from indicators.fractal_chaos.mfdfa import compute as multifractal_dfa


# =========================================================
# 🧩 Função auxiliar
# =========================================================
def normalize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


# =========================================================
# 🌠 Núcleo simbiótico de features
# =========================================================
def make_feats(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ===============================
    # 🔹 Retornos e volatilidade básica
    # ===============================
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)

    # ===============================
    # 🔹 Indicadores Técnicos
    # ===============================
    _, _, macd_hist = macd(df["close"].values)
    df["macd_hist"] = macd_hist
    df["rsi_n"] = (rsi(df["close"].values, period=14) - 50) / 50
    df["trix"] = trix(df["close"].values, period=15)

    # Médias móveis diferenciais
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)

    # Corpo e amplitude das velas
    df["range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["body"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 1e-9)

    # ===============================
    # 🔹 Volume e desvio Z
    # ===============================
    df["volume_z"] = normalize(
        df["volume"].rolling(48).apply(lambda x: x.iloc[-1] - x.mean())
    )
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"] - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    # ===============================
    # 🔸 Métricas Estatísticas & Fractais
    # ===============================
    df["hurst"] = df["close"].rolling(200).apply(lambda x: hurst(x.values).mean(), raw=False)
    df["entropy"] = df["close"].rolling(200).apply(lambda x: shannon_entropy(x.values).mean(), raw=False)
    df["kurtosis"] = df["close"].rolling(200).apply(lambda x: kurtosis(x.values).mean(), raw=False)

    # Multifractalidade (dimensão fractal local)
    df["mfdfa"] = df["close"].rolling(300).apply(
        lambda x: multifractal_dfa(x.values).get("dfa_alpha", np.nan), raw=False
    )

    # ===============================
    # 🔹 Energia de Sinal (Wavelet)
    # ===============================
    df["wave_energy"] = df["close"].rolling(256).apply(
        lambda x: wavelet_energy(x.values), raw=False
    )

    # ===============================
    # 🔹 Normalização e limpeza
    # ===============================
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # ===============================
    # 🔹 Montagem final do vetor simbiótico
    # ===============================
    base_cols = [
        "ret", "vol_ret", "range", "body",
        "ema_diff", "rsi_n", "macd_hist", "trix",
        "hurst", "entropy", "kurtosis", "mfdfa",
        "wave_energy", "z", "volume_z"
    ]

    base = df[base_cols].astype(np.float32).values
    price = df["close"].astype(np.float32).values

    return base, price
