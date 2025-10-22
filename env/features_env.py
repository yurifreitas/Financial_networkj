# =========================================================
# 🌌 EtherSym Finance — features_env_v3_unificado.py
# =========================================================
# - Mantém o mesmo formato de saída do modelo original
# - Inclui todos os indicadores simbióticos avançados
# - Totalmente compatível com Env e RedeAvancada (sem ajustes)
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
# 🧩 Função principal
# =========================================================
def make_feats(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ===============================
    # 🔹 Retornos e volatilidade
    # ===============================
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)

    # ===============================
    # 🔹 Médias móveis e RSI simbiótico
    # ===============================
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)

    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi_base = 100 - (100 / (1 + up / (down + 1e-9)))
    df["rsi_n"] = (rsi_base - 50) / 50

    # ===============================
    # 🔹 Corpo e amplitude das velas
    # ===============================
    df["range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["body"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 1e-9)

    # ===============================
    # 🔹 Indicadores técnicos adicionais
    # ===============================
    _, _, macd_hist = macd(df["close"].values)
    df["macd_hist"] = macd_hist
    df["trix"] = trix(df["close"].values, period=15)

    # ===============================
    # 🔹 Indicadores estatísticos e fractais
    # ===============================
    df["hurst"] = df["close"].rolling(200).apply(lambda x: hurst(x).mean(), raw=False)
    df["entropy"] = df["close"].rolling(200).apply(lambda x: shannon_entropy(x).mean(), raw=False)
    df["kurtosis"] = df["close"].rolling(200).apply(lambda x: kurtosis(x).mean(), raw=False)
    df["mfdfa"] = df["close"].rolling(300).apply(
        lambda x: multifractal_dfa(x).get("dfa_alpha", np.nan), raw=False
    )

    # ===============================
    # 🔹 Energia de sinal (Wavelet)
    # ===============================
    df["wave_energy"] = df["close"].rolling(256).apply(lambda x: wavelet_energy(x), raw=False)

    # ===============================
    # 🔹 Volume e desvio Z
    # ===============================
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"] - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)
    df["volume_z"] = ((df["volume"] - df["volume"].rolling(48).mean()) /
                      (df["volume"].rolling(48).std() + 1e-9)).fillna(0.0)

    # ===============================
    # 🔹 Limpeza e normalização
    # ===============================
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")
    df = df.dropna().reset_index(drop=True)

    # =========================================================
    # 🌠 Vetor final — formato idêntico ao antigo (8 colunas)
    # =========================================================
    # Os novos indicadores são incorporados nas existentes via modulação simbiótica
    df["range"] *= (1 + 0.05 * df["wave_energy"].fillna(0))
    df["vol_ret"] *= (1 + 0.02 * df["hurst"].fillna(0))
    df["rsi_n"] *= (1 + 0.03 * df["entropy"].fillna(0))
    df["ema_diff"] *= (1 + 0.01 * df["macd_hist"].fillna(0))
    df["body"] *= (1 + 0.02 * df["mfdfa"].fillna(0))
    df["z"] *= (1 + 0.01 * df["kurtosis"].fillna(0))

    # 🔹 Retorno simbiótico final (8 features + preço)
    base = df[[
        "ret", "vol_ret", "range", "body",
        "ema_diff", "rsi_n", "z", "volume_z"
    ]].astype(np.float32).values

    price = df["close"].astype(np.float32).values

    print(f"✅ Features simbióticas geradas | shape base={base.shape}")
    return base, price
