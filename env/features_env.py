# =========================================================
# 🌌 EtherSym Finance — features_env_v2_cached.py
# =========================================================
# - Gera features simbióticas completas (estatísticas, fractais, energéticas e técnicas)
# - Usa cache automático para evitar recálculo desnecessário
# - Verifica integridade pelo hash do dataset de entrada
# - Compatível com o ambiente Env do treino simbiótico
# =========================================================

import numpy as np
import pandas as pd
import hashlib, os, time, json

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
# 🧩 Funções auxiliares
# =========================================================
def normalize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


def hash_dataframe(df: pd.DataFrame) -> str:
    """Gera hash simbiótico para o dataset (para verificar se já existe cache)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(str(df.head(100).to_dict()).encode())
    return h.hexdigest()[:16]


# =========================================================
# 🌠 Núcleo simbiótico com cache
# =========================================================
def make_feats(df: pd.DataFrame, cache_dir="cache_features", force=False):
    """
    Retorna (base, price) e salva automaticamente um cache.
    - cache_dir: diretório onde os .npz serão salvos
    - force: se True, recalcula mesmo se já existir cache válido
    """

    os.makedirs(cache_dir, exist_ok=True)
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    data_hash = hash_dataframe(df)
    cache_file = os.path.join(cache_dir, f"features_{data_hash}.npz")
    meta_file = cache_file.replace(".npz", ".json")

    # 🔍 Se já existe cache válido
    if not force and os.path.exists(cache_file):
        data = np.load(cache_file)
        base, price = data["base"], data["price"]
        print(f"⚡ Cache simbiótico carregado: {cache_file} | base={base.shape}")
        return base, price

    print("🔄 Calculando features simbióticas (pode demorar alguns minutos)...")
    start = time.time()

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

    # 💾 Salvar cache comprimido e metadados
    np.savez_compressed(cache_file, base=base, price=price)
    with open(meta_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hash": data_hash,
            "n_samples": len(df),
            "n_features": base.shape[1],
            "duration_sec": round(time.time() - start, 2)
        }, f, indent=2)

    print(f"✅ Features simbióticas salvas em {cache_file} ({base.shape[0]}x{base.shape[1]})")
    return base, price
