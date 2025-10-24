# =========================================================
# 🌌 EtherSym Finance — features_env_v15_cache_full.py
# =========================================================
# - Cache simbiótico completo: evita recalcular todos os indicadores
# - Salva e restaura automaticamente todas as features calculadas
# =========================================================

import os
import numpy as np
import pandas as pd
import hashlib
from indicators.statistical.shannon_entropy import compute as shannon_entropy
from indicators.statistical.kurtosis import compute as kurtosis
from indicators.signal_energy.wavelet_transform import compute as wavelet_energy
from indicators.technical.macd import compute as macd
from indicators.technical.rsi import compute as rsi
from indicators.technical.trix import compute as trix
from indicators.fractal_chaos.mfdfa import compute as multifractal_dfa
from indicators.statistical.hurst import compute as hurst


def make_feats(df: pd.DataFrame, cache_dir: str = "cache_features", cache_name: str = None):
    os.makedirs(cache_dir, exist_ok=True)

    # =====================================================
    # 🧬 Gera nome de cache único baseado no arquivo de origem ou conteúdo
    # =====================================================
    if cache_name is None:
        # Se o DataFrame tiver um atributo 'source_file' (adicionado no SimuladorReplay)
        if hasattr(df, "source_file") and df.source_file:
            base_name = os.path.basename(df.source_file)
        else:
            # fallback — usa o número de linhas + hash
            base_name = f"df_{len(df)}"
        # hash curto para evitar nomes repetidos
        hash_id = hashlib.md5(str(len(df)).encode()).hexdigest()[:6]
        cache_name = f"features_{base_name}_{hash_id}.parquet"

    cache_path = os.path.join(cache_dir, cache_name)

    # =====================================================
    # ♻️ 1️⃣ Tentativa de restaurar cache completo
    # =====================================================
    if os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            if len(cached) == len(df):
                print(f"♻️ Cache simbiótico completo carregado ({cache_path})")
                base_cols = [
                    "ret", "vol_ret", "range", "body", "ema_diff",
                    "rsi_n", "z", "volume_z", "macd_hist", "trix",
                    "hurst", "entropy", "kurtosis", "wave_energy", "mfdfa"
                ]
                base = cached[base_cols].astype(np.float32).values
                price = cached["close"].astype(np.float32).values
                return base, price
            else:
                print(f"⚠️ Cache inconsistente ({len(cached)} vs {len(df)}) — recalculando...")
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}. Recalculando indicadores...")
    # =====================================================
    # 🧮 2️⃣ Recalcular tudo do zero (primeira vez)
    # =====================================================
    df.columns = [c.lower() for c in df.columns]

    # 🔹 Núcleo base
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)

    # 🔹 RSI simbiótico
    rsi_vals = rsi(df["close"].values, period=14)
    df["rsi_n"] = np.clip(np.nan_to_num((rsi_vals - 50) / 50, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    # 🔹 Estrutura candle e volume
    df["range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["body"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 1e-9)
    df["volume_z"] = ((df["volume"] - df["volume"].rolling(48).mean()) /
                      (df["volume"].rolling(48).std() + 1e-9)).fillna(0.0)
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"] - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    # 🔹 Indicadores técnicos
    macd_line, signal_line, macd_hist = macd(df["close"].values)
    df["macd_hist"] = np.clip(np.nan_to_num(macd_hist, nan=0.0), -5.0, 5.0)
    df["trix"] = np.clip(np.nan_to_num(trix(df["close"].values, period=15), nan=0.0), -5.0, 5.0)

    # 🔹 Estatísticos e fractais
    print("🧮 Calculando Hurst...")
    hurst_vals = hurst(df["close"].values, window=70)
    df["hurst"] = np.clip(np.nan_to_num(hurst_vals - 1.0, nan=0.0), -1.0, 1.0)

    print("🧩 Calculando Entropy...")
    df["entropy"] = df["close"].rolling(120).apply(lambda x: np.nanmean(shannon_entropy(x)), raw=False)
    df["entropy"] = np.clip(np.nan_to_num(df["entropy"], nan=0.0), -1.0, 1.0)

    print("📈 Calculando Kurtosis...")
    df["kurtosis"] = df["close"].rolling(120).apply(lambda x: float(np.nanmean(kurtosis(x))), raw=False)
    df["kurtosis"] = np.clip(np.nan_to_num(df["kurtosis"], nan=0.0), -1.0, 1.0)

    print("⚡ Calculando Wavelet Energy...")
    df["wave_energy"] = df["close"].rolling(256).apply(lambda x: wavelet_energy(x), raw=False)
    df["wave_energy"] = np.clip(np.nan_to_num(df["wave_energy"], nan=0.0), -5.0, 5.0)

    print("🌌 Calculando MF-DFA...")
    df["mfdfa"] = df["close"].rolling(300).apply(lambda x: multifractal_dfa(x).get("dfa_alpha", np.nan), raw=False)
    df["mfdfa"] = np.clip(np.nan_to_num(df["mfdfa"], nan=0.0), -2.0, 2.0)

    # =====================================================
    # 🔹 Limpeza e montagem simbiótica
    # =====================================================
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    base_cols = [
        "ret", "vol_ret", "range", "body", "ema_diff",
        "rsi_n", "z", "volume_z", "macd_hist", "trix",
        "hurst", "entropy", "kurtosis", "wave_energy", "mfdfa"
    ]
    base = df[base_cols].astype(np.float32).values
    price = df["close"].astype(np.float32).values

    # =====================================================
    # 💾 3️⃣ Salvar cache simbiótico completo
    # =====================================================
    try:
        df_to_save = df[["close"] + base_cols].copy()
        df_to_save.to_parquet(cache_path, index=False)
        print(f"💾 Cache simbiótico salvo com sucesso → {cache_path}")
    except Exception as e:
        print(f"⚠️ Falha ao salvar cache simbiótico: {e}")

    # =====================================================
    # 🧠 Log simbiótico
    # =====================================================
    print(f"✅ base={base.shape} | price={price.shape}")
    print(f"🧩 Indicadores simbióticos vivos: {len(base_cols)}")
    print(f"📈 Colunas: {base_cols}")

    return base, price
