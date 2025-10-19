# =============================================
# 🪙 Binance OHLCV Extractor — Últimos 3 Dias (1h)
# =============================================
# Usa o SDK oficial 'binance-python' para baixar candles
# de 1 hora (1h) dos últimos 3 dias e salvar em CSV.
#
# Instalação:
#   pip install binance pandas numpy
#
# Execução:
#   python extract_binance_1h_3d.py
#
# Resultado:
#   binance_BTC_USDT_1h_3d.csv
# =============================================

from binance import BinanceSync
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import pandas as pd
import time, os

# -----------------------------
# ⚙️ Configurações principais
# -----------------------------
SYMBOL = "BTC/USDT"      # Par de criptomoedas
TIMEFRAME = "1h"         # Intervalo de candles (1 hora)
DAYS = 3                 # Quantos dias no passado buscar
CSV_PATH = f"binance_{SYMBOL.replace('/','_')}_{TIMEFRAME}_{DAYS}d.csv"

# -----------------------------
# 🔧 Função auxiliar
# -----------------------------
def fetch_range(start_ms, end_ms):
    """Baixa candles em um intervalo de tempo"""
    ex = BinanceSync({})
    data = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=start_ms, limit=1000)
    return data or []

# -----------------------------
# 🚀 Função principal
# -----------------------------
def fetch_fast():
    """Baixa candles dos últimos 3 dias (1h)"""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"📁 Cache detectado: {CSV_PATH} ({len(df)} candles)")
        return df

    print(f"⬇️  Iniciando download de {DAYS} dias de {SYMBOL} ({TIMEFRAME})...")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS)
    step = timedelta(days=1)  # janelas de 1 dia (24 candles cada)
    ranges = []
    cur = start
    while cur < now:
        nxt = cur + step
        ranges.append((int(cur.timestamp() * 1000), int(nxt.timestamp() * 1000)))
        cur = nxt

    print(f"🧠 Criadas {len(ranges)} janelas de requisição...")

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, chunk in enumerate(pool.map(lambda r: fetch_range(*r), ranges)):
            if chunk:
                results.extend(chunk)
            print(f"  ✅ Janela {i+1}/{len(ranges)} concluída ({len(chunk)} candles)")
            time.sleep(0.05)

    if not results:
        raise RuntimeError("❌ Nenhum dado retornado da API Binance.")

    # cria DataFrame estruturado
    df = pd.DataFrame(results, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Histórico salvo: {CSV_PATH}")
    print(f"📊 Total de candles: {len(df)}")
    print(f"📅 Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df

# -----------------------------
# 🧭 Execução direta
# -----------------------------
if __name__ == "__main__":
    try:
        df = fetch_fast()
        print("\n🧾 Primeiras linhas:\n", df.head())
    except Exception as e:
        print("⚠️ Erro:", e)
