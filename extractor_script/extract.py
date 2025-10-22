# =============================================
# 🪙 Binance OHLCV Extractor — v1.0
# =============================================
# Usa o novo SDK oficial 'binance-python' (https://pypi.org/project/binance)
# para baixar candles históricos (OHLCV) e salvar em CSV.
#
# Instalação:
#   pip install binance pandas numpy
#
# Exemplo:
#   python extract_binance.py
#
# Resultado:
#   binance_BTC_USDT_1h_2y.csv (salvo localmente)
# =============================================

from binance import BinanceSync
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import pandas as pd
import time, os

# -----------------------------
# ⚙️ Configurações principais
# -----------------------------
SYMBOL = "BTC/USDT"       # Par de criptomoedas
TIMEFRAME = "15m"          # Pode usar "15m", "4h", "1d", etc.
YEARS = 2                 # Quantos anos de histórico baixar
CSV_PATH = f"binance_{SYMBOL.replace('/','_')}_{TIMEFRAME}_{YEARS}y.csv"

# -----------------------------
# 🔧 Função auxiliar
# -----------------------------
def fetch_range(start_ms, end_ms):
    """Baixa candles em um intervalo (15 dias)"""
    ex = BinanceSync({})
    data = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=start_ms, limit=1000)
    return data or []

# -----------------------------
# 🚀 Função principal
# -----------------------------
def fetch_fast():
    """Baixa todos os candles em paralelo e salva em CSV"""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"📁 Cache detectado: {CSV_PATH} ({len(df)} candles)")
        return df

    print(f"⬇️  Iniciando download de {YEARS} anos de {SYMBOL} ({TIMEFRAME})...")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(365 * YEARS))
    step = timedelta(days=15)  # cada thread cobre 15 dias
    ranges = []
    cur = start
    while cur < now:
        nxt = cur + step
        ranges.append((int(cur.timestamp()*1000), int(nxt.timestamp()*1000)))
        cur = nxt

    print(f"🧠 Criadas {len(ranges)} janelas de requisição...")

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, chunk in enumerate(pool.map(lambda r: fetch_range(*r), ranges)):
            if chunk:
                results.extend(chunk)
            print(f"  ✅ Janela {i+1}/{len(ranges)} concluída ({len(chunk)} candles)")
            time.sleep(0.05)

    if not results:
        raise RuntimeError("❌ Nenhum dado retornado da API Binance.")

    df = pd.DataFrame(results, columns=["ts","open","high","low","close","volume"])
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
