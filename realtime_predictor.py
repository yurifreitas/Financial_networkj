# =========================================================
# 🌌 EtherSym Finance — Previsor Simbiótico em Tempo Real
# =========================================================
# - Busca candles 1h da Binance (BTC/USDT)
# - Aplica make_feats() para gerar features
# - Prevê o próximo candle
# - Compara com preço real quando novo candle chega
# - Exibe acertos e erros em tempo real
# - Gera gráfico interativo + log CSV
# =========================================================

from binance import BinanceSync
from datetime import datetime, timedelta, timezone
import pandas as pd, numpy as np, time, torch
import plotly.graph_objects as go
from env import make_feats
from network import criar_modelo
import os

# -----------------------------
# ⚙️ Configurações principais
# -----------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
MODEL_PATH = "estado_treinamento_finance.pth"
LOG_FILE = "realtime_log.csv"
PLOT_FILE = "realtime_graph.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFRESH_MINUTES = 10  # intervalo de checagem da API (pode ser 60 para 1h)

# -----------------------------
# 📡 Função para buscar últimos candles
# -----------------------------
def fetch_last(hours=48):
    ex = BinanceSync({})
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    data = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=int(start.timestamp() * 1000), limit=1000)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# 🔮 Prever próximo candle
# -----------------------------
@torch.no_grad()
def predict_next(model, df):
    base, price = make_feats(df)
    feat = np.concatenate([base[-1], [0.0, 0.0]]).astype(np.float32)
    x = torch.tensor(feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q_vals, y_pred = model(x)
    ret = float(y_pred.squeeze().cpu().numpy())
    price_next = price[-1] * (1 + ret)
    return ret, price_next

# -----------------------------
# 🧮 Avaliar acerto
# -----------------------------
def calc_error(real, predicted):
    return abs(real - predicted) / real

def check_accuracy(df_log):
    df_log["erro_%"] = 100 * abs(df_log["preco_real"] - df_log["preco_previsto"]) / df_log["preco_real"]
    acertos = (df_log["erro_%"] < 0.3).sum()
    total = len(df_log)
    taxa = 100 * acertos / total if total > 0 else 0
    return taxa

# -----------------------------
# 📊 Atualizar gráfico
# -----------------------------
def update_plot(df_log):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_log["data"], y=df_log["preco_real"], name="Preço Real", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df_log["data"], y=df_log["preco_previsto"], name="Previsto", line=dict(color="cyan", dash="dot")))
    fig.update_layout(template="plotly_dark", title="📈 EtherSym — Previsão Simbiótica em Tempo Real",
                      width=1200, height=700, hovermode="x unified")
    fig.write_html(PLOT_FILE, include_plotlyjs="cdn")

# -----------------------------
# 🚀 Loop principal
# -----------------------------
def main():
    print("🧠 Iniciando previsor simbiótico em tempo real...")
    modelo, _, _ = criar_modelo(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    modelo.load_state_dict(state.get("modelo", state), strict=False)
    modelo.eval()

    # carrega log antigo
    df_log = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame(columns=["data","preco_real","preco_previsto","retorno_pred"])

    ultima_previsao = None
    ultima_data = None

    while True:
        try:
            df = fetch_last(48)
            last_row = df.iloc[-1]
            timestamp = last_row["timestamp"]
            price_real = float(last_row["close"])

            # se temos previsão anterior, avalia acerto
            if ultima_data and timestamp > ultima_data:
                df_log.loc[len(df_log)-1, "preco_real"] = price_real
                taxa = check_accuracy(df_log)
                print(f"✅ Atualizado {timestamp} | Erro médio: {taxa:.2f}%")

            # prever próximo
            ret_pred, price_pred = predict_next(modelo, df)
            nova_linha = {
                "data": timestamp + pd.Timedelta(hours=1),
                "preco_real": np.nan,
                "preco_previsto": price_pred,
                "retorno_pred": ret_pred
            }
            df_log = pd.concat([df_log, pd.DataFrame([nova_linha])], ignore_index=True)
            ultima_previsao = price_pred
            ultima_data = timestamp

            # salvar e plotar
            df_log.to_csv(LOG_FILE, index=False)
            update_plot(df_log)

            print(f"🔮 Previsto {timestamp + pd.Timedelta(hours=1)}: {price_pred:.2f} (ret={ret_pred:.5f})")
            print(f"📊 Total registros: {len(df_log)} | Próxima checagem em {REFRESH_MINUTES} min...\n")

        except Exception as e:
            print("⚠️ Erro:", e)

        time.sleep(REFRESH_MINUTES * 60)

if __name__ == "__main__":
    main()
