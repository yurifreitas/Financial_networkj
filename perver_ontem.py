# =========================================================
# 🌌 EtherSym Finance — Extract + Predict (de Ontem até Agora)
# =========================================================
# - Baixa candles de BTC/USDT desde ontem (1h)
# - Extrai features via env.make_feats (8 dimensões)
# - Adiciona [pos, a_prev] = [0, 0] para compatibilidade (total 10)
# - Carrega modelo pré-treinado e gera previsões
# - Exporta CSV e gráfico interativo Plotly
# =========================================================

from binance import BinanceSync
from datetime import datetime, timedelta, timezone
import pandas as pd, torch, numpy as np, plotly.graph_objects as go
from env import make_feats
from network import criar_modelo

# -----------------------------
# ⚙️ Configurações principais
# -----------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
MODEL_PATH = "estado_treinamento_finance.pth"
CSV_PATH = f"binance_{SYMBOL.replace('/','_')}_{TIMEFRAME}_ontem.csv"
OUTPUT_PRED = "previsoes_ontem.csv"
HTML_PLOT = "previsoes_ontem.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 📡 Extrair candles da Binance
# -----------------------------
def fetch_yesterday():
    ex = BinanceSync({})
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    print(f"📡 Baixando candles de {SYMBOL} desde {start} até {now} ({TIMEFRAME})...")

    data = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=int(start.timestamp() * 1000), limit=1000)
    if not data:
        raise RuntimeError("❌ Nenhum dado retornado da API Binance.")

    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)

    print(f"✅ {len(df)} candles salvos: {CSV_PATH}")
    print(f"📅 Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df

# -----------------------------
# 🔮 Gerar previsões simbióticas
# -----------------------------
@torch.no_grad()
def predict(modelo, base, price, timestamps):
    preds = []
    for i in range(len(base)):
        # adiciona 2 dimensões extras (pos=0, a_prev=0)
        feat = np.concatenate([base[i], [0.0, 0.0]]).astype(np.float32)
        x = torch.tensor(feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        q_vals, y_pred = modelo(x)
        ret_pred = float(y_pred.squeeze().cpu().numpy())
        preco_real = float(price[i])
        preco_prev = preco_real * (1 + ret_pred)

        preds.append({
            "data": timestamps[i],
            "preco_real": preco_real,
            "preco_previsto": preco_prev,
            "retorno_pred": ret_pred
        })

    df_pred = pd.DataFrame(preds)
    df_pred.to_csv(OUTPUT_PRED, index=False)
    print(f"✅ Previsões salvas em {OUTPUT_PRED}")
    return df_pred

# -----------------------------
# 📊 Visualização Plotly
# -----------------------------
def plot(df_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred["data"], y=df_pred["preco_real"],
                             name="Preço Real", line=dict(color="white", width=2)))
    fig.add_trace(go.Scatter(x=df_pred["data"], y=df_pred["preco_previsto"],
                             name="Previsto", line=dict(color="cyan", dash="dot")))
    fig.update_layout(template="plotly_dark", title=f"📈 EtherSym Finance — {SYMBOL} ({TIMEFRAME})",
                      width=1200, height=700, hovermode="x unified")
    fig.write_html(HTML_PLOT, include_plotlyjs="cdn")
    print(f"💡 Gráfico salvo: {HTML_PLOT}")

# -----------------------------
# 🚀 Execução principal
# -----------------------------
def main():
    df = fetch_yesterday()
    base, price = make_feats(df)

    print(f"🧠 Carregando modelo pré-treinado: {MODEL_PATH}")
    modelo, _, _ = criar_modelo(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    modelo.load_state_dict(state.get("modelo", state), strict=False)
    modelo.eval()

    df_pred = predict(modelo, base, price, df["timestamp"].iloc[-len(base):].values)
    plot(df_pred)

if __name__ == "__main__":
    main()
