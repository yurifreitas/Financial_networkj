# =========================================================
# 🌌 EtherSym Finance — Simulador de 1 Minuto
# =========================================================
# - Usa modelo pré-treinado (Dueling DQN)
# - Busca candles 1m da Binance
# - Prevê direção e calcula equity simbiótica
# - Finaliza após X minutos
# - Mostra curva de preço real, previsto e equity
# =========================================================

from binance import BinanceSync
from datetime import datetime, timedelta, timezone
import pandas as pd, numpy as np, torch, time
import plotly.graph_objects as go
from env import make_feats
from network import criar_modelo

# -----------------------------
# ⚙️ Configurações
# -----------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
MODEL_PATH = "estado_treinamento_finance.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALOR_INICIAL = 1000.0   # 💰 capital inicial simbiótico
DURACAO_MINUTOS = 60     # ⏱ duração total da simulação
INTERVALO = 60           # ⏲️ segundos entre candles (1 min)

# -----------------------------
# 📡 Função para buscar últimos candles
# -----------------------------
def fetch_last(n=120):
    ex = BinanceSync({})
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=n)
    data = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=int(start.timestamp()*1000), limit=n)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# 🔮 Função de previsão
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
# 🎮 Simulação com posição contínua
# -----------------------------
def simulate_trade(model, capital_inicial=1000.0, minutos=60):
    ex = BinanceSync({})
    saldo = capital_inicial
    posicao = 0        # -1 = vendido, 0 = fora, +1 = comprado
    preco_entrada = 0  # preço médio simbiótico de entrada
    df_log = []

    print(f"🚀 Iniciando simulação com {capital_inicial:.2f} USD por {minutos} minutos...")

    for m in range(minutos):
        df = fetch_last(120)
        ret_pred, preco_pred = predict_next(model, df)
        preco_atual = df["close"].iloc[-1]
        acao = np.sign(ret_pred)  # decisão do modelo
        ret_real = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]

        # ======== LÓGICA DE POSIÇÃO ========
        # Entrar ou sair conforme sinal
        if posicao == 0 and acao != 0:
            posicao = acao
            preco_entrada = preco_atual
            print(f"🟢 Entrando em posição {posicao:+} a {preco_atual:.2f}")

        elif posicao != 0 and acao == 0:
            lucro = (preco_atual - preco_entrada) * posicao
            saldo += saldo * (lucro / preco_entrada)
            print(f"🔴 Saindo da posição a {preco_atual:.2f} | Lucro simbiótico: {lucro/preco_entrada*100:.3f}%")
            posicao = 0

        elif posicao != 0 and acao != posicao:
            # Inversão direta (de long→short ou short→long)
            lucro = (preco_atual - preco_entrada) * posicao
            saldo += saldo * (lucro / preco_entrada)
            posicao = acao
            preco_entrada = preco_atual
            print(f"🔁 Inversão para {posicao:+} | Novo preço base: {preco_entrada:.2f}")

        # ======== Registro contínuo ========
        df_log.append({
            "tempo": datetime.now(timezone.utc),
            "preco_real": preco_atual,
            "preco_previsto": preco_pred,
            "ret_pred": ret_pred,
            "acao": acao,
            "posicao": posicao,
            "saldo": saldo
        })

        print(f"[{m+1:02d}/{minutos}] 🕒 {datetime.now().strftime('%H:%M:%S')} | "
              f"Preço={preco_atual:.2f} | Prev={preco_pred:.2f} | "
              f"Ação={acao:+} | Posição={posicao:+} | Saldo={saldo:.2f}")

        time.sleep(INTERVALO)

    df_log = pd.DataFrame(df_log)
    return df_log

# -----------------------------
# 📊 Visualização final
# -----------------------------
def plot_result(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["tempo"], y=df["preco_real"], name="Preço Real", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df["tempo"], y=df["preco_previsto"], name="Previsto", line=dict(color="cyan", dash="dot")))
    fig.add_trace(go.Scatter(x=df["tempo"], y=df["saldo"], name="Saldo", yaxis="y2", line=dict(color="orange")))
    fig.update_layout(
        template="plotly_dark",
        title="📈 EtherSym — Simulação 1m",
        xaxis=dict(title="Tempo"),
        yaxis=dict(title="Preço BTC/USDT"),
        yaxis2=dict(title="Saldo (USD)", overlaying="y", side="right"),
        width=1200, height=700
    )
    fig.write_html("simulador_1min.html", include_plotlyjs="cdn")
    print("✅ Gráfico salvo: simulador_1min.html")

# -----------------------------
# 🚀 Execução
# -----------------------------
def main():
    modelo, _, _ = criar_modelo(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    modelo.load_state_dict(state.get("modelo", state), strict=False)
    modelo.eval()

    df_result = simulate_trade(modelo, VALOR_INICIAL, DURACAO_MINUTOS)
    plot_result(df_result)
    print(f"\n🏁 Simulação finalizada. Lucro total: {df_result['saldo'].iloc[-1]-VALOR_INICIAL:.2f} USD")

if __name__ == "__main__":
    main()
