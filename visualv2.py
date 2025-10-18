# =========================================================
# 🌌 EtherSym Finance — visualizar_previsoes_plotly.py
# =========================================================
# - Lê debug_previsoes.csv
# - Mostra gráfico interativo com datas reais
# - Exibe preços reais vs previstos e erro percentual
# - Calcula métricas automáticas (MAE, MAPE, Corr)
# =========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# ⚙️ Configurações
# ---------------------------------------------------------
CSV_FILE = "previsoes_com_preco.csv"
OUTPUT_HTML = "ether_debug_previsoes.html"

# ---------------------------------------------------------
# 📈 Carrega CSV
# ---------------------------------------------------------
df = pd.read_csv(CSV_FILE, parse_dates=["data"])
df = df.dropna(subset=["preco_real", "preco_previsto"])

# ---------------------------------------------------------
# 📊 Métricas globais
# ---------------------------------------------------------
mae = np.mean(np.abs(df["preco_real"] - df["preco_previsto"]))
mape = np.mean(df["erro_percentual"]) * 100
corr = df["preco_real"].corr(df["preco_previsto"])

print(f"📊 Métricas gerais:")
print(f" - MAE : {mae:,.3f}")
print(f" - MAPE: {mape:.3f}%")
print(f" - Corr: {corr:.3f}")

# ---------------------------------------------------------
# 🎨 Gráfico interativo
# ---------------------------------------------------------
fig = go.Figure()

# Linha do preço real
fig.add_trace(go.Scatter(
    x=df["data"],
    y=df["preco_real"],
    mode="lines+markers",
    name="Preço Real",
    line=dict(color="white", width=2),
    hovertemplate="Data: %{x}<br>Preço Real: %{y:.2f}<extra></extra>"
))

# Linha do preço previsto
fig.add_trace(go.Scatter(
    x=df["data"],
    y=df["preco_previsto"],
    mode="lines+markers",
    name="Preço Previsto",
    line=dict(color="cyan", dash="dot", width=2),
    hovertemplate=(
        "Data: %{x}<br>"
        "Preço Previsto: %{y:.2f}<br>"
        "<extra></extra>"
    )
))

# Erro percentual (eixo secundário)
fig.add_trace(go.Bar(
    x=df["data"],
    y=df["erro_percentual"] * 100,
    name="Erro Percentual (%)",
    yaxis="y2",
    marker_color="orange",
    opacity=0.4,
    hovertemplate="Erro: %{y:.3f}%<extra></extra>"
))

# ---------------------------------------------------------
# ⚙️ Layout
# ---------------------------------------------------------
fig.update_layout(
    title=f"🧭 EtherSym Finance — Previsões de Preço ({len(df)} pontos)",
    template="plotly_dark",
    xaxis_title="Data / Tempo",
    yaxis_title="Preço (USD)",
    yaxis2=dict(
        title="Erro Percentual (%)",
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1
    ),
    font=dict(family="JetBrains Mono", size=13),
    width=1300, height=700
)

# Adiciona métricas no topo
fig.add_annotation(
    text=(
        f"<b>MAE:</b> {mae:,.3f} | "
        f"<b>MAPE:</b> {mape:.3f}% | "
        f"<b>Corr:</b> {corr:.3f}"
    ),
    xref="paper", yref="paper", x=0.5, y=1.08,
    showarrow=False, font=dict(size=14, color="gold")
)

# ---------------------------------------------------------
# 💾 Salva HTML
# ---------------------------------------------------------
fig.write_html(OUTPUT_HTML)
print(f"✅ Gráfico interativo salvo em: {OUTPUT_HTML}")
print("💡 Abra o arquivo no navegador para explorar as previsões.")

# ---------------------------------------------------------
# 🧠 Dica: se quiser abrir direto no Python (sem HTML)
# ---------------------------------------------------------
# fig.show()
