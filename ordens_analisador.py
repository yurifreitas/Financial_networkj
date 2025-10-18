# =========================================================
# 📈 EtherSym Finance — ordens_analisador.py
# =========================================================
# Lê ordens_executadas.json e gera relatório simbiótico
# =========================================================

import json
import pandas as pd
import matplotlib.pyplot as plt

ARQ = "ordens_executadas.json"

# ---------------------------------------------------------
# 🧠 Carregar e preparar dados
# ---------------------------------------------------------
with open(ARQ, "r") as f:
    ordens = json.load(f)

df = pd.DataFrame(ordens)
df["tempo"] = pd.to_numeric(df["tempo"])
df["preco"] = pd.to_numeric(df["preco"])
df["quantidade"] = pd.to_numeric(df["quantidade"])
df["capital_restante"] = pd.to_numeric(df["capital_restante"])

# ---------------------------------------------------------
# 🔍 Calcular lucros por operação SELL vs BUY anterior
# ---------------------------------------------------------
lucros = []
last_buy = None
for _, row in df.iterrows():
    if row["tipo"] == "BUY":
        last_buy = row
    elif row["tipo"] == "SELL" and last_buy is not None:
        lucro = (row["preco"] - last_buy["preco"]) * last_buy["quantidade"]
        lucros.append({
            "tempo": row["tempo"],
            "preco_compra": last_buy["preco"],
            "preco_venda": row["preco"],
            "quantidade": last_buy["quantidade"],
            "lucro": lucro,
        })
        last_buy = None

df_lucros = pd.DataFrame(lucros)

# ---------------------------------------------------------
# 📊 Estatísticas simbióticas
# ---------------------------------------------------------
total_ops = len(df)
n_buy = (df["tipo"] == "BUY").sum()
n_sell = (df["tipo"] == "SELL").sum()
lucro_total = df_lucros["lucro"].sum()
lucro_medio = df_lucros["lucro"].mean()
win_rate = (df_lucros["lucro"] > 0).mean() * 100

print("=========================================================")
print("📈 RELATÓRIO SINTÉTICO — EtherSym Finance")
print("=========================================================")
print(f"Total de operações: {total_ops}")
print(f"Ordens BUY: {n_buy} | SELL: {n_sell}")
print(f"Lucro total simbiótico: {lucro_total:.4f}")
print(f"Lucro médio por trade: {lucro_medio:.4f}")
print(f"Taxa de acerto (win rate): {win_rate:.2f}%")
print("=========================================================\n")

# ---------------------------------------------------------
# 📉 Gráficos simbióticos
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# --- 1. Capital ao longo do tempo
plt.subplot(3, 1, 1)
plt.plot(df["tempo"], df["capital_restante"], label="Capital", color="cyan")
plt.title("💎 Evolução do Capital Simbiótico")
plt.ylabel("Capital")
plt.grid(True)

# --- 2. Preço e ordens
plt.subplot(3, 1, 2)
plt.plot(df["tempo"], df["preco"], color="gray", label="Preço")
plt.scatter(df[df["tipo"]=="BUY"]["tempo"], df[df["tipo"]=="BUY"]["preco"], color="green", label="BUY", marker="^")
plt.scatter(df[df["tipo"]=="SELL"]["tempo"], df[df["tipo"]=="SELL"]["preco"], color="red", label="SELL", marker="v")
plt.title("📊 Preços e Pontos de Execução")
plt.ylabel("Preço")
plt.legend()
plt.grid(True)

# --- 3. Lucros individuais
if not df_lucros.empty:
    plt.subplot(3, 1, 3)
    plt.bar(df_lucros["tempo"], df_lucros["lucro"], color=["green" if x > 0 else "red" for x in df_lucros["lucro"]])
    plt.title("📈 Lucro por Operação")
    plt.ylabel("Lucro")
    plt.xlabel("Tempo")
    plt.grid(True)

plt.tight_layout()
plt.show()
