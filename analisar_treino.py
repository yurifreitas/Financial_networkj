# ============================================================
# 📊 EtherSym Finance — Analisador de Treino (.pth + Métricas)
# ============================================================
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from v1.network import RedeAvancada

# Caminhos
STATE_PATH = "/home/yuri/Documents/code2/binance-model/estado_treinamento_finance.pth"
LOG_PATH = "/home/yuri/Documents/code2/binance-model/metrics_history.jsonl"  # arquivo com as linhas JSON

# ============================================================
# 🔍 1. Carregar modelo e inspecionar integridade
# ============================================================
def carregar_modelo(path=STATE_PATH):
    if not os.path.exists(path):
        print(f"❌ Arquivo não encontrado: {path}")
        return None

    print(f"🧠 Carregando modelo de {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "modelo" in checkpoint:
            state_dict = checkpoint["modelo"]
        else:
            state_dict = checkpoint
        modelo = RedeAvancada(state_dim=10, n_actions=3)
        modelo.load_state_dict(state_dict, strict=False)
        print("✅ Modelo carregado com sucesso.")
        return modelo
    except Exception as e:
        print(f"⚠️ Falha ao carregar modelo: {e}")
        return None

# ============================================================
# 🧩 2. Carregar métricas salvas
# ============================================================
def carregar_metricas(path=LOG_PATH):
    if not os.path.exists(path):
        print(f"⚠️ Nenhum arquivo de métricas encontrado em {path}")
        return None

    linhas = []
    with open(path, "r") as f:
        for line in f:
            try:
                linhas.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(linhas)
    print(f"📈 {len(df)} episódios carregados")
    return df

# ============================================================
# 🧬 3. Visualização simbiótica das métricas
# ============================================================
def plotar_metricas(df):
    if df is None or df.empty:
        print("⚠️ Nenhuma métrica para plotar.")
        return

    plt.figure(figsize=(14, 10))
    plt.suptitle("🧠 Evolução do Treinamento Simbiótico — EtherSym Finance", fontsize=14, weight="bold")

    # Subplots organizados
    plt.subplot(3, 2, 1)
    plt.plot(df["ep"], df["pontuacao"], label="Pontuação", color="cyan")
    plt.plot(df["ep"], df["melhor"], label="Melhor", linestyle="--", color="orange")
    plt.xlabel("Episódio")
    plt.ylabel("Pontuação")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.plot(df["ep"], df["taxa_acerto"], label="Taxa de Acerto", color="lime")
    plt.xlabel("Episódio")
    plt.ylabel("Taxa de Acerto")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.plot(df["ep"], df["patrimonio_final"], label="Patrimônio", color="violet")
    plt.plot(df["ep"], df["max_patrimonio"], label="Máximo", color="magenta", linestyle="--")
    plt.xlabel("Episódio")
    plt.ylabel("Patrimônio")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.plot(df["ep"], df["eficiencia"], label="Eficiência", color="gold")
    plt.plot(df["ep"], df["sharpe_like"], label="Sharpe-like", color="orange", linestyle="--")
    plt.xlabel("Episódio")
    plt.ylabel("Desempenho")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    plt.plot(df["ep"], df["entropia"], label="Entropia", color="red")
    plt.xlabel("Episódio")
    plt.ylabel("Entropia (informacional)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 6)
    plt.plot(df["ep"], df["max_drawdown"], label="Max Drawdown", color="salmon")
    plt.xlabel("Episódio")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# ============================================================
# ⚙️ Execução
# ============================================================
if __name__ == "__main__":
    modelo = carregar_modelo()
    df = carregar_metricas()
    if df is not None:
        print(df[["ep", "pontuacao", "taxa_acerto", "eficiencia", "sharpe_like", "entropia"]].tail(10))
        plotar_metricas(df)
