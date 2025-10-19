# =========================================================
# 🌌 EtherSym Finance — simulador_realtime.py (modo simbiótico completo)
# =========================================================
# - Inclui preço previsto (rede markoviana) e ponderado por prob.
# - Coerência simbiótica (energia × estabilidade preditiva)
# - Reinício automático por patrimônio (falência realista)
# - Histórico salvo em CSV (para análise futura)
# - Integração direta com frontend via WebSocket /ws/robo (caller externo)
# =========================================================

import os
import asyncio
import datetime
import numpy as np
import pandas as pd

# 🔗 Importações internas do backend
from backend.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictor import prever_tendencia
from backend.strategy_markov import EstrategiaVariacaoMarkovTurbo as EstrategiaVariacao
from backend.markov_predictor import rede_markoviana

# =========================================================
# ⚙️ Configurações gerais
# =========================================================
CAPITAL_INICIAL = 10000.0
ALOCACAO = 0.8
CUSTO_TRADE = 0.0004           # taxa de execução simulada (0.04%)
LIMIAR_PATRIMONIO = 2000.0     # falência simbiótica
SAVE_RUNS = True
RUNS_DIR = "runs"

# =========================================================
# 🤖 Classe principal de simulação
# =========================================================
class SimuladorRobo:
    def __init__(self):
        # Carrega modelo preditivo e estratégia
        self.modelo = carregar_modelo()
        self.estrategia = EstrategiaVariacao()
        self.episode = 1
        self.tick_idx = 0
        self.capital = float(CAPITAL_INICIAL)
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []

        if SAVE_RUNS:
            os.makedirs(RUNS_DIR, exist_ok=True)
        print("🧠 Simulador simbiótico inicializado.")

    # =====================================================
    # 🔁 Reinício do episódio
    # =====================================================
    def reset_episode(self, motivo="limite atingido"):
        if SAVE_RUNS and self.historico:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(self.historico)
            path = os.path.join(RUNS_DIR, f"{ts}_ep{self.episode:03d}.csv")
            df.to_csv(path, index=False)
            print(f"[episódio {self.episode}] salvo → {path}")

        print(f"[reinício simbiótico] motivo: {motivo}")
        self.episode += 1
        self.tick_idx = 0
        self.capital = float(CAPITAL_INICIAL)
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []

    # =====================================================
    # 🔮 Tick (loop principal da simulação)
    # =====================================================
    async def tick(self):
        # Coleta candles recentes
        df = get_recent_candles(limit=120)

        # Predição da rede (sem treino)
        pred = prever_tendencia(self.modelo, df)

        # Árvores de futuros (Markov simbiótica)
        df_prev = rede_markoviana(self.modelo, df, profundidade=5, bifurcacoes=3)

        # ---------------------------
        # 📈 Métricas preditivas
        # ---------------------------
        preco_atual = float(pred["preco"])

        if not df_prev.empty and "preco" in df_prev.columns:
            preco_previsto = float(df_prev["preco"].mean())
        else:
            preco_previsto = preco_atual  # fallback robusto
        print(preco_previsto)
        print(preco_atual)
        # média ponderada por probabilidade dos caminhos
        if not df_prev.empty and {"preco", "prob"}.issubset(df_prev.columns):
            # normaliza prob em cada t (segurança extra)
            df_norm = df_prev.copy()
            prob_sum = df_norm.groupby("t")["prob"].transform(lambda s: s.sum() if s.sum() > 0 else 1.0)
            df_norm["prob_norm"] = df_norm["prob"] / prob_sum
            preco_previsto_ponderado = float((df_norm["preco"] * df_norm["prob_norm"]).groupby(df_norm["t"]).sum().mean())
        else:
            preco_previsto_ponderado = preco_previsto

        # probabilidade de alta a partir dos caminhos
        if not df_prev.empty and "ret" in df_prev.columns:
            prob_alta = float((df_prev["ret"] > 0).mean())
        else:
            prob_alta = 0.5

        energia = float(pred.get("energia", 1.0))
        retorno_pred = float(pred.get("retorno_pred", 0.0))
        acao_modelo = int(pred.get("acao_modelo", 0))
        coerencia = float((1.0 - abs(retorno_pred)) * energia)

        # ---------------------------
        # ⚖️ Estratégia simbiótica
        # ---------------------------
        sinal = self.estrategia.aplicar(pred)
        acao = None

        if sinal == "comprar" and self.posicao == 0:
            qtd = (self.capital * ALOCACAO) / preco_atual if preco_atual > 0 else 0.0
            qtd = float(max(qtd, 0.0))
            self.capital -= self.capital * ALOCACAO
            self.posicao = qtd
            self.preco_entrada = preco_atual
            acao = "BUY"
            print(f"🟢 Compra simbiótica — {qtd:.5f} @ {preco_atual:.2f}")

        elif sinal == "vender" and self.posicao > 0:
            self.capital += self.posicao * preco_atual * (1.0 - CUSTO_TRADE)
            self.posicao = 0.0
            self.preco_entrada = None
            acao = "SELL"
            print(f"🔴 Venda simbiótica @ {preco_atual:.2f}")

        # ---------------------------
        # 🧮 Capital/Patrimônio
        # ---------------------------
        patrimonio = float(self.capital + (self.posicao * preco_atual))
        retorno_pct = float((patrimonio - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100.0)

        # ---------------------------
        # 🪶 Registro do tick
        # ---------------------------
        self.tick_idx += 1
        agora_iso = datetime.datetime.utcnow().isoformat()

        registro = {
            "ep": int(self.episode),
            "tick": int(self.tick_idx),
            "tempo": agora_iso,
            "tempo_execucao": agora_iso,         # opcional p/ RobotStatus "⏱ Última Execução"
            "preco": preco_atual,
            "preco_previsto": preco_previsto,    # usado no Chart.tsx
            "preco_previsto_ponderado": preco_previsto_ponderado,  # opcional extra
            "acao": acao or "-",
            "acao_modelo": acao_modelo,
            "capital": float(self.capital),
            "posicao": float(self.posicao),
            "preco_entrada": float(self.preco_entrada or 0.0),
            "patrimonio": patrimonio,
            "retorno_pct": retorno_pct,
            "retorno_pred": retorno_pred,
            "prob_alta": prob_alta,
            "energia": energia,
            "coerencia": coerencia,
        }

        # Log humano
        print(
            f"[Ep {self.episode:03d} | Tick {self.tick_idx:05d}] "
            f"Preço={preco_atual:.2f} | Prev(media)={preco_previsto:.2f} | "
            f"Prev(pond)={preco_previsto_ponderado:.2f} | Ação={acao or '-':>4} | "
            f"Patrimônio={patrimonio:.2f} | Δpred={retorno_pred:+.5f} | "
            f"P(Alta)={prob_alta:.3f} | Enr={energia:.2f} | Coer={coerencia:.2f}"
        )

        self.historico.append(registro)

        # ---------------------------
        # 💀 Falência simbiótica
        # ---------------------------
        if patrimonio <= LIMIAR_PATRIMONIO:
            self.reset_episode(
                motivo=f"patrimônio {patrimonio:.2f} <= limite {LIMIAR_PATRIMONIO}"
            )
            registro["episode_reset"] = True
        else:
            registro["episode_reset"] = False

        return registro


# =========================================================
# 🚀 Execução contínua (loop infinito)
# =========================================================
simulador = SimuladorRobo()

if __name__ == "__main__":
    async def main():
        robo = SimuladorRobo()
        while True:
            try:
                _ = await robo.tick()
                # aqui seu servidor WS externo pode ler o retorno
                await asyncio.sleep(60)  # 1 tick/min
            except Exception as e:
                print(f"[ERRO simbiótico] {e}")
                await asyncio.sleep(5)

    asyncio.run(main())
