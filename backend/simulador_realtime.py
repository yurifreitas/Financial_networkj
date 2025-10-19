# =========================================================
# 🌌 EtherSym Finance — simulador_realtime.py (modo simbiótico direto)
# =========================================================
# - Usa apenas a previsão direta da rede principal (sem Markov)
# - Coerência simbiótica (energia × estabilidade preditiva)
# - Reinício automático por patrimônio (falência realista)
# - Histórico salvo em CSV (para análise futura)
# - Integração com frontend via WebSocket /ws/robo
# =========================================================

import os
import asyncio
import datetime
import numpy as np
import pandas as pd

# 🔗 Importações internas do backend
from backend.loader.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictors.predictor import prever_tendencia
from backend.strategies.strategy import EstrategiaVariacao

# =========================================================
# ⚙️ Configurações gerais
# =========================================================
CAPITAL_INICIAL = 10_000.0
ALOCACAO = 0.8
CUSTO_TRADE = 0.0004           # taxa de execução simulada (0.04%)
LIMIAR_PATRIMONIO = 2_000.0    # falência simbiótica
SAVE_RUNS = True
RUNS_DIR = "runs"

# =========================================================
# 🤖 Classe principal de simulação
# =========================================================
class SimuladorRobo:
    def __init__(self):
        # Carrega modelo preditivo e estratégia simbiótica
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
        print("🧠 Simulador simbiótico direto inicializado (sem Markov).")

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
        # === 📡 Coleta candles recentes ===
        df = get_recent_candles(limit=120)

        # === 🔮 Predição simbiótica direta ===
        pred = prever_tendencia(self.modelo, df)
        preco_atual = float(pred.get("preco", df["close"].iloc[-1]))
        preco_previsto = float(pred.get("preco_previsto", preco_atual))
        energia = float(pred.get("energia", 1.0))
        retorno_pred = float(pred.get("retorno_pred", 0.0))
        prob_actions = pred.get("prob_actions", [0.33, 0.33, 0.34])
        prob_alta = float(prob_actions[2]) if isinstance(prob_actions, (list, np.ndarray)) else 0.5

        # Coerência simbiótica (energia × estabilidade)
        coerencia = float((1.0 - abs(retorno_pred)) * energia)

        # ---------------------------
        # ⚖️ Estratégia simbiótica
        # ---------------------------
        sinal = self.estrategia.aplicar(pred)
        acao = None

        if sinal == "comprar" and self.posicao == 0:
            qtd = (self.capital * ALOCACAO) / preco_atual if preco_atual > 0 else 0.0
            self.capital -= self.capital * ALOCACAO
            self.posicao = float(qtd)
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
            "preco": preco_atual,
            "preco_previsto": preco_previsto,
            "acao": acao or "-",
            "acao_modelo": int(pred.get("acao_modelo", 0)),
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

        print(
            f"[Ep {self.episode:03d} | Tick {self.tick_idx:05d}] "
            f"Preço={preco_atual:.2f} | Prev={preco_previsto:.2f} | "
            f"Ação={acao or '-':>4} | Patrimônio={patrimonio:.2f} | "
            f"Δpred={retorno_pred:+.5f} | P(Alta)={prob_alta:.3f} | "
            f"Enr={energia:.2f} | Coer={coerencia:.2f}"
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
if __name__ == "__main__":
    async def main():
        robo = SimuladorRobo()
        while True:
            try:
                _ = await robo.tick()
                await asyncio.sleep(60)  # 1 tick/minuto
            except Exception as e:
                print(f"[ERRO simbiótico] {e}")
                await asyncio.sleep(5)

    asyncio.run(main())
