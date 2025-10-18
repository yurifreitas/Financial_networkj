# =========================================================
# 🤖 EtherSym Finance — simulador_realtime.py (reinício por patrimônio)
# =========================================================
import os, asyncio, datetime, json, numpy as np, pandas as pd
from backend.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictor import prever_tendencia
from backend.strategy_markov import EstrategiaVariacaoMarkovTurbo as EstrategiaVariacao
from backend.markov_predictor import rede_markoviana

CAPITAL_INICIAL = 10000
ALOCACAO = 0.8
CUSTO_TRADE = 0.0004

# 👉 limite simbiótico
LIMIAR_PATRIMONIO = 2000
SAVE_RUNS = True
RUNS_DIR = "runs"

class SimuladorRobo:
    def __init__(self):
        self.modelo = carregar_modelo()
        self.estrategia = EstrategiaVariacao()
        self.episode = 1
        self.tick_idx = 0
        self.capital = CAPITAL_INICIAL
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []
        if SAVE_RUNS:
            os.makedirs(RUNS_DIR, exist_ok=True)

    def reset_episode(self, motivo="limite atingido"):
        if SAVE_RUNS and len(self.historico) > 0:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            df = pd.DataFrame(self.historico)
            path = os.path.join(RUNS_DIR, f"{ts}_ep{self.episode:03d}.csv")
            df.to_csv(path, index=False)
            print(f"[episódio {self.episode}] salvo → {path}")

        print(f"[reinício] motivo: {motivo}")
        self.episode += 1
        self.tick_idx = 0
        self.capital = CAPITAL_INICIAL
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []

    async def tick(self):
        df = get_recent_candles(limit=120)
        pred = prever_tendencia(self.modelo, df)
        df_prev = rede_markoviana(self.modelo, df, profundidade=5, bifurcacoes=3)

        preco_previsto = float(df_prev["preco"].mean())
        prob_alta = float((df_prev.get("ret", pd.Series([0])) > 0).mean())
        preco = float(pred["preco"])
        energia = float(pred.get("energia", 1.0))
        sinal = self.estrategia.aplicar(pred)

        acao = None
        if sinal == "comprar" and self.posicao == 0:
            qtd = (self.capital * ALOCACAO) / preco
            self.capital -= self.capital * ALOCACAO
            self.posicao = qtd
            self.preco_entrada = preco
            acao = "BUY"

        elif sinal == "vender" and self.posicao > 0:
            self.capital += self.posicao * preco * (1 - CUSTO_TRADE)
            self.posicao = 0.0
            self.preco_entrada = None
            acao = "SELL"

        patrimonio = self.capital + (self.posicao * preco)
        retorno = (patrimonio - CAPITAL_INICIAL) / CAPITAL_INICIAL

        self.tick_idx += 1
        registro = {
            "ep": self.episode,
            "tick": self.tick_idx,
            "tempo": datetime.datetime.utcnow().isoformat(),
            "preco": preco,
            "acao": acao or "-",
            "capital": float(self.capital),
            "posicao": float(self.posicao),
            "patrimonio": float(patrimonio),
            "retorno_pct": float(retorno * 100),
            "prob_alta": float(prob_alta),
            "energia": float(energia),
        }
        print(registro)
        self.historico.append(registro)

        # 🔁 reinício apenas se patrimônio cair abaixo do limite (falência)
        if patrimonio <= LIMIAR_PATRIMONIO:
            self.reset_episode(motivo=f"patrimônio {patrimonio:.2f} abaixo do limite {LIMIAR_PATRIMONIO}")
            registro["episode_reset"] = True
        else:
            registro["episode_reset"] = False

        return registro

simulador = SimuladorRobo()

if __name__ == "__main__":
    async def main():
        robo = SimuladorRobo()
        while True:
            dado = await robo.tick()
            await asyncio.sleep(60)

    asyncio.run(main())
