# =========================================================
# 🕰️ EtherSym Finance — backend/simulador_replay.py
# =========================================================
# - Executa o modelo com dados históricos (passado)
# - Usa estratégia simbiótica adaptativa v5.2 (com parâmetros variáveis)
# - Salva previsões e patrimônio tick a tick
# =========================================================

import os, datetime, asyncio
import pandas as pd

# 🔗 Imports internos
from backend.loader.model_loader import carregar_modelo
from backend.predictors.predictor import prever_tendencia
from backend.strategies.strategy import EstrategiaVariacao

# =========================================================
# ⚙️ Configurações gerais do replay
# =========================================================
CAPITAL_INICIAL = 10_000.0
ALOCACAO = 0.9
CUSTO_TRADE = 0.0004
SAVE_RUNS = True
RUNS_DIR = os.path.join("backend", "runs_replay")
WINDOW = 60  # tamanho da janela de contexto (em candles)

# =========================================================
# ⚙️ Parâmetros simbióticos da estratégia (ajustáveis)
# =========================================================
ESTRATEGIA_CONFIG = {
    "vol_base": 0.003,
    "limiar_base": 0.0015,
    "sl_base": 0.005,
    "tp_base": 0.008,
    "min_stop": 0.001,
    "max_stop": 0.02,
    "min_take": 0.002,
    "max_take": 0.04,
    "agress_min": 0.3,
    "agress_max": 2.0,
    "coef_reversao": 2.0,
    "energia_min": 0.25,
    "energia_max": 1.8,
    "hist_energia": 30,
    "hist_ret": 30,
}

# =========================================================
# 🤖 Simulador de Replay (modo simbiótico)
# =========================================================
class SimuladorReplay:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        if "timestamp" not in self.df.columns:
            raise ValueError("CSV precisa ter coluna 'timestamp'")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)

        self.modelo = carregar_modelo()
        # Estratégia simbiótica dinâmica configurável
        self.estrategia = EstrategiaVariacao(**ESTRATEGIA_CONFIG)

        self.capital = CAPITAL_INICIAL
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []

        os.makedirs(RUNS_DIR, exist_ok=True)
        print(f"🧩 Replay iniciado com {len(self.df)} candles no arquivo {csv_path}")

    async def executar(self):
        total = len(self.df)
        for i in range(WINDOW, total):
            df_slice = self.df.iloc[i - WINDOW:i].copy()
            preco_atual = float(df_slice["close"].iloc[-1])

            # === 🔮 Predição simbiótica direta ===
            pred = prever_tendencia(self.modelo, df_slice)
            pred["retorno_pred"] = float(pred.get("retorno_pred", 0.0))

            # Normaliza se vier em porcentagem (ex: 12.3 → 0.123)
            if abs(pred["retorno_pred"]) > 10:
                pred["retorno_pred"] /= 100.0

            preco_previsto = float(pred.get("preco_previsto", preco_atual))
            energia = float(pred.get("energia", 1.0))
            coerencia = float(pred.get("coerencia", 1.0))
            retorno_pred = pred["retorno_pred"]

            # === Aplicar estratégia simbiótica ===
            sinal = self.estrategia.aplicar(pred)
            acao = "-"

            # === Execução de trades ===
            if sinal == "comprar" and self.posicao == 0:
                qtd = (self.capital * ALOCACAO) / preco_atual
                self.capital -= self.capital * ALOCACAO
                self.posicao = qtd
                self.preco_entrada = preco_atual
                acao = "BUY"

            elif sinal == "vender" and self.posicao > 0:
                self.capital += self.posicao * preco_atual * (1.0 - CUSTO_TRADE)
                self.posicao = 0
                self.preco_entrada = None
                acao = "SELL"

            elif sinal in ("stop_loss", "take_profit", "flip_para_venda", "flip_para_compra"):
                if self.posicao > 0:
                    self.capital += self.posicao * preco_atual * (1.0 - CUSTO_TRADE)
                self.posicao = 0
                self.preco_entrada = None
                acao = sinal.upper()

            # === Atualizar patrimônio ===
            patrimonio = self.capital + self.posicao * preco_atual
            retorno_pct = (patrimonio - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100.0

            # === Registro simbiótico ===
            self.historico.append({
                "tick": i,
                "timestamp": self.df["timestamp"].iloc[i],
                "preco": preco_atual,
                "preco_previsto": preco_previsto,
                "acao": acao,
                "capital": self.capital,
                "patrimonio": patrimonio,
                "retorno_pct": retorno_pct,
                "retorno_pred": retorno_pred,
                "energia": energia,
                "coerencia": coerencia,
            })

            print(
                f"[Tick {i:05d}/{total}] preço={preco_atual:.2f} | prev={preco_previsto:.2f} "
                f"| ação={acao:>10} | pat={patrimonio:.2f} | Δpred={retorno_pred:+.4f} | "
                f"Enr={energia:.2f} | Coer={coerencia:.2f}"
            )

            await asyncio.sleep(0)  # evita travar loop async

        # === Salvar resultado final ===
        if SAVE_RUNS:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RUNS_DIR, f"replay_{ts}.csv")
            pd.DataFrame(self.historico).to_csv(path, index=False)
            print(f"💾 Replay completo salvo em: {path}")


# =========================================================
# 🚀 Execução principal
# =========================================================
if __name__ == "__main__":
    async def main():
        replay = SimuladorReplay(
            "/home/yuri/Documents/code2/binance-model/binance_BTC_USDT_1h_2y.csv"
        )
        await replay.executar()

    asyncio.run(main())
