# =========================================================
# 🤖 EtherSym Finance — simulador_realtime.py
# =========================================================

import asyncio, datetime, json, numpy as np, pandas as pd
from backend.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictor import prever_tendencia
from backend.strategy import EstrategiaVariacao

CAPITAL_INICIAL = 10000
ALOCACAO = 0.5
CUSTO_TRADE = 0.0004

class SimuladorRobo:
    def __init__(self):
        self.modelo = carregar_modelo()
        self.estrategia = EstrategiaVariacao()
        self.capital = CAPITAL_INICIAL
        self.posicao = 0.0
        self.preco_entrada = None
        self.historico = []

    async def tick(self):
        df = get_recent_candles(limit=120)
        pred = prever_tendencia(self.modelo, df)
        sinal = self.estrategia.aplicar(pred)
        preco = float(pred["preco"])

        acao = None
        if sinal == "comprar" and self.posicao == 0:
            self.posicao = (self.capital * ALOCACAO) / preco
            self.capital -= self.capital * ALOCACAO
            self.preco_entrada = preco
            acao = "BUY"

        elif sinal == "vender" and self.posicao > 0:
            self.capital += self.posicao * preco * (1 - CUSTO_TRADE)
            self.posicao = 0.0
            acao = "SELL"

        patrimonio = self.capital + (self.posicao * preco)
        retorno = (patrimonio - CAPITAL_INICIAL) / CAPITAL_INICIAL

        registro = {
            "tempo": datetime.datetime.utcnow().isoformat(),
            "preco": preco,
            "preco_entrada": self.preco_entrada,
            "acao": acao or "-",
            "capital": self.capital,
            "posicao": self.posicao,
            "patrimonio": patrimonio,
            "retorno_pct": retorno * 100,
            "retorno_pred": pred["retorno_pred"],
            "energia": getattr(pred, "energia", 1.0),
        }

        self.historico.append(registro)
        return registro

simulador = SimuladorRobo()
