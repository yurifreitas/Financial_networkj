# ============================================================
# ⚡ Estratégia Simbiótica Agressiva com Rede Markoviana
# ============================================================
# Age com menos hesitação, entra com confiança parcial
# e considera energia simbiótica e reversão direta.
# ============================================================

import numpy as np

LIMIAR_ENTRADA = 0.003   # 0.8%
LIMIAR_SAIDA   = 0.0015   # 0.4%
LIMIAR_CONFIANCA = 0.49  # 55% dos caminhos positivos já é suficiente

class EstrategiaVariacaoMarkovTurbo:
    def __init__(self):
        self.posicao = 0  # -1 = vendido, 0 = neutro, +1 = comprado

    def aplicar(self, pred):
        ret = pred["retorno_pred"]
        futuros = pred.get("futuro_markov", [])
        energia = pred.get("energia", 1.0)

        # Se não houver previsões, mantém neutro
        if not futuros:
            return "neutro"

        # Probabilidades Markovianas
        positivos = sum(1 for f in futuros if f["ret"] > 0)
        negativos = sum(1 for f in futuros if f["ret"] < 0)
        total = max(positivos + negativos, 1)
        prob_alta = positivos / total
        prob_baixa = negativos / total

        # Intensidade simbiótica ponderada
        direcao = ret * energia
        bias = prob_alta - prob_baixa  # coerência direcional

        # ======== 1️⃣ Sem posição aberta ========
        if self.posicao == 0:
            if direcao > LIMIAR_ENTRADA and prob_alta > LIMIAR_CONFIANCA:
                self.posicao = +1
                return "comprar"
            elif direcao < -LIMIAR_ENTRADA and prob_baixa > LIMIAR_CONFIANCA:
                self.posicao = -1
                return "vender"
            else:
                return "neutro"

        # ======== 2️⃣ Posição comprada ========
        if self.posicao == +1:
            if direcao < LIMIAR_SAIDA or prob_baixa > 0.45:
                self.posicao = 0
                return "sair"
            if bias < -0.3 and ret < -LIMIAR_ENTRADA:
                self.posicao = -1
                return "reverter_venda"
            return "manter_compra"

        # ======== 3️⃣ Posição vendida ========
        if self.posicao == -1:
            if direcao > -LIMIAR_SAIDA or prob_alta > 0.45:
                self.posicao = 0
                return "sair"
            if bias > 0.3 and ret > LIMIAR_ENTRADA:
                self.posicao = +1
                return "reverter_compra"
            return "manter_venda"

        return "neutro"
