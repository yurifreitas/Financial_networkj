# ============================================================
# Estratégia simbiótica com limiar de 2%
# ============================================================

LIMIAR_ENTRADA = 0.02   # 2%
LIMIAR_SAIDA   = 0.015  # 1.5%

class EstrategiaVariacao:
    def __init__(self):
        self.posicao = 0  # -1 = short, 0 = neutro, +1 = long

    def aplicar(self, pred):
        ret = pred["retorno_pred"]
        # ======== 1. Sem posição aberta ========
        if self.posicao == 0:
            if ret >= LIMIAR_ENTRADA:
                self.posicao = +1
                return +1  # 📈 entrar comprado
            elif ret <= -LIMIAR_ENTRADA:
                self.posicao = -1
                return -1  # 📉 entrar vendido
            else:
                return 0  # nada a fazer

        # ======== 2. Posição aberta ========
        if self.posicao == +1 and ret < LIMIAR_SAIDA:
            self.posicao = 0
            return 0  # 🟡 sair da compra
        if self.posicao == -1 and ret > -LIMIAR_SAIDA:
            self.posicao = 0
            return 0  # 🟡 sair da venda

        return self.posicao
