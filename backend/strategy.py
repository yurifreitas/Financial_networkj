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
                return "comprar"   # 📈 entrar comprado
            elif ret <= -LIMIAR_ENTRADA:
                self.posicao = -1
                return "vender"    # 📉 entrar vendido
            else:
                return "neutro"

        # ======== 2. Posição comprada ========
        if self.posicao == +1:
            if ret < LIMIAR_SAIDA:
                self.posicao = 0
                return "sair"      # 🟡 sair da compra
            else:
                return "manter_compra"

        # ======== 3. Posição vendida ========
        if self.posicao == -1:
            if ret > -LIMIAR_SAIDA:
                self.posicao = 0
                return "sair"      # 🟡 sair da venda
            else:
                return "manter_venda"

        return "neutro"
