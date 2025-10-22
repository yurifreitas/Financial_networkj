# ============================================================
# 🌌 EtherSym Finance — Estratégia simbiótica adaptativa v5.5
# ============================================================
# - Ativação automática de limiar direcional
# - Reforço simbiótico pela coerência média real
# - Autoajuste da inércia e momentum
# ============================================================

import numpy as np
from collections import deque

class EstrategiaVariacao:
    def __init__(self,
        vol_base=0.003,
        limiar_base=0.0015,
        hist_energia=30,
        hist_ret=40,
        energia_min=0.25,
        energia_max=1.8,
        sl_base=0.005,
        tp_base=0.008,
        min_stop=0.001,
        max_stop=0.02,
        min_take=0.002,
        max_take=0.04,
        agress_min=0.3,
        agress_max=2.5,
        coef_reversao=1.6,
        inercia_base=0.65,
        momentum_gain=1.25,
        suavizacao=0.25,
        atividade_min=0.05,   # ativa reação mínima
    ):
        self.vol_base = vol_base
        self.limiar_base = limiar_base
        self.hist_energia = hist_energia
        self.hist_ret = hist_ret
        self.energia_min = energia_min
        self.energia_max = energia_max
        self.sl_base = sl_base
        self.tp_base = tp_base
        self.min_stop = min_stop
        self.max_stop = max_stop
        self.min_take = min_take
        self.max_take = max_take
        self.agress_min = agress_min
        self.agress_max = agress_max
        self.coef_reversao = coef_reversao
        self.inercia_base = inercia_base
        self.momentum_gain = momentum_gain
        self.suavizacao = suavizacao
        self.atividade_min = atividade_min

        self.posicao = 0
        self.preco_entrada = None
        self._ultima_energia = 1.0
        self._historico_energia = deque(maxlen=hist_energia)
        self._historico_ret = deque(maxlen=hist_ret)
        self._momentum = 0.0
        self._forca_direcional = 0.0
        self._agressividade = 1.0

    # ============================================================
    # Núcleo simbiótico refinado
    # ============================================================
    def aplicar(self, pred):
        preco = float(pred.get("preco", 0.0))
        ret = float(pred.get("retorno_pred", 0.0))
        coerencia = float(pred.get("coerencia", 0.8)) or 0.8
        energia = float(pred.get("energia", 1.0))
        vol = float(pred.get("vol", self.vol_base))

        if preco <= 0.0:
            return "neutro"

        # --- Energia e coerência médias ---
        energia = (1 - self.suavizacao) * self._ultima_energia + self.suavizacao * energia
        self._ultima_energia = energia
        self._historico_energia.append(energia)
        self._historico_ret.append(ret)
        mean_energy = np.mean(self._historico_energia)
        mean_coer = np.mean([abs(x) for x in self._historico_ret]) if self._historico_ret else abs(ret)

        # --- Ajuste agressividade ---
        self._agressividade = np.clip(
            (mean_energy * (coerencia + 0.2)) / (0.6 + abs(mean_coer) * 8),
            self.agress_min,
            self.agress_max,
        )

        # --- Limiar adaptativo real ---
        base_limiar = self.limiar_base * np.clip(1.0 / (self._agressividade + 1e-6), 0.5, 2.0)
        # aumenta a sensibilidade se coerência média for muito baixa
        limiar_entrada = base_limiar * (0.7 + (1.0 - coerencia) * 0.8)

        # --- Stops/Takes dinâmicos ---
        energia_factor = np.interp(energia, [self.energia_min, self.energia_max], [0.8, 1.25])
        sl_eff = np.clip(self.sl_base / energia_factor, self.min_stop, self.max_stop)
        tp_eff = np.clip(self.tp_base * energia_factor, self.min_take, self.max_take)

        # --- Momentum e direção ---
        self._momentum = 0.9 * self._momentum + 0.1 * np.sign(ret) * coerencia * energia
        momentum_factor = 1.0 + abs(self._momentum) * (self.momentum_gain - 1.0)
        self._forca_direcional = (
            0.7 * self._forca_direcional + 0.3 * (ret * coerencia * energia * momentum_factor)
        )
        fs = np.tanh(self._forca_direcional * 5)  # compressão simbiótica
        ret_ponderado = ret * coerencia * energia

        # ============================================================
        # 🔹 SEM POSIÇÃO
        # ============================================================
        if self.posicao == 0:
            # força mínima simbiótica
            if abs(fs) < self.atividade_min:
                fs *= (1.0 + np.random.uniform(0.2, 0.5))
            if fs >= limiar_entrada:
                self.posicao = +1
                self.preco_entrada = preco
                return "comprar"
            elif fs <= -limiar_entrada:
                self.posicao = -1
                self.preco_entrada = preco
                return "vender"
            return "neutro"

        # ============================================================
        # 🔸 COMPRADO
        # ============================================================
        if self.posicao == +1:
            variacao = (preco - self.preco_entrada) / (self.preco_entrada + 1e-12)
            if variacao <= -sl_eff:
                self.posicao = 0; return "stop_loss"
            if variacao >= tp_eff:
                self.posicao = 0; return "take_profit"
            if ret_ponderado < -limiar_entrada * self.coef_reversao:
                prob_flip = np.clip(1.0 - self.inercia_base * energia_factor, 0.0, 1.0)
                if np.random.rand() < prob_flip:
                    self.posicao = -1; self.preco_entrada = preco
                    return "flip_para_venda"
            return "manter_compra"

        # ============================================================
        # 🔻 VENDIDO
        # ============================================================
        if self.posicao == -1:
            variacao = (self.preco_entrada - preco) / (self.preco_entrada + 1e-12)
            if variacao <= -sl_eff:
                self.posicao = 0; return "stop_loss"
            if variacao >= tp_eff:
                self.posicao = 0; return "take_profit"
            if ret_ponderado > limiar_entrada * self.coef_reversao:
                prob_flip = np.clip(1.0 - self.inercia_base * energia_factor, 0.0, 1.0)
                if np.random.rand() < prob_flip:
                    self.posicao = +1; self.preco_entrada = preco
                    return "flip_para_compra"
            return "manter_venda"

        return "neutro"
