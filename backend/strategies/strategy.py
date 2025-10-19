# ============================================================
# 🌌 EtherSym Finance — Estratégia simbiótica adaptativa v5.3
# ============================================================
# - Limiar e agressividade auto-ajustáveis
# - Momentum simbiótico e inércia direcional
# - Stop/take dinâmicos com suavização adaptativa
# ============================================================

import numpy as np
from collections import deque

class EstrategiaVariacao:
    def __init__(
        self,
        vol_base: float = 0.003,
        limiar_base: float = 0.0015,
        energia_min: float = 0.25,
        energia_max: float = 1.8,
        hist_energia: int = 30,
        hist_ret: int = 40,
        sl_base: float = 0.005,
        tp_base: float = 0.008,
        min_stop: float = 0.001,
        max_stop: float = 0.02,
        min_take: float = 0.002,
        max_take: float = 0.04,
        agress_min: float = 0.3,
        agress_max: float = 2.5,
        coef_reversao: float = 2.0,
        inercia_base: float = 0.6,      # resistência à mudança de posição
        momentum_gain: float = 1.3,     # força de sequência coerente
    ):
        # parâmetros principais
        self.vol_base = vol_base
        self.limiar_base = limiar_base
        self.energia_min = energia_min
        self.energia_max = energia_max
        self.hist_energia = hist_energia
        self.hist_ret = hist_ret
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

        # estados simbióticos
        self.posicao = 0
        self.preco_entrada = None
        self.steps_em_posicao = 0
        self._ultima_energia = 1.0
        self._historico_energia = deque(maxlen=hist_energia)
        self._historico_ret = deque(maxlen=hist_ret)
        self._agressividade = 1.0
        self._forca_direcional = 0.0
        self._momentum = 0.0

    # ============================================================
    # 🧩 Núcleo simbiótico principal
    # ============================================================
    def aplicar(self, pred):
        preco = float(pred.get("preco", 0.0))
        ret = float(pred.get("retorno_pred", 0.0))
        coerencia = float(pred.get("coerencia", 1.0))
        energia = float(pred.get("energia", 1.0))
        vol = float(pred.get("vol", self.vol_base))

        if preco <= 0.0:
            return "neutro"

        # === 1️⃣ Energia simbiótica média ===
        energia = 0.7 * self._ultima_energia + 0.3 * energia
        self._ultima_energia = energia
        self._historico_energia.append(energia)
        self._historico_ret.append(ret)

        mean_energy = np.mean(self._historico_energia)
        std_ret = np.std(self._historico_ret) if len(self._historico_ret) > 5 else abs(ret)

        # === 2️⃣ Agressividade simbiótica dinâmica ===
        self._agressividade = np.clip(
            (mean_energy * (coerencia + 0.3)) / (0.8 + std_ret * 10),
            self.agress_min,
            self.agress_max,
        )

        # === 3️⃣ Limiar adaptativo de entrada ===
        base_limiar = self.limiar_base * np.clip(1.0 / (self._agressividade + 1e-6), 0.5, 2.0)
        limiar_entrada = base_limiar * np.clip(vol / self.vol_base, 0.6, 1.5)

        # === 4️⃣ Stops e takes dinâmicos ===
        sl_eff = np.clip(self.sl_base * (self.vol_base / (vol + 1e-9)) / energia, self.min_stop, self.max_stop)
        tp_eff = np.clip(self.tp_base * (vol / self.vol_base) * energia, self.min_take, self.max_take)

        # === 5️⃣ Momentum simbiótico ===
        self._momentum = (
            0.8 * self._momentum + 0.2 * np.sign(ret) * coerencia * energia
        )
        momentum_factor = 1.0 + abs(self._momentum) * (self.momentum_gain - 1.0)

        # === 6️⃣ Força direcional simbiótica ===
        self._forca_direcional = (
            0.6 * self._forca_direcional
            + 0.4 * (ret * coerencia * energia * momentum_factor)
        )
        fs = self._forca_direcional
        ret_ponderado = ret * coerencia * energia

        # ============================================================
        # 🔹 SEM POSIÇÃO
        # ============================================================
        if self.posicao == 0:
            if fs >= limiar_entrada:
                self.posicao = +1
                self.preco_entrada = preco
                self.steps_em_posicao = 0
                return "comprar"
            elif fs <= -limiar_entrada:
                self.posicao = -1
                self.preco_entrada = preco
                self.steps_em_posicao = 0
                return "vender"
            return "neutro"

        # ============================================================
        # 🔸 COMPRADO
        # ============================================================
        if self.posicao == +1:
            self.steps_em_posicao += 1
            variacao = (preco - self.preco_entrada) / (self.preco_entrada + 1e-12)

            # reversão rápida
            if ret_ponderado < -limiar_entrada * self.coef_reversao and coerencia > 0.6:
                if np.random.rand() > self.inercia_base:
                    self.posicao = -1
                    self.preco_entrada = preco
                    self.steps_em_posicao = 0
                    return "flip_para_venda"

            # stop/take
            if variacao <= -sl_eff:
                self.posicao = 0
                self.preco_entrada = None
                return "stop_loss"
            elif variacao >= tp_eff:
                self.posicao = 0
                self.preco_entrada = None
                return "take_profit"

            return "manter_compra"

        # ============================================================
        # 🔻 VENDIDO
        # ============================================================
        if self.posicao == -1:
            self.steps_em_posicao += 1
            variacao = (self.preco_entrada - preco) / (self.preco_entrada + 1e-12)

            if ret_ponderado > limiar_entrada * self.coef_reversao and coerencia > 0.6:
                if np.random.rand() > self.inercia_base:
                    self.posicao = +1
                    self.preco_entrada = preco
                    self.steps_em_posicao = 0
                    return "flip_para_compra"

            if variacao <= -sl_eff:
                self.posicao = 0
                self.preco_entrada = None
                return "stop_loss"
            elif variacao >= tp_eff:
                self.posicao = 0
                self.preco_entrada = None
                return "take_profit"

            return "manter_venda"

        return "neutro"
