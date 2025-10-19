# =========================================================
# 🌌 EtherSym Finance — core/env_parallel.py (v8f simbiótico corrigido)
# =========================================================
# - Execução paralela de múltiplos ambientes Env
# - Núcleo Numba vetorizado e seguro (sem ambiguidade booleana)
# - Totalmente compatível com collector e replay simbiótico
# =========================================================

import numpy as np
from numba import njit, prange
from .core_env import Env

# =========================================================
# ⚙️ Núcleo Numba JIT — cálculo vetorizado simbiótico
# =========================================================
@njit(fastmath=True, nogil=True, parallel=True)
def _step_core_batch(precos, posicoes, capitais, entradas, acoes, custo_trade, alocacao):
    """
    Núcleo paralelo: processa N ambientes de forma vetorizada e estável.
    Corrige ambiguity errors e mantém coerência energética.
    """
    n = precos.shape[0]
    novos_capitais = np.zeros_like(capitais)
    recompensas = np.zeros_like(capitais)
    done = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        preco = precos[i]
        pos = posicoes[i]
        capital = capitais[i]
        entrada = entradas[i]
        a = acoes[i]

        # --- lógica de compra
        if a == 1 and pos <= 1e-12:
            qtd = (capital * alocacao) / (preco + 1e-12)
            custo = qtd * preco * (1 + custo_trade)
            if custo <= capital:
                capital -= custo
                pos = qtd
                entrada = preco

        # --- lógica de venda
        elif a == -1 and pos > 1e-12:
            receita = pos * preco * (1 - custo_trade)
            capital += receita
            pos = 0.0
            entrada = 0.0

        # --- atualização patrimonial
        patrimonio = capital + pos * preco
        recompensas[i] = patrimonio - capitais[i]
        novos_capitais[i] = capital

        # --- flag simbiótica de falência
        done[i] = patrimonio <= 500.0

        # --- sincroniza backbuffers
        posicoes[i] = pos
        capitais[i] = capital
        entradas[i] = entrada

    # 🔹 garante coerência de retorno numérico
    return novos_capitais, recompensas, done


# =========================================================
# 🌱 Classe simbiótica paralela
# =========================================================
class EnvParallel:
    def __init__(self, base, price, n_envs=8):
        self.n_envs = n_envs
        self.envs = [Env(base, price) for _ in range(n_envs)]
        self.device = "cpu"
        self.last_actions = np.zeros(n_envs, dtype=np.int64)
        self._sync_initial_state()

    # -----------------------------------------------------
    def _sync_initial_state(self):
        self.precos = np.array([env.p[env.t] for env in self.envs], dtype=np.float32)
        self.capitais = np.array([env.capital for env in self.envs], dtype=np.float32)
        self.posicoes = np.array([env.pos for env in self.envs], dtype=np.float32)
        self.entradas = np.array(
            [env.preco_entrada or 0.0 for env in self.envs], dtype=np.float32
        )

    # =========================================================
    # 🔁 Reset coletivo simbiótico
    # =========================================================
    def reset(self):
        for env in self.envs:
            env.reset()
        self._sync_initial_state()
        return np.stack([env.obs(0) for env in self.envs])

    # =========================================================
    # ⚙️ Step batelado simbiótico
    # =========================================================
    def step_batch(self, actions):
        """
        Executa múltiplos steps em paralelo com Numba otimizado e coerente.
        """
        actions = np.clip(actions, -1, 1).astype(np.int64)
        self.precos = np.array([env.p[env.t] for env in self.envs], dtype=np.float32)

        # --- executa núcleo JIT vetorizado
        novos_capitais, rewards, done = _step_core_batch(
            self.precos,
            self.posicoes,
            self.capitais,
            self.entradas,
            actions,
            self.envs[0].__dict__.get("CUSTO_TRADE", 0.0004),
            self.envs[0].__dict__.get("ALOCACAO", 0.9),
        )

        # 🔹 atualiza buffers simbióticos
        self.capitais[:] = novos_capitais
        for i, env in enumerate(self.envs):
            env.capital = self.capitais[i]
            env.pos = self.posicoes[i]
            env.preco_entrada = self.entradas[i] if self.entradas[i] > 0 else None
            env.t = min(env.t + 1, env.n - 1)
            env.energia = float(np.clip(env.energia - 0.001, 0, 1))
            env.hist_patrimonio.append(self.capitais[i])
            env.done = bool(done[i])  # <-- correção principal

        # 🔹 normaliza recompensas (sem NaN)
        rewards = np.tanh(rewards / (np.abs(self.capitais) + 1e-9)).astype(np.float32)

        # 🔹 observações e info simbiótica
        obs = np.stack([env.obs(actions[i]) for i, env in enumerate(self.envs)])
        info = {
            "capital_medio": float(self.capitais.mean()),
            "energia_media": float(np.mean([env.energia for env in self.envs])),
            "done_count": int(np.sum(done)),
            "falidos": int(np.count_nonzero(done)),
        }

        return obs, rewards, done, info

    # =========================================================
    # 🔁 Alias compatível com Env normal
    # =========================================================
    def step(self, a, repeats=1):
        """
        Mantém compatibilidade com o collector do main atual.
        Executa steps batelados, mas retorna o primeiro ambiente.
        """
        total_r, s_next, done, info = 0.0, None, False, {}
        for _ in range(max(1, repeats)):
            s_next, r, done, info = self.step_once(a)
            total_r += r
            if done:
                break
        return s_next, total_r, done, info

    # =========================================================
    # 🔹 Compatibilidade com env normal
    # =========================================================
    def step_once(self, action):
        """
        Executa step_batch replicando a ação para todos os ambientes.
        Retorna apenas o primeiro para compatibilidade com loops legados.
        """
        obs, r, d, info = self.step_batch(np.full(self.n_envs, action, dtype=np.int64))
        return obs[0], float(r[0]), bool(d[0]), info
