# =========================================================
# 🌌 EtherSym Finance — core_env.py (pétricas + persistência + stops dinâmicos)
# =========================================================
import os, json, time
import numpy as np
import random
import logging
from .config_env import (
    COST, SLIP, CUSTO_TRADE, ALOCACAO, CAPITAL_INICIAL,
    TARGET_RET, H_FUTURO, MIN_STEPS,
    ENERGIA_INICIAL, ENERGIA_LIMITE, ENERGIA_DECAIMENTO,
    ENERGIA_BONUS, ENERGIA_PENALTY, PONTUACAO_BONUS,
    RANDOM_START, START_MODE, CYCLE_AT_END,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, HOLD_MIN,
    ENERGIA_RECOMPENSA_SCALING, ENERGIA_REGEN_LIMIT
)

PERSIST_PATH = "metrics_history.jsonl"   # histórico incremental longo
FALENCIA_HARD = 500.0                    # limiar absoluto secundário
VOL_WIN = 64                             # janela p/ vol local
DD_WIN = 2048                            # janela p/ drawdown local

def _rolling_std(x, win):
    if len(x) < win + 1:
        return 0.0
    s = np.std(np.diff(x[-win:]))
    return float(s)

def _max_drawdown(series):
    if len(series) == 0:
        return 0.0
    arr = np.array(series, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())  # valores negativos

class Env:
    def __init__(self, base, price):
        self.b, self.p = base, price
        self.n = len(price)

        self.window_min = 20
        self.window_max = max(self.window_min + 1, self.n - 800)

        if self.window_max <= self.window_min:
            logging.warning(f"[Env] Dataset curto demais ({self.n} candles). Ajustando janelas automaticamente.")
            self.window_min = 0
            self.window_max = max(1, self.n - 2)

        vol = np.abs(np.diff(self.p))
        self._vol = vol / (vol.sum() + 1e-9)

        self._reset_episode_counters()
        self.reset()

    # -------------------------------
    # estado interno auxiliar
    # -------------------------------
    def _reset_episode_counters(self):
        self.metrics_buffer = []
        self.episodios = 0
        self.hist_patrimonio = []  # p/ drawdown
        self.trades_win = 0
        self.trades_lose = 0
        self.trades_total = 0

    # =====================================================
    # 🔁 Reset simbiótico universal
    # =====================================================
    def reset(self):
        if RANDOM_START:
            if START_MODE == "volatility":
                idxs = np.arange(self.window_min, self.window_max)
                p = self._vol[self.window_min:self.window_max]
                if len(idxs) == 0 or p.sum() == 0:
                    logging.warning("[Env] idxs ou p vazio. Reposicionando aleatoriamente.")
                    self.t = np.random.randint(0, max(1, self.n - 1))
                else:
                    self.t = int(np.random.choice(idxs, p=p / (p.sum() + 1e-12)))
            else:
                self.t = np.random.randint(self.window_min, self.window_max)
        else:
            self.t = 0

        # Estado simbiótico inicial
        self.pos = 0.0
        self.energia = ENERGIA_INICIAL
        self.pontuacao = 0.0
        self.done = False
        self.steps = 0
        self.capital = CAPITAL_INICIAL
        self.preco_entrada = None
        self.max_patrimonio = CAPITAL_INICIAL
        self.hist_patrimonio = [CAPITAL_INICIAL]
        self.total_rewards = 0.0
        self.acertos = 0
        self.erros = 0

        logging.info(f"♻ Reset simbiótico em t={self.t} | energia={self.energia:.2f}")
        return self.obs(0)

    # =====================================================
    # 🔍 Observação
    # =====================================================
    def obs(self, a_prev):
        return np.concatenate([self.b[self.t], [self.pos, a_prev]]).astype(np.float32)

    # =====================================================
    # ⚙️ Step simbiótico
    # =====================================================
    def step_once(self, a):
        if self.done:
            return self.obs(0), 0.0, True, {}

        a = int(np.clip(int(a), -1, 1))
        preco = float(self.p[self.t])
        self.t += 1
        self.steps += 1

        if CYCLE_AT_END and self.t >= self.n - H_FUTURO:
            self.t = np.random.randint(self.window_min, self.window_max)

        prev_pos = self.pos
        opened = False
        closed = False

        # === EXECUÇÃO DE TRADE ===
        if a == 1 and self.capital > 0 and self.pos <= 1e-12:
            qtd = (self.capital * ALOCACAO) / (preco + 1e-12)
            custo = qtd * preco * (1 + CUSTO_TRADE)
            if custo <= self.capital:
                self.capital -= custo
                self.pos = qtd
                self.preco_entrada = preco
                opened = True

        elif a == -1 and self.pos > 1e-12:
            receita = self.pos * preco * (1 - CUSTO_TRADE)
            pnl = receita - (self.preco_entrada * self.pos)
            self.capital += receita
            self.pos = 0.0
            self.preco_entrada = None
            closed = True
            if pnl >= 0:
                self.trades_win += 1
            else:
                self.trades_lose += 1
            self.trades_total += 1

        # === STOPS DINÂMICOS (vol + energia) ===
        if self.pos > 1e-12 and self.preco_entrada is not None:
            variacao = (preco - self.preco_entrada) / (self.preco_entrada + 1e-12)

            # vol local e energia influenciam SL/TP efetivos
            vol_loc = _rolling_std(self.p[:self.t], VOL_WIN) + 1e-12
            # normaliza em ~banda razoável
            vol_factor = np.clip(vol_loc / 0.003, 0.5, 2.0)

            # energia baixa → stops mais apertados; alta → mais folga
            energia_factor = np.interp(self.energia, [0.0, ENERGIA_REGEN_LIMIT], [0.8, 1.25])

            sl_eff = STOP_LOSS_PCT * vol_factor / energia_factor
            tp_eff = TAKE_PROFIT_PCT * vol_factor * energia_factor

            if (variacao <= -sl_eff) and (self.steps > HOLD_MIN):
                receita = self.pos * preco * (1 - CUSTO_TRADE)
                pnl = receita - (self.preco_entrada * self.pos)
                self.capital += receita
                self.pos = 0.0
                self.preco_entrada = None
                logging.info(f"🚫 Stop loss | {variacao*100:.2f}% | cap={self.capital:.2f}")
                self.energia -= ENERGIA_PENALTY * 0.5
                self.erros += 1
                self.trades_lose += 1
                self.trades_total += 1
                closed = True

            elif (variacao >= tp_eff) and (self.steps > HOLD_MIN):
                receita = self.pos * preco * (1 - CUSTO_TRADE)
                pnl = receita - (self.preco_entrada * self.pos)
                self.capital += receita
                self.pos = 0.0
                self.preco_entrada = None
                logging.info(f"💰 Take profit | {variacao*100:.2f}% | cap={self.capital:.2f}")
                self.energia += ENERGIA_BONUS * 0.8
                self.acertos += 1
                self.trades_win += 1
                self.trades_total += 1
                closed = True

        # === PATRIMÔNIO E RECOMPENSA ===
        patrimonio = self.capital + self.pos * preco
        self.max_patrimonio = max(self.max_patrimonio, patrimonio)
        self.hist_patrimonio.append(patrimonio)

        futuro = min(self.t + H_FUTURO, self.n - 1)
        ret_futuro = (self.p[futuro] - preco) / (preco + 1e-9)
        previsao = a * TARGET_RET
        erro_abs = abs(ret_futuro - previsao)

        delta_pos = abs(self.pos - prev_pos)
        trade_cost = CUSTO_TRADE if (delta_pos > 1e-12) else 0.0

        # recompensa base conforme acurácia da previsão
        reward_base = (TARGET_RET - erro_abs) / (TARGET_RET + 1e-9) - float(trade_cost)

        # modulação por energia (mais energia → amplifica bons acertos)
        reward = reward_base * (1.0 + ENERGIA_RECOMPENSA_SCALING * (self.energia - 1.0))
        reward = float(np.clip(reward, -1.5, 1.5))
        self.total_rewards += reward
        # =====================================================
        # 🌿 Feedback simbiótico dinâmico (inserido)
        # =====================================================

        # 1️⃣ Clamp energético adaptativo na previsão
        #    (regula o delta de risco conforme energia simbiótica)
        risk_gain = np.interp(self.energia, [0.0, ENERGIA_REGEN_LIMIT], [0.4, 1.5])
        reward *= risk_gain

        # 2️⃣ Reforço simbiótico: energia influencia motivação
        #    (quanto maior a energia, mais o agente confia em si)
        reward += 0.15 * (self.energia - 1.0)

        # 3️⃣ Penalidade por energia crítica
        #    (se energia cair demais, o agente sente “fadiga”)
        if self.energia < 0.2:
            reward -= 0.2 * (0.2 - self.energia)

        # 4️⃣ Feedback metabólico no erro absoluto
        #    (reduz punição se energia alta = confiança)
        reward -= 0.05 * erro_abs * (1.0 - self.energia)

        # 5️⃣ Limite final simbiótico de estabilidade
        reward = float(np.clip(reward, -2.0, 2.0))

        # === ENERGIA ===
        self.energia -= ENERGIA_DECAIMENTO
        if erro_abs < TARGET_RET:
            self.energia += ENERGIA_BONUS * (1 - erro_abs / TARGET_RET)
            self.pontuacao += PONTUACAO_BONUS
        else:
            self.energia -= ENERGIA_PENALTY * min(erro_abs / TARGET_RET, 2.0)

        self.energia = float(np.clip(self.energia, 0.0, ENERGIA_REGEN_LIMIT))

        # === CONDIÇÕES DE TÉRMINO ===
        done_env = False
        # =====================================================
        # 🏆 Condição de Vitória — Patrimônio Duplicado
        # =====================================================
        FATOR_VITORIA = 20  # dobra o capital inicial
        if patrimonio >= FATOR_VITORIA * CAPITAL_INICIAL:
            done_env = True
            logging.info(f"🏆 Vitória simbiótica! Patrimônio dobrado ({patrimonio:.2f}) no episódio {self.episodios + 1}")

            vitoria_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "episodio": int(self.episodios + 1),
                "capital_final": float(self.capital),
                "patrimonio_final": float(patrimonio),
                "max_patrimonio": float(self.max_patrimonio),
                "energia_final": float(self.energia),
                "pontuacao": float(self.pontuacao),
                "taxa_acerto": float(self.acertos / max(1, (self.acertos + self.erros))),
                "trades_win": int(self.trades_win),
                "trades_lose": int(self.trades_lose),
                "trades_total": int(self.trades_total)
            }

            try:
                os.makedirs("runs", exist_ok=True)
                with open(f"runs/vitoria_ep{self.episodios + 1}_{int(time.time())}.json", "w") as f:
                    json.dump(vitoria_data, f, indent=2)
                logging.info("💾 Vitória simbiótica registrada em runs/")
            except Exception as e:
                logging.info(f"[WARN] Falha ao salvar vitória simbiótica: {e}")

        # falência “hard” só permite fim após MIN_STEPS; também respeita limite absoluto
        if (self.steps >= MIN_STEPS and patrimonio <= ENERGIA_LIMITE * CAPITAL_INICIAL) or (patrimonio <= FALENCIA_HARD):
            done_env = True

        # === RECORDES / PERSISTÊNCIA ===
        if done_env:
            self.episodios += 1
            # pétricas de episódio
            taxa_acerto = self.acertos / max(1, (self.acertos + self.erros))
            eficiencia = self.total_rewards / max(1, self.steps)
            # sharpe “rápido”: média(retornos) / std(retornos) ~ aqui usando reward como proxy
            # para não guardar toda série de rewards, aproximamos via eficiencia e vol local
            vol_loc = _rolling_std(self.p[:self.t], VOL_WIN) + 1e-12
            sharpe_like = float(eficiencia / (vol_loc + 1e-12))
            max_dd = _max_drawdown(self.hist_patrimonio)  # negativo
            entropia = float(
                -(
                    (taxa_acerto * np.log2(taxa_acerto + 1e-9)) +
                    ((1 - taxa_acerto) * np.log2(1 - taxa_acerto + 1e-9))
                )
            )
            registro = {
                "ep": int(self.episodios),
                "steps": int(self.steps),
                "pontuacao": float(self.pontuacao),
                "energia_final": float(self.energia),
                "taxa_acerto": float(taxa_acerto),
                "eficiencia": float(eficiencia),
                "entropia": float(np.clip(entropia, 0.0, 1.0)),
                "sharpe_like": sharpe_like,
                "capital_final": float(self.capital),
                "patrimonio_final": float(patrimonio),
                "max_patrimonio": float(self.max_patrimonio),
                "max_drawdown": float(max_dd),  # negativo
                "trades_total": int(self.trades_total),
                "trades_win": int(self.trades_win),
                "trades_lose": int(self.trades_lose),
                "timestamp": time.time(),
            }
            self.metrics_buffer.append(registro)
            # salva a cada 5 episódios
            if len(self.metrics_buffer) >= 5:
                import threading
                threading.Thread(target=self._persist_metrics, daemon=True).start()


            self.pos = 0.0
            self.preco_entrada = None
            self.capital = CAPITAL_INICIAL
            self.energia = ENERGIA_INICIAL
            self.pontuacao = 0.0
            self.total_rewards = 0.0
            self.acertos = 0
            self.erros = 0
            self.trades_total = self.trades_win = self.trades_lose = 0

            # reposiciona no dataset
            self.t = np.random.randint(self.window_min, self.window_max)
            self.max_patrimonio = CAPITAL_INICIAL
            self.hist_patrimonio = [CAPITAL_INICIAL]

        info = {
            "ret": float(ret_futuro),
            "erro": float(erro_abs),
            "energia": float(self.energia),
            "pontuacao": float(self.pontuacao),
            "capital": float(self.capital),
            "patrimonio": float(patrimonio),
            "max_patrimonio": float(self.max_patrimonio),
            "steps": int(self.steps),
        }

        self.done = done_env
        return self.obs(a), reward, self.done, info

    def step(self, a, repeats=1):
        total_r, s_next, done, info = 0.0, None, False, {}
        for _ in range(max(1, repeats)):
            s_next, r, done, info = self.step_once(a)
            total_r += r
            if done:
                break
        return s_next, total_r, done, info

    # =====================================================
    # 💾 Persistência de longo prazo
    # =====================================================
    def _persist_metrics(self):
        if not self.metrics_buffer:
            return
        count = len(self.metrics_buffer)
        with open(PERSIST_PATH, "a") as f:
            for rec in self.metrics_buffer:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.metrics_buffer.clear()
        logging.info(f"🧩 {count} pétricas salvas em {PERSIST_PATH}")
