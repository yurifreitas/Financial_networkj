# =========================================================
# 🌌 EtherSym Finance — env.py (v4 simbiótico robusto)
# =========================================================
# - Episódios longos e orgânicos
# - Início aleatório ponderado por volatilidade
# - Energia simbiótica viva (decai e se regenera)
# - Pontuação e melhor score persistentes
# - Reset suave e contínuo, nunca do zero
# =========================================================

import numpy as np
import pandas as pd
import json, random, os, time

# =========================
# ⚙️ Parâmetros simbióticos
# =========================
COST = 0.0005
SLIP = 0.0002
TARGET_RET = 0.003
H_FUTURO = 3

# Energia e dinâmica simbiótica
ENERGIA_INICIAL = 1.0
ENERGIA_LIMITE = 0.25         # morre se cair abaixo disso
ENERGIA_DECAIMENTO = 0.0008   # perda passiva a cada step
ENERGIA_BONUS = 0.012         # ganho por acerto (erro < TARGET_RET)
ENERGIA_PENALTY = 0.025       # perda por erro alto
PONTUACAO_BONUS = 1.0         # pontos simbólicos por acerto

# Início aleatório
RANDOM_START = True
START_MODE = "volatility"     # "uniform" ou "volatility"
CYCLE_AT_END = True           # ciclo no fim (não termina o dataset)

# Duração mínima simbiótica
MIN_STEPS = 400

# Arquivo de persistência
BEST_SCORE_FILE = "best_score.json"


# =========================================================
# 🧩 Geração de features (compatível com main/network)
# =========================================================
def make_feats(df: pd.DataFrame):
    df.columns = [c.lower() for c in df.columns]

    # === Retornos e volatilidade ===
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)

    # === Médias exponenciais e diferencial de tendência ===
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)

    # === RSI normalizado ===
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_n"] = (rsi - 50.0) / 50.0

    # === Faixa e corpo do candle ===
    df["range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["body"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + 1e-9)

    # === Volume normalizado ===
    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(48).mean()) /
        (df["volume"].rolling(48).std() + 1e-9)
    ).fillna(0.0)

    # === Z-score de preço (desvio padrão local) ===
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"] - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    # === Remover NaN e alinhar ===
    df = df.dropna().reset_index(drop=True)

    # === Vetor de features (8 dimensões simbióticas) ===
    base = df[["ret", "vol_ret", "range", "body", "ema_diff", "rsi_n", "z", "volume_z"]].astype(np.float32).values
    price = df["close"].astype(np.float32).values
    return base, price



# =========================================================
# 🎮 Ambiente simbiótico prolongado (v5 — correção do início)
# =========================================================
class Env:
    def __init__(self, base, price):
        self.b, self.p = base, price
        self.n = len(price)

        self.window_min = 48
        self.window_max = self.n - 800

        vol = np.abs(np.diff(self.p))
        self._vol = vol / (vol.sum() + 1e-9)

        self.best_score = self._load_best_score()
        self.score_file = BEST_SCORE_FILE
        self.initialized = False  # controle simbiótico de primeiro reset

        # agora o primeiro reset também é aleatório
        self.reset()

    # -----------------------------
    # 🔁 Reset simbiótico universal
    # -----------------------------
    def reset(self):
        # Escolhe ponto inicial com base no modo
        if RANDOM_START:
            if START_MODE == "volatility":
                idxs = np.arange(self.window_min, self.window_max)
                p = self._vol[self.window_min:self.window_max]
                p = p / (p.sum() + 1e-9)
                self.t = int(np.random.choice(idxs, p=p))
            else:
                self.t = np.random.randint(self.window_min, self.window_max)
        else:
            self.t = 0

        # Estado simbiótico inicial
        self.pos = 0
        self.eq = 1.0
        self.energia = ENERGIA_INICIAL
        self.pontuacao = 0
        self.done = False
        self.steps = 0

        # Loga apenas após o primeiro reset (para não poluir)
        if self.initialized:
            print(f"♻ Reset simbiótico em t={self.t} | energia={self.energia:.2f} | modo={START_MODE}")
        else:
            self.initialized = True
            print(f"🧠 Iniciando treino simbiótico — modo START='{START_MODE}' — t_inicial={self.t}")

        return self.obs(0)

    # -----------------------------
    # 🔍 Observação
    # -----------------------------
    def obs(self, a_prev):
        return np.concatenate([self.b[self.t], [self.pos, a_prev]]).astype(np.float32)

    # -----------------------------
    # ⚙️ Step simbiótico
    # -----------------------------
    def step_once(self, a):
        if self.done:
            return self.obs(0), 0.0, True, {}

        a = int(np.clip(int(a), -1, 1))
        prev_pos = self.pos
        self.pos = a
        t0 = self.t
        self.t += 1
        self.steps += 1

        # ciclo contínuo
        if CYCLE_AT_END and self.t >= self.n - H_FUTURO:
            self.t = np.random.randint(self.window_min, self.window_max)

        # calcula retorno
        futuro = min(self.t + H_FUTURO, self.n - 1)
        ret_futuro = (self.p[futuro] - self.p[t0]) / (self.p[t0] + 1e-9)
        previsao = a * TARGET_RET
        erro_abs = float(abs(ret_futuro - previsao))
        trade_cost = (COST * abs(self.pos - prev_pos) + SLIP) if self.pos != prev_pos else 0.0

        # recompensa simbiótica
        reward = (TARGET_RET - erro_abs) / (TARGET_RET + 1e-9) - trade_cost
        reward = float(np.clip(reward, -1.5, +1.5))
        self.eq *= (1.0 + reward * 0.05)

        # 🔋 energia viva
        self.energia -= ENERGIA_DECAIMENTO
        if erro_abs < TARGET_RET:
            self.energia += ENERGIA_BONUS * (1 - erro_abs / TARGET_RET)
            self.pontuacao += PONTUACAO_BONUS
        else:
            self.energia -= ENERGIA_PENALTY * min(erro_abs / TARGET_RET, 2.0)

        self.energia = float(np.clip(self.energia, 0.0, 2.0))

        # 🧬 condição simbiótica de término
        done_env = False
        if self.steps > MIN_STEPS and self.energia < ENERGIA_LIMITE:
            done_env = True

        # 🔖 salva best_score se bater recorde
        if done_env and self.pontuacao > self.best_score:
            self.best_score = float(self.pontuacao)
            self._save_best_score(self.best_score)
            print(f"🏆 Novo recorde simbiótico: {self.best_score:.1f} pontos!")

        info = {
            "ret": float(ret_futuro),
            "erro": erro_abs,
            "eq": float(self.eq),
            "energia": float(self.energia),
            "pontuacao": float(self.pontuacao),
            "melhor": float(self.best_score),
            "steps": int(self.steps)
        }

        self.done = done_env
        return self.obs(a), reward, self.done, info

    # -----------------------------
    # 🔁 Step com repetição
    # -----------------------------
    def step(self, a, repeats=1):
        total_r, s_next, done, info = 0.0, None, False, {}
        for _ in range(max(1, repeats)):
            s_next, r, done, info = self.step_once(a)
            total_r += r
            if done:
                break
        return s_next, float(total_r), done, info

    # -----------------------------
    # 💾 Persistência simbiótica
    # -----------------------------
    def _load_best_score(self):
        try:
            if not os.path.exists(BEST_SCORE_FILE):
                return 0.0
            data = json.load(open(BEST_SCORE_FILE))
            if isinstance(data, dict):
                return float(data.get("score", 0.0))
            return float(data)
        except Exception:
            return 0.0

    def _save_best_score(self, score):
        try:
            json.dump({"score": float(score)}, open(BEST_SCORE_FILE, "w"), indent=2)
        except Exception as e:
            print(f"[WARN] Falha ao salvar best_score: {e}")
