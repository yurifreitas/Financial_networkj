# =========================================================
# 🌌 EtherSym Finance — env.py (v7 Falência Realista)
# =========================================================
import numpy as np, pandas as pd, json, random, os

# =========================================================
# ⚙️ Parâmetros simbióticos e financeiros
# =========================================================
COST = 0.0005
SLIP = 0.0002
TARGET_RET = 0.003
H_FUTURO = 3
CAPITAL_INICIAL = 1_000.0
ALOCACAO = 0.5
CUSTO_TRADE = COST + SLIP

# Energia simbiótica (agora apenas métrica interna, não reset)
ENERGIA_INICIAL = 1.0
ENERGIA_LIMITE = 0.25        # mantido apenas para log
ENERGIA_DECAIMENTO = 0.0008
ENERGIA_BONUS = 0.012
ENERGIA_PENALTY = 0.025
PONTUACAO_BONUS = 1.0

# Controle de episódios
RANDOM_START = True
START_MODE = "volatility"
CYCLE_AT_END = True
MIN_STEPS = 400
BEST_SCORE_FILE = "best_score.json"


# =========================================================
# 🧩 Geração de features
# =========================================================
def make_feats(df):
    df.columns = [c.lower() for c in df.columns]
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["vol_ret"] = df["ret"].rolling(24).std().fillna(0.0)
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    rsi = 100 - (100 / (1 + up/(down+1e-9)))
    df["rsi_n"] = (rsi - 50)/50
    df["range"] = (df["high"]-df["low"])/(df["close"]+1e-9)
    df["body"] = (df["close"]-df["open"])/((df["high"]-df["low"])+1e-9)
    df["volume_z"] = ((df["volume"]-df["volume"].rolling(48).mean())/
                      (df["volume"].rolling(48).std()+1e-9)).fillna(0.0)
    roll = df["close"].rolling(48)
    df["z"] = ((df["close"]-roll.mean())/(roll.std()+1e-9)).fillna(0.0)
    df = df.dropna().reset_index(drop=True)
    base = df[["ret","vol_ret","range","body","ema_diff","rsi_n","z","volume_z"]].astype(np.float32).values
    price = df["close"].astype(np.float32).values
    return base, price


# =========================================================
# 🎮 Ambiente simbiótico-realista (v7)
# =========================================================
class Env:
    def __init__(self, base, price):
        self.b, self.p = base, price
        self.n = len(price)
        self.window_min = 48
        self.window_max = self.n - 800
        vol = np.abs(np.diff(self.p))
        self._vol = vol / (vol.sum()+1e-9)
        self.best_score = self._load_best_score()
        self.initialized = False
        self.reset()

    # -----------------------------
    # 🔁 Reset simbiótico universal
    # -----------------------------
    def reset(self):
        if RANDOM_START:
            if START_MODE == "volatility":
                idxs = np.arange(self.window_min, self.window_max)
                p = self._vol[self.window_min:self.window_max]
                self.t = int(np.random.choice(idxs, p=p/p.sum()))
            else:
                self.t = np.random.randint(self.window_min, self.window_max)
        else:
            self.t = 0

        self.pos = 0.0
        self.energia = ENERGIA_INICIAL
        self.pontuacao = 0.0
        self.done = False
        self.steps = 0
        self.capital = CAPITAL_INICIAL
        self.preco_entrada = None
        self.max_patrimonio = CAPITAL_INICIAL

        print(f"♻ Reset simbiótico em t={self.t} | energia={self.energia:.2f}")
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
        preco = self.p[self.t]
        self.t += 1
        self.steps += 1

        if CYCLE_AT_END and self.t >= self.n - H_FUTURO:
            self.t = np.random.randint(self.window_min, self.window_max)

        # --- operações reais ---
        prev_pos = self.pos
        if a == 1 and self.capital > 0 and self.pos <= 1e-12:
            qtd = (self.capital*ALOCACAO)/(preco+1e-12)
            custo = qtd*preco*(1+CUSTO_TRADE)
            if custo <= self.capital:
                self.capital -= custo
                self.pos = qtd
                self.preco_entrada = preco
        elif a == -1 and self.pos > 1e-12:
            receita = self.pos*preco*(1-CUSTO_TRADE)
            self.capital += receita
            self.pos = 0.0
            self.preco_entrada = None

        # --- patrimônio ---
        patrimonio = self.capital + self.pos*preco
        self.max_patrimonio = max(self.max_patrimonio, patrimonio)

        # --- retorno futuro ---
        futuro = min(self.t+H_FUTURO, self.n-1)
        ret_futuro = (self.p[futuro]-preco)/(preco+1e-9)
        previsao = a*TARGET_RET
        erro_abs = abs(ret_futuro-previsao)
        delta_pos = abs(self.pos - prev_pos)
        trade_cost = CUSTO_TRADE*(delta_pos > 1e-12)
        reward = (TARGET_RET-erro_abs)/(TARGET_RET+1e-9)-float(trade_cost)
        reward = float(np.clip(reward, -1.5, 1.5))

        # --- energia simbiótica (só métrica, não reset) ---
        self.energia -= ENERGIA_DECAIMENTO
        if erro_abs < TARGET_RET:
            self.energia += ENERGIA_BONUS*(1-erro_abs/TARGET_RET)
            self.pontuacao += PONTUACAO_BONUS
        else:
            self.energia -= ENERGIA_PENALTY*min(erro_abs/TARGET_RET,2.0)
        self.energia = float(np.clip(self.energia,0.0,2.0))

        # --- término apenas por falência real ---
        done_env = False
        if patrimonio <= 1.0:
            print(f"💀 Falência simbiótica detectada | capital={self.capital:.2f}")
            done_env = True

        if done_env and self.pontuacao > self.best_score:
            self.best_score = self.pontuacao
            self._save_best_score(self.best_score)
            print(f"🏆 Novo recorde simbiótico: {self.best_score:.1f} pontos!")

        info = {
            "ret": float(ret_futuro),
            "erro": float(erro_abs),
            "energia": float(self.energia),
            "pontuacao": float(self.pontuacao),
            "melhor": float(self.best_score),
            "capital": float(self.capital),
            "patrimonio": float(patrimonio),
            "max_patrimonio": float(self.max_patrimonio),
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
        return s_next, total_r, done, info

    # -----------------------------
    # 💾 Persistência simbiótica
    # -----------------------------
    def _load_best_score(self):
        try:
            if not os.path.exists(BEST_SCORE_FILE):
                return 0.0
            data = json.load(open(BEST_SCORE_FILE))
            return float(data.get("score", 0.0)) if isinstance(data, dict) else float(data)
        except Exception:
            return 0.0

    def _save_best_score(self, score):
        try:
            json.dump({"score": float(score)}, open(BEST_SCORE_FILE, "w"), indent=2)
        except Exception as e:
            print(f"[WARN] Falha ao salvar best_score: {e}")
