# =========================================================
# ⚙️ EtherSym Finance — config_env.py (versão aprimorada)
# =========================================================
# - Ajustes simbióticos mais estáveis
# - Equilíbrio entre custo, energia e capital
# - Base para stop-loss e take-profit dinâmicos
# =========================================================

# ==== 🧮 Trade e custos ====
COST = 0.0004             # taxa de corretagem (~0.04%)
SLIP = 0.0003             # slippage médio (~0.03%)
CUSTO_TRADE = COST + SLIP # custo total efetivo por operação
ALOCACAO = 0.9            # porcentagem do capital usada por trade
CAPITAL_INICIAL = 1_000.0 # capital inicial simbiótico

# ==== 🎯 Metas de retorno e horizonte ====
TARGET_RET = 0.003        # alvo de retorno base (0.4%)
H_FUTURO = 3              # horizonte preditivo curto (3 candles)
MIN_STEPS = 400           # mínimo de steps por episódio

# ==== 🔋 Dinâmica de energia simbiótica ====
ENERGIA_INICIAL = 1.0
ENERGIA_LIMITE = 0.25
ENERGIA_DECAIMENTO = 0.0007   # perda passiva por tempo
ENERGIA_BONUS = 0.015         # ganho por acerto bom
ENERGIA_PENALTY = 0.030       # penalidade por erro forte
PONTUACAO_BONUS = 1.0         # pontos por acerto (reforço simbólico)

# ==== 🔀 Inicialização e aleatoriedade ====
RANDOM_START = True
START_MODE = "volatility"     # escolhe início por volatilidade
CYCLE_AT_END = True           # recicla dataset no final (loop contínuo)

# ==== 🧠 Arquivo de persistência ====
BEST_SCORE_FILE = "best_score.json"

# ==== 📈 Stops e limites globais (valores padrão, usados em core_env) ====
STOP_LOSS_PCT = 0.08          # perda máxima 8%
TAKE_PROFIT_PCT = 0.12        # ganho alvo 12%
HOLD_MIN = 3                  # número mínimo de candles antes de stop/take

# ==== ⚡ Ajustes simbióticos extras (opcional) ====
ENERGIA_RECOMPENSA_SCALING = 0.5  # influência da energia nas recompensas
ENERGIA_REGEN_LIMIT = 1.8         # limite máximo de regeneração
