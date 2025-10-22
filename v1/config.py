# =========================================================
# 🌌 EtherSym Finance — Configurações Principais
# =========================================================

# Ambiente
LARGURA = 800
ALTURA = 600
FPS = 120
FAST_MODE = True               # True = headless, False = renderiza
RENDER_INTERVAL = 10           # renderiza a cada N steps
ACTION_REPEAT = 2              # repete a ação N vezes

# Treinamento
GAMMA = 0.99                   # fator de desconto
BATCH = 512                   # tamanho do batch
MEMORIA_MAX = 16384            # capacidade máxima do replay
MIN_REPLAY = 10000             # mínimo antes de treinar
N_STEP = 3                     # N-step return
TARGET_TAU = 0.005             # soft update rate
TARGET_SYNC_HARD = 10_000      # sincronização dura
EPSILON_INICIAL = 0.5
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
LR = 1e-4                      # taxa de aprendizado

LR_MIN = 1e-6
LR_WARMUP_STEPS = 5_000

TEMP_INI, TEMP_MIN, TEMP_DECAY = 0.95, 0.60, 0.9995
BETA_PER_INI, BETA_PER_MAX, BETA_PER_DECAY = 0.6, 1.0, 0.9999
# Log / Salvamento
LOG_INTERVAL = 500             # passos entre logs
AUTOSAVE_EVERY = 120           # segundos entre autosaves
SAVE_PATH = "estado_treinamento_finance.pth"

# Homeostase simbiótica
PODA_BASE = 0.002
NEUROGENESE_LIMIAR = 0.15
HOMEOSTASE_TOLERANCIA = 0.015

# Visualização / Tema
COR_FUNDO = (10, 10, 30)
COR_EQ_UP = (0, 255, 80)
COR_EQ_DOWN = (255, 60, 60)
FONTE = "DejaVuSans"
# =========================================================
# ⚙️ Hiperparâmetros principais
# =========================================================
CSV = "binance_BTC_USDT_15m_2y.csv"
SEED = 64




# Perdas / Clamps
LAMBDA_REG_BASE = 0.05      # ↓ menos competição no início
Y_CLAMP = 0.05
Q_CLAMP = 50.0
Q_TARGET_CLAMP = 500.0

# Estabilidade
GRAD_CLIP = 0.3
LOSS_GUARD = 5e4
COOLDOWN_STEPS = 12000
REG_FREEZE_STEPS = 10_000
ROLLBACK_EVERY = 2_000
MAX_ROLLBACKS = 5

# Logs / manutenção
PRINT_EVERY = 400
SAVE_EVERY = 10_000
PODA_EVERY = 5_000
HOMEOSTASE_EVERY = 2_000
TARGET_TAU_BASE = 0.005
HARD_SYNC_EVERY = 50_000
CAPITAL_INICIAL = 1_000.0