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
BATCH = 128                    # tamanho do batch
MEMORIA_MAX = 16384            # capacidade máxima do replay
MIN_REPLAY = 4096              # mínimo antes de treinar
N_STEP = 3                     # N-step return
TARGET_TAU = 0.005             # soft update rate
TARGET_SYNC_HARD = 10_000      # sincronização dura
EPSILON_INICIAL = 0.5
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
LR = 1e-4                      # taxa de aprendizado

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
