import torch, numpy as np

# =========================================================
# ⚙️ Hiperparâmetros principais
# =========================================================
CSV = "binance_BTC_USDT_1h_2y.csv"
SEED = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda")


# Treino
BATCH = 512
GAMMA = 0.995
LR = 3e-5
LR_MIN = 1e-6
LR_WARMUP_STEPS = 2_000

N_STEP = 4

# Replay / Epsilon / Temperatura
MEMORIA_MAX = 250_000
MIN_REPLAY = 3_000
EPSILON_INICIAL, EPSILON_DECAY, EPSILON_MIN = 1.0, 0.9992, 0.05
TEMP_INI, TEMP_MIN, TEMP_DECAY = 0.95, 0.60, 0.9995
BETA_PER_INI, BETA_PER_MAX, BETA_PER_DECAY = 0.6, 1.0, 0.9999

# Perdas / Clamps
LAMBDA_REG_BASE = 0.05      # ↓ menos competição no início
Y_CLAMP = 0.05
Q_CLAMP = 50.0
Q_TARGET_CLAMP = 500.0

# Estabilidade
GRAD_CLIP = 0.3
LOSS_GUARD = 5e4
COOLDOWN_STEPS = 1200
REG_FREEZE_STEPS = 10_000
ROLLBACK_EVERY = 2_000
MAX_ROLLBACKS = 5

# Logs / manutenção
PRINT_EVERY = 800
SAVE_EVERY = 10_000
PODA_EVERY = 5_000
HOMEOSTASE_EVERY = 4_000
TARGET_TAU_BASE = 0.005
HARD_SYNC_EVERY = 50_000
CAPITAL_INICIAL = 1_000.0

ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)


total_steps, episodio = 0, 0
last_loss, last_y_pred = 0.0, 0.0
temp_now, beta_per = TEMP_INI, BETA_PER_INI
ema_q, ema_r = None, None
cooldown_until = 0
rollbacks = 0
last_good = None
