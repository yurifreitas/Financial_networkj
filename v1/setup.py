try:
    import torch._logging as _logging
    _logging.set_logs()  # Limpa logs simbióticos
except Exception:
    pass

try:
    import torch._dynamo.config as dynamo_cfg
    dynamo_cfg.verbose = False
    dynamo_cfg.suppress_errors = True
    dynamo_cfg.log_level = "ERROR"
except Exception:
    pass

try:
    import torch._inductor.config as inductor_cfg
    inductor_cfg.debug = False
    inductor_cfg.triton.cudagraphs = False
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_pointwise = True
    inductor_cfg.max_autotune_gemm_backends = "cublas,triton,aten"  # ✅ inclui fallback ATEN
except Exception as e:
    print(f"⚠️ Patch inductor parcial: {e}")

import os, time, math, random, warnings
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# =========================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.update({
    "TORCH_COMPILE_DEBUG": "0",
    "TORCHINDUCTOR_DISABLE_FX_VALIDATION": "1",
    "TORCHINDUCTOR_FUSE_TRIVIAL_OPS": "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "cublas,triton",
    "TORCHINDUCTOR_CACHE_DIR": os.path.expanduser("~/.cache/torch/inductor"),
})

# =========================================================
# 🧩 Utilitários
# =========================================================
def turbo_cuda():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def reseed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"🌱 RNG reseed com seed={seed}")

@torch.no_grad()
def soft_update(target, online, tau=0.005):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)

def loss_q_hibrida(q_pred, q_tgt):
    mse = F.mse_loss(q_pred, q_tgt)
    hub = F.smooth_l1_loss(q_pred, q_tgt, beta=0.8)
    return 0.7 * mse + 0.3 * hub

def loss_regressao(y_pred, y_tgt, l2=1e-4):
    reg = F.smooth_l1_loss(y_pred, y_tgt, beta=0.5)
    return reg + l2 * (y_pred**2).mean()

def escolher_acao(modelo, estado_np, device, eps,
                  temp_base=0.95, temp_min=0.60,
                  capital=0.0, posicao=0.0, gate_conf=0.35):
    ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
    if np.random.rand() < eps:
        return int(np.random.choice(ACTIONS)), 0.5
    x = torch.tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals, y = modelo(x)
        # Confiança suave (sem saturar em 1.0 tão cedo)
        conf = float(torch.sigmoid(torch.abs(y)).item())
        # Temperatura vinculada ao epsilon
        temp = max(temp_min, temp_base * (0.8 + 0.2 * (eps / max(1e-6, eps))))
        probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).clamp_(1e-6, 1.0)
        probs = (probs / probs.sum()).cpu().numpy()
    a = int(np.random.choice(ACTIONS, p=probs))
    if conf < gate_conf: a = 0
    if a == 1 and capital <= 0: a = 0
    if a == -1 and posicao <= 0: a = 0
    return a, conf

def set_lr(optim, lr):
    for g in optim.param_groups:
        g.update(lr=lr)

def is_bad_number(x):
    return torch.isnan(x).any() or torch.isinf(x).any()




ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)
def a_to_idx(a: int): return int(a + 1)
from v1.config import *
# Estado anti-explosão
cooldown_until = 0
rollbacks = 0
last_good = None   # Snapshot periódico para rollback
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda")
turbo_cuda(); reseed(SEED)
if not os.path.exists(CSV):
    raise FileNotFoundError(f"CSV não encontrado: {CSV}")

df = pd.read_csv(CSV)