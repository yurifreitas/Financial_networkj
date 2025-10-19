import os, random, warnings, torch, numpy as np

# =========================================================
# 🧩 Utilitários
# =========================================================
def turbo_cuda():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count() // 2)

def reseed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"🌱 RNG reseed com seed={seed}")

warnings.filterwarnings("ignore", category=UserWarning)
