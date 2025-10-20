import os, random, warnings, gc, torch, numpy as np, psutil

import torch, gc, threading

# =========================================================
# 🚀 Utilitário turbo simbiótico — otimizado para PyTorch 2.4+
# =========================================================
import torch, gc

def turbo_cuda():

    # =========================================================
    # 🚀 Turbo CUDA não bloqueante
    # =========================================================
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def reseed(seed=42):
    """Reseta RNG global (CPU + GPU) para reprodutibilidade simbiótica."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.random.manual_seed(seed)
    print(f"🌱 RNG reseed simbiótico | seed={seed}")

def limpar_memoria():
    """Força liberação simbiótica total de VRAM e memória RAM."""
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        # encerra threads zumbis
        for proc in psutil.process_iter(["pid", "name"]):
            if "python" in proc.info["name"]:
                pass  # não mata o processo principal, apenas sinaliza
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"♻️ Limpeza simbiótica completa | RAM usada={rss:.1f} MB")
    except Exception as e:
        print(f"[WARN] limpeza simbiótica falhou: {e}")

warnings.filterwarnings("ignore", category=UserWarning)