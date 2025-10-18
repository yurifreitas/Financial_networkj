# utils_v9.py
import random, numpy as np, torch

def turbo_cuda():
    """Ativa otimizações TF32 e benchmark CUDA."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_num_threads(1)
    except:
        pass

def reseed(seed=42):
    """Reseta todos os geradores de número aleatório."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🌱 RNG reseed com seed={seed}")

@torch.no_grad()
def soft_update(target, online, tau=0.005):
    """Atualização suave dos pesos alvo."""
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)
