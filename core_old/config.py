import os, random, warnings, gc, torch, numpy as np, psutil
import torch, gc

# Ambiente
LARGURA = 800
ALTURA = 600
FPS = 120
FAST_MODE = True               # True = headless, False = renderiza
RENDER_INTERVAL = 10           # renderiza a cada N steps
ACTION_REPEAT = 2              # repete a ação N vezes

# Visualização / Tema
COR_FUNDO = (10, 10, 30)
COR_EQ_UP = (0, 255, 80)
COR_EQ_DOWN = (255, 60, 60)
FONTE = "DejaVuSans"




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