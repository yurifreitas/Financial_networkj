# =========================================================
# 🧩 EtherSym Finance — core/utils.py (corrigido e robusto)
# =========================================================

import numpy as np
import torch
import torch.nn.functional as F

# =========================================================
# 🧠 Atualização simbiótica dos parâmetros-alvo
# =========================================================
@torch.no_grad()
def soft_update(target, online, tau: float = 0.005):
    """Atualiza os pesos do modelo-alvo com suavização (Polyak averaging)."""
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)

# =========================================================
# 🎯 Escolha de ação simbiótica com controle de temperatura
# =========================================================
def escolher_acao(
    modelo,
    estado_np,
    device,
    eps,
    temp_base: float = 0.95,
    temp_min: float = 0.60,
    capital: float = 0.0,
    posicao: float = 0.0,
    gate_conf: float = 0.35,
):
    """
    Escolhe uma ação (-1, 0, 1) com base no modelo simbiótico.
    Inclui suavização de temperatura, filtro de confiança e proteção contra NaNs.
    """
    ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

    # 🎲 Epsilon-greedy aleatória
    if np.random.rand() < eps:
        return int(np.random.choice(ACTIONS)), 0.5

    # 🔢 Converte estado e faz inferência no modelo
    x = torch.tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = modelo(x)
        # Compatível com modelos que retornam (q_vals, y)
        if isinstance(out, tuple):
            q_vals, y = out
        else:
            q_vals, y = out, torch.zeros(1, device=device)

        # 🧩 Sanitiza Q-values
        q_vals = torch.nan_to_num(q_vals, nan=0.0, posinf=1e6, neginf=-1e6)

        # Confiança suave (evita saturação)
        conf = float(torch.sigmoid(torch.abs(y)).item())

        # Temperatura adaptativa vinculada ao epsilon
        temp = max(temp_min, temp_base * (0.8 + 0.2 * (eps / max(1e-6, eps))))
        probs = torch.softmax(q_vals / temp, dim=1).squeeze(0)
        probs = probs.clamp_(1e-6, 1.0)
        probs = (probs / probs.sum()).cpu().numpy().flatten()

    # ⚙️ Escolha simbiótica final
    a = int(np.random.choice(ACTIONS, p=probs))

    # 🛡️ Filtros simbióticos de coerência
    if conf < gate_conf:
        a = 0
    if a == 1 and capital <= 0:
        a = 0
    if a == -1 and posicao <= 0:
        a = 0

    return a, conf

# =========================================================
# ⚙️ Utilitários auxiliares
# =========================================================
def set_lr(optim, lr):
    """Atualiza taxa de aprendizado de forma segura."""
    for g in optim.param_groups:
        g["lr"] = lr

def is_bad_number(x):
    """Detecta NaN ou Inf em tensores PyTorch."""
    return torch.isnan(x).any() or torch.isinf(x).any()

def a_to_idx(a):
    """
    Converte ação simbiótica (-1, 0, 1) em índice (0, 1, 2).
    Compatível com escalares e arrays.
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        return (np.asarray(a, dtype=np.int64) + 1).astype(np.int64)
    else:
        return int(a + 1)
