# =========================================================
# 🧮 EtherSym Finance — core/losses_v8j.py (Resiliente)
# =========================================================
# - Anti-saturação simbiótica e energia mínima garantida
# - Normalização logarítmica e suavização dinâmica
# - Mantém compatibilidade total com AMP e GradScaler
# =========================================================

import torch
import torch.nn.functional as F

# =========================================================
# 🎯 Loss Q-Híbrida Resiliente
# =========================================================
def loss_q_hibrida(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    """
    Combina MSE + SmoothL1 + normalização logarítmica.
    Evita saturação quando q_tgt é muito maior que q_pred.
    """
    # Erro simbiótico
    err = q_pred - q_tgt
    abs_err = err.abs() + 1e-6

    # Normalização logarítmica → suaviza gradientes grandes
    norm = torch.log1p(abs_err)
    mse = F.mse_loss(q_pred, q_tgt, reduction="mean")
    huber = F.smooth_l1_loss(q_pred, q_tgt, beta=0.5, reduction="mean")

    # Mistura adaptativa
    base = 0.6 * mse + 0.4 * huber
    reg = 1e-6 * (q_pred ** 2).mean()
    jitter = 1e-4 * torch.randn_like(q_pred).mean()

    # Normalização simbiótica e energia mínima
    energy = norm.mean().detach().clamp_min(1e-3)
    loss = (base / energy) + reg + jitter

    # Saturação simbiótica (evita perda explosiva)
    loss = torch.tanh(loss / 5.0) * 5.0
    return torch.nan_to_num(loss.clamp_min(1e-9))


# =========================================================
# 📈 Loss Regressão Simbiótica Resiliente
# =========================================================
def loss_regressao(y_pred: torch.Tensor, y_tgt: torch.Tensor, l2: float = 1e-4) -> torch.Tensor:
    """
    Suaviza e equaliza gradiente para previsões contínuas.
    Evita congelamento e reduz a amplitude de erro.
    """
    diff = y_pred - y_tgt
    abs_diff = diff.abs() + 1e-6

    # Log-scale normalização para estabilidade
    norm = torch.log1p(abs_diff)
    smooth = F.smooth_l1_loss(y_pred, y_tgt, beta=0.5, reduction="mean")
    reg_l2 = l2 * (y_pred ** 2).mean()

    # Energia mínima simbiótica
    energ = norm.mean().detach().clamp_min(1e-3)
    loss = (smooth / energ) + reg_l2

    # Saturação simbiótica
    loss = torch.tanh(loss / 3.0) * 3.0
    return torch.nan_to_num(loss.clamp_min(1e-9))
