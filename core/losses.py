import torch
import torch.nn.functional as F

# =========================================================
# 🎯 Loss Q-Híbrida Reativa (mantém compatibilidade)
# =========================================================
def loss_q_hibrida(q_pred, q_tgt):
    """
    Mesmo retorno, mas com:
    - leve ruído simbiótico anti-saturação
    - normalização adaptativa para escapar de loss=0
    """
    # erro e fator de energia adaptativo
    err = (q_pred - q_tgt)
    energy = (err**2).mean().sqrt().detach() + 1e-6

    # perda base
    mse = F.mse_loss(q_pred, q_tgt, reduction="mean")
    hub = F.smooth_l1_loss(q_pred, q_tgt, beta=0.8, reduction="mean")

    # excitação simbiótica — injeta ruído microcontrolado no gradiente
    jitter = 1e-3 * torch.randn_like(q_pred)
    reg = 1e-6 * (q_pred ** 2).mean()

    # mistura equilibrada com fator dinâmico
    mix = 0.7 * mse + 0.3 * hub
    loss = mix / (energy + 1e-6) + reg + (jitter * err).mean() * 0.01
    return torch.nan_to_num(loss.clamp_min(1e-9))


# =========================================================
# 📈 Loss Regressão Reativa (mantém compatibilidade)
# =========================================================
def loss_regressao(y_pred, y_tgt, l2=1e-4):
    """
    Reforça variação e evita congelamento:
    - Mantém retorno escalar
    - Suaviza, mas injeta energia mínima
    """
    reg = F.smooth_l1_loss(y_pred, y_tgt, beta=0.5, reduction="mean")
    reg_l2 = l2 * (y_pred ** 2).mean()

    # pequena energia simbiótica evita perda nula
    energ = (y_pred - y_tgt).pow(2).mean().sqrt() * 1e-3
    loss = reg + reg_l2 + energ
    return torch.nan_to_num(loss.clamp_min(1e-9))
