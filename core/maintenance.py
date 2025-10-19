# =========================================================
# 🌿 EtherSym Finance — core/maintenance.py (GPU-safe)
# =========================================================
# - Poda, regeneração e homeostase simbiótica otimizadas
# - Totalmente assíncronas, sem .item() nem .cpu() dentro do loop
# =========================================================

import torch, numpy as np

# =========================================================
# 🌿 Poda simbiótica GPU-safe adaptativa
# =========================================================
@torch.no_grad()
def aplicar_poda(modelo, limiar_base=0.002):
    """
    Remove pesos muito pequenos (|w| < limiar adaptativo) de forma
    totalmente GPU-safe (sem .item dentro do loop).
    Retorna taxa de poda global.
    """
    total_params = torch.zeros(1, device="cuda")
    total_podados = torch.zeros(1, device="cuda")

    for n, p in modelo.named_parameters():
        if "weight" not in n or not p.requires_grad:
            continue

        # média e limiar calculados inteiramente em GPU
        limiar = limiar_base * (1 + p.abs().mean() * 5)
        mask = (p.abs() > limiar)
        total_params += p.numel()
        total_podados += (p.numel() - mask.sum())
        p.mul_(mask)  # zera pesos pequenos in-place

    taxa = (total_podados / total_params).detach().to("cpu", non_blocking=True).item()
    return taxa


# =========================================================
# 🧬 Regeneração simbiótica GPU-safe
# =========================================================
@torch.no_grad()
def regenerar_sinapses(modelo, taxa_poda, limiar_regen=0.15, taxa_regen=0.03):
    """
    Reinicializa uma pequena fração de pesos quando há poda alta.
    Operações todas na GPU; nada de sincronização CPU.
    """
    if taxa_poda <= limiar_regen:
        return

    for n, p in modelo.named_parameters():
        if "weight" not in n or not p.requires_grad:
            continue

        var = torch.var(p)
        mask = torch.rand_like(p) < taxa_regen
        novos = torch.randn_like(p) * (var.sqrt() * 0.5)
        p.add_(mask * novos)

    print(f"🧬 Regeneração simbiótica ativada | taxa_poda={taxa_poda:.3f}")


# =========================================================
# ⚖️ Homeostase simbiótica (autoestabilização leve)
# =========================================================
def verificar_homeostase(modelo, media):
    """
    Mantém estabilidade simbiótica com histórico médio de perda.
    Atua sem interferir no fluxo GPU principal.
    """
    if media is None:
        return

    if not hasattr(modelo, "historico"):
        modelo.historico = []
        modelo.media_antiga = None
        modelo.estavel = 0
        modelo.limiar_homeostase = 0.015

    modelo.historico.append(media)
    if len(modelo.historico) < 20:
        return

    m = np.mean(modelo.historico[-10:])
    if modelo.media_antiga is not None:
        delta = abs(m - modelo.media_antiga)
        modelo.estavel = modelo.estavel + 1 if delta < modelo.limiar_homeostase else 0

        if modelo.estavel >= 15:
            _reset_norms(modelo)
            modelo.estavel = 0
            print("⚖️ Homeostase simbiótica restaurada (reset leve)")
    modelo.media_antiga = m


# =========================================================
# 🧠 Reset leve de normas e pesos (GPU-safe)
# =========================================================
@torch.no_grad()
def _reset_norms(modelo):
    for n, p in modelo.named_parameters():
        if "weight" in n:
            p.add_(torch.randn_like(p) * 0.02)
    for m in modelo.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.reset_parameters()


# =========================================================
# ♻️ Homeostase simbiótica do replay buffer
# =========================================================
def homeostase_replay(replay):
    """
    Normaliza prioridades do replay sem travar o treino.
    """
    try:
        if hasattr(replay, "p"):
            p = replay.p
            if isinstance(p, np.ndarray):
                p = np.nan_to_num(p, nan=1.0)
                p /= np.max(p) + 1e-9
                replay.p = np.clip(p, 1e-3, 1.0)
                print("♻️ Homeostase simbiótica aplicada ao replay buffer")
    except Exception as e:
        print(f"[WARN] homeostase_replay falhou: {e}")
