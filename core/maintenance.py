# =========================================================
# 🌿 EtherSym Finance — core/maintenance.py
# =========================================================
# - Poda simbiótica adaptativa
# - Regeneração e homeostase
# - Compatível com main_v8f_full e network.py
# - Também trata a homeostase do replay
# =========================================================

import torch
import numpy as np

# =========================================================
# 🌿 Poda simbiótica adaptativa
# =========================================================
def aplicar_poda(modelo, limiar_base=0.002):
    """
    Remove pesos muito pequenos (próximos de zero) de forma adaptativa.
    Mantém a rede enxuta, evitando saturação e explosão de gradientes.
    """
    total = sum(p.numel() for p in modelo.parameters())
    podadas = 0

    with torch.no_grad():
        for n, p in modelo.named_parameters():
            if "weight" in n:
                # limiar adaptativo com fator de escala dependente da média
                limiar = limiar_base * (1 + p.abs().mean().item() * 5)
                mask = p.abs() > limiar
                podadas += torch.numel(p) - mask.sum().item()
                p.mul_(mask)

    taxa = podadas / total if total > 0 else 0.0
    return taxa


# =========================================================
# 🧬 Regeneração simbiótica (neurogênese controlada)
# =========================================================
def regenerar_sinapses(modelo, taxa_poda, limiar_regen=0.15, taxa_regen=0.03):
    """
    Regenera sinapses perdidas quando a taxa de poda é alta.
    Pequenas porcentagens de pesos são reinicializadas para
    reintroduzir diversidade e plasticidade simbiótica.
    """
    if taxa_poda > limiar_regen:
        with torch.no_grad():
            for n, p in modelo.named_parameters():
                if "weight" in n:
                    var = torch.var(p)
                    mask = torch.rand_like(p) < taxa_regen
                    novos = torch.randn_like(p) * (var.sqrt() * 0.5)
                    p.add_(mask.float() * novos)
        print(f"🧬 Regeneração simbiótica ativada | taxa_poda={taxa_poda:.3f}")


# =========================================================
# ⚖️ Homeostase simbiótica (autoestabilização)
# =========================================================
def verificar_homeostase(modelo, media):
    """
    Mantém o equilíbrio dinâmico da rede simbiótica.
    Reinicia levemente os parâmetros normalizados quando a
    variação média de perda fica muito baixa (rede estabilizada).
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

        # Se a rede estabilizou demais, aplica leve perturbação simbiótica
        if modelo.estavel >= 15:
            _reset_norms(modelo)
            modelo.estavel = 0
            print("⚖️ Homeostase simbiótica restaurada (reset leve)")
    modelo.media_antiga = m


# =========================================================
# 🧠 Reset leve de normas e pesos
# =========================================================
def _reset_norms(modelo):
    """
    Perturba levemente os pesos e reinicializa LayerNorms
    para evitar que a rede entre em estado morto.
    """
    with torch.no_grad():
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
    Reajusta prioridades e pesos do replay buffer para
    evitar concentração de amostras antigas.
    """
    try:
        if hasattr(replay, "prioridades"):
            p = replay.prioridades
            if isinstance(p, np.ndarray):
                p = np.nan_to_num(p, nan=1.0)
                p /= np.max(p) + 1e-9
                replay.prioridades = np.clip(p, 1e-3, 1.0)
                print("♻️ Homeostase simbiótica aplicada ao replay buffer")
    except Exception as e:
        print(f"[WARN] homeostase_replay falhou: {e}")
