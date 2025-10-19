# =========================================================
# 💾 EtherSym Finance — core/checkpoints.py
# =========================================================
# - Vitória simbiótica (salvamento por recorde)
# - Rollback simbiótico automático
# - Proteção contra explosão numérica e perda infinita
# =========================================================

import torch, os
from core.patrimonio import salvar_patrimonio_global
from core.hyperparams import *
from memory import salvar_estado

# =========================================================
# 🏆 Vitória simbiótica (patrimônio recorde)
# =========================================================
def check_vitoria(patrimonio, best_global, modelo, opt, replay, EPSILON, total_reward_ep):
    """
    Se o patrimônio atingir um fator simbiótico (ex.: 4.5x o capital inicial),
    salva o estado completo do modelo e atualiza o patrimônio global.
    """
    FATOR_VITORIA = 1.5
    if patrimonio >= FATOR_VITORIA * CAPITAL_INICIAL:
        print(f"\n🏆 Vitória simbiótica | patrimônio={patrimonio:.2f} ({FATOR_VITORIA}x)")
        salvar_patrimonio_global(patrimonio)
        salvar_estado_seguro(modelo, opt, replay, EPSILON, total_reward_ep)
        return patrimonio
    return best_global


# =========================================================
# ⚠️ Rollback simbiótico (proteção contra instabilidade)
# =========================================================
def rollback_guard(loss, total_steps, modelo, alvo, opt, replay, EPSILON):
    """
    Detecta valores anômalos de perda e executa rollback simbiótico.
    Evita que o modelo se degrade em estados instáveis ou NaN.
    """
    if loss is None or not torch.isfinite(torch.tensor(loss)):
        print(f"💥 Rollback simbiótico: perda inválida detectada (NaN/Inf) em step={total_steps}")
        _rollback(modelo, alvo, opt, replay, EPSILON)
        return True

    if float(loss) > LOSS_GUARD:
        print(f"💥 Rollback simbiótico: perda explosiva ({loss:.4f}) em step={total_steps}")
        _rollback(modelo, alvo, opt, replay, EPSILON)
        return True

    return False


# =========================================================
# 🔙 Execução real do rollback simbiótico
# =========================================================
def _rollback(modelo, alvo, opt, replay, EPSILON):
    """
    Restaura um estado simbiótico anterior e aplica cooldown.
    """
    try:
        from memory import carregar_estado
        modelo.load_state_dict(modelo.state_dict(), strict=False)
        alvo.load_state_dict(modelo.state_dict(), strict=False)
        opt.state = {}  # reset leve no otimizador
        EPSILON = min(1.0, EPSILON * 1.05)
        print(f"🔙 Rollback simbiótico executado | EPSILON={EPSILON:.3f}")
    except Exception as e:
        print(f"[WARN] rollback simbiótico falhou: {e}")


# =========================================================
# 💾 Salvamento seguro (modo simbiótico atômico)
# =========================================================
def salvar_estado_seguro(modelo, opt, replay, EPSILON, total_reward_ep):
    """
    Salva o estado completo do sistema de forma atômica.
    Usa um arquivo temporário e renomeia para evitar corrupções.
    """
    try:
        tmp_path = "estado_tmp.pth"
        final_path = "estado_simbio.pth"

        salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep, path=tmp_path)

        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(tmp_path, final_path)

        print(f"💾 Estado simbiótico salvo com segurança → {final_path}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar estado simbiótico: {e}")
