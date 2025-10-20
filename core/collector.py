# =========================================================
# 🎯 EtherSym Finance — core/collector.py (versão otimizada)
# =========================================================
# - Coleta uma etapa de experiência simbiótica
# - Move tensores para GPU de forma estável
# - Minimiza recompilações do modelo entre episódios
# =========================================================

import torch
from core.utils import escolher_acao, a_to_idx

def collect_step(env, modelo, device, eps_now, replay, nbuf, total_reward_ep):
    """
    Executa uma etapa no ambiente simbiótico e registra transições no replay buffer.
    Compatível com Env completo (sem .state persistente).
    """
    # 📥 Estado atual
    s_cur = getattr(env, "current_state", None)
    if s_cur is None:
        s_cur = env.reset()

    # 🔁 Converte para tensor GPU fixo (shape constante)
    if not torch.is_tensor(s_cur):
        s_cur_t = torch.tensor(s_cur, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        s_cur_t = s_cur.to(device, non_blocking=True).unsqueeze(0)

    # 🎯 Escolha simbiótica da ação (no device)
    a, conf = escolher_acao(
        modelo, s_cur_t, device,
        eps_now,
        getattr(env, "capital", 0.0),
        getattr(env, "pos", 0.0)
    )

    # 🚀 Passo no ambiente
    sp, r, done, info = env.step(a)

    # Atualiza o total de recompensa para o episódio
    total_reward_ep += r  # Acumula a recompensa no episódio

    # 💾 Transição N-Step
    y_ret = float(info.get("ret", 0.0))
    nbuf.push(s_cur, a, r, y_ret)

    if len(nbuf.traj) == nbuf.n or done:
        item = nbuf.flush(sp, done)
        if item:
            s0, a0, Rn, sn, dn, y0 = item
            replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

    # 🔁 Atualiza o estado
    env.current_state = sp

    # Retorna o próximo estado, se o episódio terminou, informações e o total de recompensa acumulado
    return sp, done, info, total_reward_ep
