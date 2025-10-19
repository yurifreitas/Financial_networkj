# =========================================================
# 🎯 EtherSym Finance — core/collector.py
# =========================================================
# - Coleta uma etapa de experiência simbiótica
# - Usa o estado vindo do main e retorna o próximo
# =========================================================

from core.utils import escolher_acao, a_to_idx

def collect_step(env, modelo, device, eps_now, replay, nbuf, total_reward_ep):
    """
    Executa uma etapa no ambiente simbiótico e registra as transições no replay buffer.
    Compatível com Env completo (sem .state).
    """
    # 📥 Obtém estado atual (mantido no loop principal)
    # → O main passa sempre `s`, vindo do reset() ou step()
    s_cur = getattr(env, "current_state", None)
    if s_cur is None:
        s_cur = env.reset()

    # 🎯 Escolhe ação simbiótica
    a, conf = escolher_acao(
        modelo, s_cur, device,
        eps_now,
        getattr(env, "capital", 0.0),
        getattr(env, "pos", 0.0)
    )

    # 🚀 Executa o passo no ambiente
    sp, r, done, info = env.step(a)
    total_reward_ep += r

    # 💾 Registra a transição N-step
    y_ret = float(info.get("ret", 0.0))
    nbuf.push(s_cur, a, r, y_ret)
    if len(nbuf.traj) == nbuf.n or done:
        item = nbuf.flush(sp, done)
        if item:
            s0, a0, Rn, sn, dn, y0 = item
            replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

    # 🔁 Atualiza o estado atual para o próximo ciclo
    env.current_state = sp

    return sp, done, info, total_reward_ep
