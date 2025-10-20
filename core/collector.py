import torch
from core.utils import escolher_acao, a_to_idx
from multiprocessing import Lock

# Bloqueio para garantir sincronização do replay
lock = Lock()

def collect_step(env, modelo, device, eps_now, replay, nbuf, total_reward_ep):
    # Verifica o estado atual do ambiente
    s_cur = getattr(env, "current_state", None)
    if s_cur is None:
        s_cur = env.reset()

    # Converte o estado atual para tensor e move para o dispositivo
    if not torch.is_tensor(s_cur):
        s_cur_t = torch.tensor(s_cur, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        s_cur_t = s_cur.to(device, non_blocking=True).unsqueeze(0)

    # Escolhe a ação com base no modelo e no estado atual
    a, conf = escolher_acao(
        modelo, s_cur_t, device,
        eps_now,
        getattr(env, "capital", 0.0),
        getattr(env, "pos", 0.0)
    )

    # Passa a ação para o ambiente
    sp, r, done, info = env.step(a)

    # Atualiza o total de recompensa
    total_reward_ep += r

    # Obtém o retorno futuro (se fornecido)
    y_ret = float(info.get("ret", 0.0))

    # Adiciona a transição ao buffer N-step
    nbuf.push(s_cur, a, r, y_ret)

    # Se o buffer atingir o tamanho ou o episódio terminar, salva as transições
    if len(nbuf.traj) == nbuf.n or done:
        item = nbuf.flush(sp, done)
        if item:
            s0, a0, Rn, sn, dn, y0 = item
            with lock:  # Sincroniza o acesso ao replay
                replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

    # Atualiza o estado atual no ambiente
    env.current_state = sp

    # Retorna o novo estado, se o episódio terminou, e a recompensa total acumulada
    return sp, done, info, total_reward_ep
