# =========================================================
# 🌌 EtherSym Finance — main.py (DQN + Regressão de Preço)
# =========================================================
import os, time, random, torch
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np, pandas as pd

from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL, GAMMA, BATCH
from network import criar_modelo
from memory import salvar_estado, carregar_estado, RingReplay, NStepBuffer
from env import Env, make_feats, START_MODE, TARGET_RET

CSV = "binance_BTC_USDT_1h_2y.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)
LIMIAR = 0.003
EPSILON_DECAY, EPSILON_MIN = 0.9985, 0.05
TARGET_TAU, TARGET_SYNC_HARD = 0.005, 10_000
LOG_INTERVAL = 500
AUTOSAVE_EVERY = 120
N_STEP, MIN_REPLAY, ACTION_REPEAT = 3, 4096, 1

STATE_DIM = 10
LAMBDA_REG = 0.3               # peso da regressão de preço
Y_CLAMP = 0.02                  # clamp em ±2% para estabilizar

def a_to_idx(a: int) -> int:
    return int(a + 1)          # {-1,0,1} -> {0,1,2}

@torch.no_grad()
def escolher_acao(modelo, s, eps):
    if random.random() < eps:
        return int(np.random.choice(ACTION_SPACE))
    x = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q, _ = modelo(x)  # q: (1,3), y_pred ignorado aqui
    probs = torch.softmax(q / 0.8, dim=1).squeeze(0).cpu().numpy()
    return int(np.random.choice(ACTION_SPACE, p=probs))

def run():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"CSV não encontrado: {CSV}")
    df = pd.read_csv(CSV)
    base, price = make_feats(df)
    env = Env(base, price)

    modelo, alvo, opt = criar_modelo(DEVICE)
    replay = RingReplay(state_dim=STATE_DIM, capacity=MEMORIA_MAX, device=DEVICE)
    nbuf = NStepBuffer(N_STEP, GAMMA)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    _mem_legacy, EPSILON, media_antiga = carregar_estado(modelo, opt)
    EPSILON = EPSILON or EPSILON_INICIAL

    total_steps, episode = 0, 0
    autosave_t0 = time.time()
    print(f"🧠 Iniciando treino simbiótico híbrido (preço + ação) | START='{START_MODE}' | device={DEVICE.type}")

    while True:
        episode += 1
        s = env.reset()
        done, total_reward_ep = False, 0.0

        while not done:
            total_steps += 1

            # política
            a = escolher_acao(modelo, s, EPSILON)
            sp, r, done_env, info = env.step(a, repeats=ACTION_REPEAT)
            total_reward_ep += r

            # salva retorno futuro real deste passo (alvo contínuo)
            y_ret = float(info.get("ret", 0.0))

            # N-step com alvo contínuo
            nbuf.push(s, a, r, y_ret)
            flush = (len(nbuf.traj) == N_STEP) or done_env
            if flush:
                item = nbuf.flush(sp, done_env)
                if item is not None:
                    s0, a0, Rn, sn, dn, y0 = item
                    replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

            s = sp
            done = done_env

            # === aprendizagem simbiótica ===
            if len(replay) >= MIN_REPLAY:
                (estados_t, acoes_t, recompensas_t, novos_estados_t,
                 finais_t, idx, w, y_ret_t) = replay.sample(BATCH)

                with torch.no_grad():
                    next_q_online, _ = modelo(novos_estados_t)
                    next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                    next_q_target, _ = alvo(novos_estados_t)
                    next_best = next_q_target.gather(1, next_actions).squeeze(1)
                    alvo_q = recompensas_t + (GAMMA ** N_STEP) * next_best * (1.0 - finais_t)

                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    q_vals, y_pred = modelo(estados_t)            # y_pred ∈ ℝ (retorno previsto)
                    q_sel = q_vals.gather(1, acoes_t).squeeze(1)

                    # RL (PER + IS)
                    per_sample = F.smooth_l1_loss(q_sel, alvo_q, reduction="none", beta=0.8)
                    loss_q = (w * per_sample).mean()

                    # Regressão de preço/retorno — clamp do alvo para estabilidade
                    y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP)
                    loss_reg = F.smooth_l1_loss(y_pred, y_target, beta=0.5)

                    loss_total = loss_q + LAMBDA_REG * loss_reg

                scaler.scale(loss_total).backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                with torch.no_grad():
                    td_error = (alvo_q - q_sel).abs().detach().cpu().numpy()
                    replay.update_priority(idx, td_error)

                    # soft update alvo
                    for p_t, p in zip(alvo.parameters(), modelo.parameters()):
                        p_t.data.mul_(1.0 - TARGET_TAU).add_(TARGET_TAU * p.data)

                    # hard sync periódico (opcional)
                    if total_steps % TARGET_SYNC_HARD == 0:
                        alvo.load_state_dict(modelo.state_dict())

            # decaimento do ε
            EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

            # autosave de segurança
            if (time.time() - autosave_t0) > AUTOSAVE_EVERY:
                autosave_t0 = time.time()
                salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)

            if done:
                salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
                print(f"🏁 Episódio {episode:4d} | Reward={total_reward_ep:+.3f} | ε={EPSILON:.3f}")
                break

if __name__ == "__main__":
    run()
