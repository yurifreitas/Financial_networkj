# =========================================================
# 🌌 EtherSym Finance — main.py (modo simbiótico completo)
# =========================================================
# - Double DQN + N-Step + AMP + Poda + Regeneração + Homeostase
# - Início aleatório (uniforme ou por volatilidade)
# - Prints só ao fim de cada episódio
# =========================================================

import os, time, random, torch
from torch.amp import GradScaler, autocast   # ✅ AMP moderno
import numpy as np, pandas as pd

from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL, GAMMA, BATCH
from network import criar_modelo
from memory import salvar_estado, carregar_estado, RingReplay, NStepBuffer
from env import Env, make_feats, START_MODE

# =========================================================
# ⚙️ Hiperparâmetros principais
# =========================================================
CSV = "binance_BTC_USDT_1h_2y.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)
LIMIAR = 0.003              # precisão exigida (0.3%)
RESTART_EQUITY = 0.85
EPSILON_DECAY, EPSILON_MIN = 0.9985, 0.05
TARGET_TAU, TARGET_SYNC_HARD = 0.005, 10_000
LOG_INTERVAL = 500
AUTOSAVE_EVERY = 120
N_STEP, MIN_REPLAY, ACTION_REPEAT = 3, 4096, 1
PODA_INTERVAL = 20_000
PODA_LIMIAR_BASE = 0.002

def a_to_idx(a: int): return int(a + 1)

# =========================================================
# 🎯 Política (ε-greedy + softmax temperado)
# =========================================================
@torch.no_grad()
def escolher_acao(modelo, s, eps):
    if random.random() < eps:
        return int(np.random.choice(ACTION_SPACE))
    x = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = modelo(x)
    probs = torch.softmax(logits / 0.8, dim=1).squeeze(0).cpu().numpy()
    return int(np.random.choice(ACTION_SPACE, p=probs))

# =========================================================
# 📏 Métricas sniper
# =========================================================
def sniper_error(ret, a, limiar=LIMIAR):
    if a == 0:
        return abs(ret)
    elif a == 1:
        return max(0.0, limiar - ret)
    else:
        return max(0.0, limiar + ret)

def is_hit(err, limiar=LIMIAR):
    return err < limiar

# =========================================================
# 🚀 Loop principal
# =========================================================
def run():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"CSV não encontrado: {CSV}")

    df = pd.read_csv(CSV)
    base, price = make_feats(df)
    env = Env(base, price)

    modelo, alvo, opt, loss_fn = criar_modelo(DEVICE)
    alvo.eval()
    modelo.train(True)

    replay = RingReplay(state_dim=10, capacity=MEMORIA_MAX, device=DEVICE)
    nbuf   = NStepBuffer(N_STEP, GAMMA)

    # ✅ GradScaler moderno (sem FutureWarning)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # carregar estado salvo
    _mem_legacy, EPSILON, media_antiga = carregar_estado(modelo, opt)
    if EPSILON is None:
        EPSILON = EPSILON_INICIAL

    # métricas
    total_steps = 0
    melhor_streak = 0
    acertos_total, erros_total = 0, 0
    melhor_media = float(media_antiga or -1e9)
    episode = 0
    autosave_t0 = time.time()
    steps_desde_poda = 0

    print(f"🧠 Iniciando treino simbiótico — modo START='{START_MODE}' — device={DEVICE.type}")

    while True:
        episode += 1
        s = env.reset()

        # ✅ t_init robusto: usa last_reset_t se existir, senão env.t
        t_init = int(getattr(env, "last_reset_t", getattr(env, "t", 0)))

        done = False
        total_reward_ep = 0.0
        streak = 0
        episode_acertos, episode_erros = 0, 0

        while not done:
            total_steps += 1
            steps_desde_poda += 1

            # escolher ação
            a = escolher_acao(modelo, s, EPSILON)
            sp, r, done_env, info = env.step(a, repeats=ACTION_REPEAT)
            total_reward_ep += r

            # avaliar erro/acerto
            ret = float(info.get("ret", 0.0))
            err = sniper_error(ret, a, LIMIAR)
            hit = is_hit(err, LIMIAR)

            if hit:
                acertos_total += 1
                episode_acertos += 1
                streak += 1
                if streak > melhor_streak:
                    melhor_streak = streak
            else:
                erros_total += 1
                episode_erros += 1
                streak = 0

            # N-step
            nbuf.push(s, a, r)
            flush = (len(nbuf.traj) == N_STEP) or (not hit) or done_env
            if flush:
                item = nbuf.flush(sp, (not hit) or done_env)
                if item is not None:
                    s0, a0, Rn, sn, dn = item
                    replay.append(s0, a_to_idx(a0), Rn, sn, float(dn))

            s = sp

            # aprendizagem
            if len(replay) >= MIN_REPLAY:
                estados_t, acoes_t, recompensas_t, novos_estados_t, finais_t = replay.sample(BATCH)

                with torch.no_grad():
                    next_online_q = modelo(novos_estados_t)
                    next_actions  = torch.argmax(next_online_q, dim=1, keepdim=True)
                    next_target_q = alvo(novos_estados_t).gather(1, next_actions).squeeze(1)
                    alvo_q        = recompensas_t + (GAMMA ** N_STEP) * next_target_q * (1.0 - finais_t)

                opt.zero_grad(set_to_none=True)
                # ✅ autocast moderno
                with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                    q_vals = modelo(estados_t).gather(1, acoes_t).squeeze(1)
                    perda  = loss_fn(q_vals, alvo_q)

                scaler.scale(perda).backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                # soft update
                with torch.no_grad():
                    for p_t, p in zip(alvo.parameters(), modelo.parameters()):
                        p_t.data.mul_(1.0 - TARGET_TAU).add_(TARGET_TAU * p.data)

                # hard sync periódico
                if total_steps % TARGET_SYNC_HARD == 0:
                    alvo.load_state_dict(modelo.state_dict())

            # decaimento do ε
            EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

            # 🧬 Poda simbiótica periódica
            if steps_desde_poda >= PODA_INTERVAL and len(replay) >= MIN_REPLAY:
                try:
                    taxa_poda = modelo.aplicar_poda(PODA_LIMIAR_BASE)
                    modelo.regenerar_sinapses(taxa_poda)
                    print(f"🌿 Poda simbiótica aplicada | taxa={taxa_poda:.3%}")
                except Exception as e:
                    print(f"⚠️ Erro na poda: {e}")
                steps_desde_poda = 0

            # condição de término
            if (not hit) or done_env or (info.get("eq", 1.0) < RESTART_EQUITY):
                done = True

            # autosave de fallback
            if (time.time() - autosave_t0) > (AUTOSAVE_EVERY * 5):
                autosave_t0 = time.time()
                salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)

        # =========================================================
        # 🧩 Fim do episódio
        # =========================================================
        taxa_acerto_total = acertos_total / max(1, acertos_total + erros_total)
        taxa_ep = episode_acertos / max(1, episode_acertos + episode_erros)

        try:
            modelo.verificar_homeostase(total_reward_ep)
        except Exception:
            pass

        print("------------------------------------------------------------")
        print(f"Episódio {episode:5d} | steps_total={total_steps:7d} | t_init={t_init:6d} | start_mode={START_MODE}")
        print(f"Reward_ep={total_reward_ep:+.4f} | acertos_ep={episode_acertos} | erros_ep={episode_erros} | taxa_ep={taxa_ep*100:5.2f}%")
        print(f"Streak={streak} | Melhor streak={melhor_streak} | taxa_total={taxa_acerto_total*100:5.2f}%")
        print(f"ε={EPSILON:.3f} | memória={len(replay):d} | melhor_media={melhor_media:+.4f}")
        print("Salvando estado no fim do episódio...")
        salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
        print("------------------------------------------------------------")

        # salva se superou melhor média
        if total_reward_ep > melhor_media:
            melhor_media = total_reward_ep
            print(f"🏆 Nova melhor média de episódio: {melhor_media:+.4f}")
            salvar_estado(modelo, opt, replay, EPSILON, melhor_media)

        time.sleep(0.05)

# =========================================================
# ▶️ Execução direta
# =========================================================
if __name__ == "__main__":
    run()
