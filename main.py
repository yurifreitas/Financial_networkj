# =========================================================
# 🌌 EtherSym Finance — main.py (Simbiótico Avançado v7c)
# =========================================================
# - Double DQN + N-Step + Prioritized Replay (RingReplay)
# - Regressão contínua normalizada (tanh + clamp + scale)
# - AMP + GradClip + Homeostase + Poda + Regeneração
# - Persistência incremental + hard/soft target sync
# - Temperatura simbiótica no softmax e bugfix de transições
# =========================================================

import os, time, random, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from network import criar_modelo
from env import Env, make_feats
from memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado

# =========================================================
# ⚙️ Hiperparâmetros principais
# =========================================================
CSV = "binance_BTC_USDT_1h_2y.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 256
GAMMA = 0.995
LR = 1e-4
N_STEP = 3
MEMORIA_MAX = 250_000
MIN_REPLAY = 3_000

# Exploração
WARMUP_STEPS   = 10_000
EPSILON_INICIAL = 1.0
EPSILON_DECAY   = 0.9995
EPSILON_MIN     = 0.05

# Regressão / estabilidade
LAMBDA_REG = 0.1        # peso base da regressão (ajustado via λ adaptativo)
Y_CLAMP    = 0.02       # alvo contínuo clamped a ±0.02 e normalizado p/ [-1,1]
GRAD_CLIP  = 0.5        # clipping de gradiente

# Manutenção / alvo / poda
PRINT_EVERY     = 400
SAVE_EVERY      = 10_000
PODA_EVERY      = 5_000
HOMEOSTASE_EVERY= 2_000
TARGET_TAU      = 0.005
HARD_SYNC_EVERY = 50_000

CAPITAL_INICIAL = 1_000.0

ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)
def a_to_idx(a: int) -> int: return int(a + 1)


# =========================================================
# 🎯 Política ε-greedy simbiótica
# =========================================================
@torch.no_grad()
def escolher_acao(modelo, s, eps, capital, posicao, warmup=False):
    """
    - warmup=True: 100% exploração (distribuição levemente enviesada p/ ±1)
    - warmup=False: ε-greedy + softmax(Q/1.2) e gating por confiança |y|
    - Regras: sem BUY sem capital; sem SELL sem posição
    """
    if warmup or (random.random() < eps):
        a = int(np.random.choice(ACTION_SPACE, p=[0.35, 0.30, 0.35]))
        conf = 0.5
    else:
        x = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q, y = modelo(x)
        probs = torch.softmax(q / 1.2, dim=1).squeeze(0).cpu().numpy()
        a = int(np.random.choice(ACTION_SPACE, p=probs / probs.sum()))
        conf = float(torch.tanh(torch.abs(y)).item())
        if conf < 0.35:
            a = 0

    if a == 1 and capital <= 0: a = 0
    if a == -1 and posicao <= 0: a = 0
    return a, conf


# =========================================================
# 🚀 Setup
# =========================================================
if not os.path.exists(CSV):
    raise FileNotFoundError(f"CSV não encontrado: {CSV}")

df = pd.read_csv(CSV)
base, price = make_feats(df)

env = Env(base, price)
modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)

# Replay priorizado + N-step
replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
nbuf = NStepBuffer(N_STEP, GAMMA)

scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

# Carregar checkpoint se existir (o seu carregar_estado retorna (replay_stub, eps, media))
_, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL

print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type}")


# =========================================================
# 🎮 Loop de treinamento
# =========================================================
total_steps, episodio = 0, 0
best_global = CAPITAL_INICIAL
last_loss = 0.0
last_y_pred = 0.0

# EMAs p/ λ adaptativo entre Q e regressão
ema_q, ema_r = None, None

while True:
    episodio += 1
    s = env.reset()
    done = False
    capital = CAPITAL_INICIAL
    posicao = 0.0
    max_patrimonio = CAPITAL_INICIAL
    total_reward_ep = 0.0

    while not done:
        total_steps += 1
        warmup = total_steps < WARMUP_STEPS

        # ε dinâmico (só decai após MIN_REPLAY)
        if warmup:
            eps_now = 1.0
        else:
            if len(replay) >= MIN_REPLAY:
                EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
            eps_now = EPSILON

        # -------- ação / ambiente (BUGFIX: use s_cur antes do step) --------
        s_cur = s
        a, conf = escolher_acao(modelo, s_cur, eps_now, capital, posicao, warmup)
        sp, r, done_env, info = env.step(a)
        total_reward_ep += r

        capital = float(info.get("capital", capital))
        patrimonio = float(info.get("patrimonio", capital))
        max_patrimonio = max(max_patrimonio, float(info.get("max_patrimonio", patrimonio)))
        posicao = float(env.pos)
        y_ret = float(info.get("ret", 0.0))

        # -------- N-step buffer + prioritized replay --------
        nbuf.push(s_cur, a, r, y_ret)
        if len(nbuf.traj) == N_STEP or done_env:
            item = nbuf.flush(sp, done_env)
            if item:
                s0, a0, Rn, sn, dn, y0 = item
                replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

        s = sp
        done = done_env

        # -------- treino --------
        if len(replay) >= MIN_REPLAY:
            (estados_t, acoes_t, recompensas_t, novos_estados_t,
             finais_t, idx, w, y_ret_t) = replay.sample(BATCH)

            with torch.no_grad():
                # Double-DQN: seleção no online, avaliação no alvo
                next_q_online, _ = modelo(novos_estados_t)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_q_target, _ = alvo(novos_estados_t)
                next_best = next_q_target.gather(1, next_actions).squeeze(1)
                alvo_q = recompensas_t + (GAMMA ** N_STEP) * next_best * (1.0 - finais_t)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                q_vals, y_pred = modelo(estados_t)
                q_sel = q_vals.gather(1, acoes_t).squeeze(1)

                # PER loss (IS weights)
                per_sample = F.smooth_l1_loss(q_sel, alvo_q, reduction="none", beta=0.8)
                loss_q = (w * per_sample).mean()

                # Regressão contínua normalizada p/ [-1,1]
                y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP
                loss_reg = F.smooth_l1_loss(y_pred, y_target, beta=0.5)

                # λ adaptativo (mantém regressão controlada)
                if ema_q is None:
                    ema_q, ema_r = float(loss_q.item()), float(loss_reg.item())
                ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
                ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())
                lambda_eff = LAMBDA_REG * max(0.05, min(1.0, ema_q / (ema_r + 1e-6)))

                loss_total = loss_q + lambda_eff * loss_reg

            scaler.scale(loss_total).backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                td_error = (alvo_q - q_sel).abs().detach().cpu().numpy()
                replay.update_priority(idx, td_error)

                # Soft update alvo
                for p_t, p in zip(alvo.parameters(), modelo.parameters()):
                    p_t.data.mul_(1.0 - TARGET_TAU).add_(TARGET_TAU * p.data)

                # Hard sync periódico
                if total_steps % HARD_SYNC_EVERY == 0:
                    alvo.load_state_dict(modelo.state_dict())

            last_loss = float(loss_total.item())
            last_y_pred = float(y_pred[-1].item())

        # -------- manutenção simbiótica --------
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = modelo.aplicar_poda()
            modelo.regenerar_sinapses(taxa)
            modelo.verificar_homeostase(last_loss)
            print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

        if total_steps % HOMEOSTASE_EVERY == 0:
            replay.homeostase()  # decai prioridades gradualmente

        modelo.verificar_homeostase(total_reward_ep / max(1, total_steps % 10_000))

        # -------- logs --------
        if (total_steps % PRINT_EVERY == 0) or done:
            energia = float(info.get("energia", 1.0))
            print(
                f"[Ep {episodio:04d} | {total_steps:>8}] "
                f"cap={capital:>9.2f} | pat={patrimonio:>9.2f} | max={max_patrimonio:>9.2f} | "
                f"ε={eps_now:.3f} | enr={energia:.2f} | Δpred={last_y_pred:+.4f} | loss={last_loss:.5f}"
            )

        # -------- checkpoints --------
        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

        # -------- reset por falência (env decide) --------
        if done and capital <= 1.0:
            best_global = max(best_global, max_patrimonio)
            print(
                f"\n💀 Falência simbiótica | reset"
                f" | cap_final={capital:.2f} | melhor_global={best_global:.2f} | ε={EPSILON:.3f}\n"
            )
            time.sleep(1.0)
            break
