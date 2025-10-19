# =========================================================
# 🌌 EtherSym Finance — main_v7g_parallel.py
# =========================================================
# - Double DQN + N-Step + Prioritized Replay (PER robusto)
# - Coleta paralela (EnvPool) com múltiplos ambientes CPU
# - Regressão contínua normalizada (tanh-free) + freeze inicial
# - AMP + GradClip + torch.compile + Homeostase + Poda
# - Anneal de ε, Temperatura e β-PER + LR Warmup + CosineDecay
# =========================================================

import os, math, time, random, warnings
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from network import criar_modelo
from core.env_workers import EnvPool
from memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
from core.utils import escolher_acao, soft_update, set_lr, is_bad_number, a_to_idx
from core.losses import loss_q_hibrida, loss_regressao
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.hyperparams import *

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# =========================================================
# 🚀 Setup
# =========================================================
if not os.path.exists(CSV):
    raise FileNotFoundError(f"CSV não encontrado: {CSV}")

df = pd.read_csv(CSV)
base, price = df.iloc[:, 1:].to_numpy(), df["close"].to_numpy()

# 🌿 Ambientes paralelos (CPU)
N_ENVS = max(2, os.cpu_count() // 2)
pool = EnvPool(N_ENVS, base, price)
states = pool.states

# 🌿 Modelo simbiótico
modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)
modelo = torch.compile(modelo)
alvo = torch.compile(alvo)

replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
nbuf = NStepBuffer(N_STEP, GAMMA)
scaler = GradScaler("cuda", enabled=AMP)

lr_now = LR
set_lr(opt, lr_now)

_, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL

print(f"🧠 Iniciando treino simbiótico paralelo | GPU={DEVICE.type} | ENVS={N_ENVS}")

total_steps, episodio = 0, 0
last_loss, last_y_pred = 0.0, 0.0
temp_now, beta_per = TEMP_INI, BETA_PER_INI
ema_q, ema_r = None, None
cooldown_until, rollbacks, last_good = 0, 0, None

best_global = carregar_patrimonio_global()
print(f"🏁 Patrimônio global inicial carregado: {best_global:.2f}")

# =========================================================
# 🎮 Loop simbiótico principal
# =========================================================
while True:
    episodio += 1
    done_flags = [False] * N_ENVS
    capital, max_patrimonio = CAPITAL_INICIAL, CAPITAL_INICIAL
    total_reward_ep, melhor_patrimonio_ep = 0.0, CAPITAL_INICIAL

    while not all(done_flags):
        total_steps += 1
        warmup = total_steps < 3_000
        eps_now = 1.0 if warmup else max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        if len(replay) >= MIN_REPLAY:
            EPSILON = eps_now

        if not warmup:
            temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)
            beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)
            try:
                replay.beta = float(beta_per)
            except:
                pass

        # LR schedule
        if total_steps <= LR_WARMUP_STEPS:
            lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
        else:
            progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)
            lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        set_lr(opt, lr_now)

        # =================================================
        # 🎮 Coleta simbiótica paralela
        # =================================================
        actions = []
        for s_cur in states:
            a, conf = escolher_acao(modelo, s_cur, DEVICE, eps_now, capital, 0.0)
            actions.append(a)

        states_next, rs, dones, infos = pool.step_all(actions)
        for i in range(N_ENVS):
            r = float(rs[i])
            total_reward_ep += r
            done_flags[i] = done_flags[i] or dones[i]
            capital = float(infos[i].get("capital", capital))
            patrimonio = float(infos[i].get("patrimonio", capital))
            melhor_patrimonio_ep = max(melhor_patrimonio_ep, patrimonio)

            nbuf.push(states[i], actions[i], r, infos[i].get("ret", 0.0))
            if len(nbuf.traj) == N_STEP or dones[i]:
                item = nbuf.flush(states_next[i], dones[i])
                if item:
                    s0, a0, Rn, sn, dn, y0 = item
                    replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

        states = states_next

        # =================================================
        # 🎓 Aprendizado simbiótico GPU
        # =================================================
        can_train = (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until)
        if can_train:
            (estados_t, acoes_t, recompensas_t, novos_estados_t,
             finais_t, idx, w, y_ret_t) = replay.sample(BATCH)

            with torch.no_grad():
                next_q_online, _ = modelo(novos_estados_t)
                next_q_target, _ = alvo(novos_estados_t)
                next_q_online = next_q_online.clamp_(-Q_CLAMP, Q_CLAMP)
                next_q_target = next_q_target.clamp_(-Q_CLAMP, Q_CLAMP)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_best = next_q_target.gather(1, next_actions).squeeze(1)
                alvo_q = recompensas_t + (GAMMA ** N_STEP) * next_best * (1.0 - finais_t)
                alvo_q = alvo_q.clamp_(-Q_TARGET_CLAMP, Q_TARGET_CLAMP)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=AMP):
                q_vals, y_pred = modelo(estados_t)
                q_vals = q_vals.clamp_(-Q_CLAMP, Q_CLAMP)
                q_sel = q_vals.gather(1, acoes_t).squeeze(1)

                do_reg = total_steps >= REG_FREEZE_STEPS
                y_ret_t = torch.nan_to_num(y_ret_t, nan=0.0)
                Y_CLAMP = max(0.05, 0.2 * (1 - math.exp(-total_steps / 50000)))
                y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP

                loss_q = loss_q_hibrida(q_sel, alvo_q)
                loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)

                ema_q = loss_q.item() if ema_q is None else 0.98 * ema_q + 0.02 * loss_q.item()
                ema_r = loss_reg.item() if ema_r is None else 0.98 * ema_r + 0.02 * loss_reg.item()

                lambda_eff = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
                loss_total = loss_q + lambda_eff * loss_reg

            # Gradientes e rollback
            if (is_bad_number(loss_total) or float(loss_total.item()) > LOSS_GUARD):
                cooldown_until = total_steps + COOLDOWN_STEPS
                if last_good is not None and rollbacks < MAX_ROLLBACKS:
                    modelo.load_state_dict(last_good["model"], strict=False)
                    alvo.load_state_dict(last_good["target"], strict=False)
                    rollbacks += 1
                    print(f"⚠ Rollback simbiótico #{rollbacks}")
                opt.zero_grad(set_to_none=True)
                scaler = GradScaler("cuda", enabled=AMP)
            else:
                scaler.scale(loss_total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
                with torch.no_grad():
                    td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).cpu().numpy()
                    replay.update_priority(idx, td_error)
                    tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, loss_q.item())))
                    soft_update(alvo, modelo, tau)

            last_loss = float(loss_total.item()) if torch.isfinite(loss_total) else last_loss
            last_y_pred = float(y_pred[-1].item()) if 'y_pred' in locals() else last_y_pred

        # =================================================
        # 🌿 Manutenção simbiótica
        # =================================================
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = modelo.aplicar_poda()
            modelo.regenerar_sinapses(taxa)
            modelo.verificar_homeostase(last_loss)
            print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

        if total_steps % HOMEOSTASE_EVERY == 0:
            replay.homeostase()

        modelo.verificar_homeostase(total_reward_ep / max(1, (total_steps % 10_000)))

        # =================================================
        # 🧾 Logs e persistência
        # =================================================
        if total_steps % PRINT_EVERY == 0:
            energia = float(np.mean([info.get("energia", 1.0) for info in infos]))
            print(f"[Ep {episodio:04d} | {total_steps:>8}] "
                  f"ε={eps_now:.3f} | τ={temp_now:.2f} | β={beta_per:.2f} | "
                  f"lr={lr_now:.6f} | enr={energia:.2f} | Δpred={last_y_pred:+.4f} | loss={last_loss:.5f}")

        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

        # =================================================
        # 💾 Patrimônio global
        # =================================================
        MARGEM_VITORIA = 0.01
        if melhor_patrimonio_ep > best_global * (1 + MARGEM_VITORIA) and len(replay) >= MIN_REPLAY:
            best_global = melhor_patrimonio_ep
            salvar_patrimonio_global(best_global)
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"🏆 Novo melhor patrimônio global={best_global:.2f} | step={total_steps}")
