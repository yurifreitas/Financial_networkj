# =========================================================
# 🌌 EtherSym Finance — main_v7f_parallel_persistent_fixtrain.py
# =========================================================
# - idêntico ao v7f original
# - apenas o env.step(a) roda em subprocesso persistente (via Pipe)
# - preserva: homeostase, poda, regeneração, rollback, LR, anneal etc.
# =========================================================

import os, time, math, random, warnings, multiprocessing as mp
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from network import criar_modelo
from env import Env, make_feats
from memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
import json
from core.utils import escolher_acao, soft_update, set_lr, is_bad_number, a_to_idx
from core.losses import loss_q_hibrida, loss_regressao
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.hyperparams import *

# =========================================================
# 🌱 Processo persistente do ambiente simbiótico
# =========================================================
def env_process(pipe, base, price):
    """Loop persistente de ambiente simbiótico em subprocesso."""
    env = Env(base, price)
    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break

        if cmd == "step":
            a = data
            sp, r, done, info = env.step(a)
            # envia arrays puros (CPU)
            pipe.send((sp.astype(np.float32), float(r), done, info,
                       env.t, env.capital, env.pos, env.energia, env.preco_entrada))
        elif cmd == "reset":
            s = env.reset()
            pipe.send((s.astype(np.float32), env.t, env.capital, env.pos, env.energia, env.preco_entrada))
        elif cmd == "close":
            break
    pipe.close()

# =========================================================
# 🚀 Setup simbiótico
# =========================================================
if not os.path.exists(CSV):
    raise FileNotFoundError(f"CSV não encontrado: {CSV}")

df = pd.read_csv(CSV)
base, price = make_feats(df)

# Inicializa ambiente em subprocesso
parent_conn, child_conn = mp.Pipe()
env_proc = mp.Process(target=env_process, args=(child_conn, base, price))
env_proc.start()

# Reset inicial
parent_conn.send(("reset", None))
s, t, cap, pos, enr, pre = parent_conn.recv()

# Modelo + otimizador
modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)
replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
nbuf = NStepBuffer(N_STEP, GAMMA)
scaler = GradScaler("cuda", enabled=AMP)

# 🔧 Precisão otimizada para Ampere/Lovelace
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

lr_now = LR
set_lr(opt, lr_now)

_, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL

best_global = carregar_patrimonio_global()
print(f"🧠 Iniciando treino simbiótico paralelo | device={DEVICE.type}")
print(f"🏁 Patrimônio global inicial carregado: {best_global:.2f}")

# =========================================================
# 🎮 Loop principal simbiótico (original completo)
# =========================================================
total_steps, episodio = 0, 0
last_loss, last_y_pred = 0.0, 0.0
temp_now, beta_per = TEMP_INI, BETA_PER_INI
ema_q, ema_r = None, None
cooldown_until = 0

while True:
    episodio += 1
    parent_conn.send(("reset", None))
    s, t, capital, posicao, energia, preco_entrada = parent_conn.recv()

    done = False
    max_patrimonio = CAPITAL_INICIAL
    total_reward_ep = 0.0
    melhor_patrimonio_ep = CAPITAL_INICIAL

    while not done:
        total_steps += 1
        warmup = total_steps < 3_000

        # parâmetros dinâmicos
        eps_now = 1.0 if warmup else max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        if len(replay) >= MIN_REPLAY:
            EPSILON = eps_now

        if not warmup:
            temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)
            beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)
            replay.beta = float(beta_per)

        # schedule de LR
        if total_steps <= LR_WARMUP_STEPS:
            lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
        else:
            progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)
            lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        set_lr(opt, lr_now)

        # =====================================================
        # ⚙️ Step simbiótico paralelo persistente
        # =====================================================
        s_cur = np.asarray(s, dtype=np.float32)
        a, conf = escolher_acao(modelo, s_cur, DEVICE, eps_now, capital, posicao)

        parent_conn.send(("step", a))
        sp, r, done_env, info, t, capital, posicao, energia, preco_entrada = parent_conn.recv()
        sp = np.asarray(sp, dtype=np.float32)

        total_reward_ep += float(r)
        patrimonio = float(info.get("patrimonio", capital))
        max_patrimonio = max(max_patrimonio, patrimonio)
        y_ret = float(info.get("ret", 0.0))
        melhor_patrimonio_ep = max(melhor_patrimonio_ep, patrimonio)

        # =====================================================
        # 🏆 Vitória simbiótica
        # =====================================================
        if patrimonio >= 2.5 * CAPITAL_INICIAL:
            print(f"\n🏆 Vitória simbiótica no episódio {episodio:04d} | patrimônio={patrimonio:.2f}\n")
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            parent_conn.send(("reset", None))
            s, t, capital, posicao, energia, preco_entrada = parent_conn.recv()
            done = True
            continue

        # Push N-step real
        nbuf.push(np.copy(s_cur), int(a), float(r), float(y_ret))
        if len(nbuf.traj) >= N_STEP or done_env:
            item = nbuf.flush(sp, done_env)
            if item:
                s0, a0, Rn, sn, dn, y0 = item
                replay.append(
                    np.asarray(s0, dtype=np.float32),
                    a_to_idx(a0),
                    float(Rn),
                    np.asarray(sn, dtype=np.float32),
                    float(dn),
                    float(y0),
                )

        s = sp
        done = done_env

        # =====================================================
        # 🎓 Aprendizado simbiótico completo
        # =====================================================
        can_train = (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until)
        if can_train:
            batch = replay.sample(BATCH)
            (estados_t, acoes_t, recompensas_t, novos_estados_t,
             finais_t, idx, w, y_ret_t) = batch

            # conversão segura → tensores CUDA
            estados_t       = torch.as_tensor(estados_t, dtype=torch.float32, device=DEVICE)
            novos_estados_t = torch.as_tensor(novos_estados_t, dtype=torch.float32, device=DEVICE)
            acoes_t         = torch.as_tensor(acoes_t, dtype=torch.long, device=DEVICE)
            recompensas_t   = torch.as_tensor(recompensas_t, dtype=torch.float32, device=DEVICE)
            finais_t        = torch.as_tensor(finais_t, dtype=torch.float32, device=DEVICE)
            y_ret_t         = torch.as_tensor(y_ret_t, dtype=torch.float32, device=DEVICE)

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
                y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP
                loss_q = loss_q_hibrida(q_sel, alvo_q)
                loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)
                if ema_q is None:
                    ema_q, ema_r = float(loss_q.item()), float(loss_reg.item())
                ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
                ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())
                lambda_eff = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
                loss_total = loss_q + lambda_eff * loss_reg

            if (is_bad_number(loss_total) or float(loss_total.item()) > LOSS_GUARD):
                cooldown_until = total_steps + COOLDOWN_STEPS
                print(f"⚠ Cooldown ativado até {cooldown_until}")
            else:
                scaler.scale(loss_total).backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).detach().cpu().numpy()
                    replay.update_priority(idx, td_error)
                    tau = min(0.01, TARGET_TAU_BASE)
                    soft_update(alvo, modelo, tau)
                    if total_steps % HARD_SYNC_EVERY == 0:
                        alvo.load_state_dict(modelo.state_dict())

            last_loss = float(loss_total.item())
            last_y_pred = float(y_pred[-1].item())

        # =====================================================
        # 🌿 Manutenção simbiótica
        # =====================================================
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = modelo.aplicar_poda()
            modelo.regenerar_sinapses(taxa)
            modelo.verificar_homeostase(last_loss)
            print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

        if total_steps % HOMEOSTASE_EVERY == 0:
            replay.homeostase()

        modelo.verificar_homeostase(total_reward_ep / max(1, (total_steps % 10_000)))

        # =====================================================
        # 🧾 Logs / checkpoints
        # =====================================================
        if (total_steps % PRINT_EVERY == 0) or done:
            print(
                f"[Ep {episodio:04d} | {total_steps:>8}] cap={capital:>9.2f} | pat={patrimonio:>9.2f} | "
                f"max={max_patrimonio:>9.2f} | ε={eps_now:.3f} | τ={temp_now:.2f} | β={beta_per:.2f} | "
                f"lr={lr_now:.6f} | enr={energia:.2f} | Δpred={last_y_pred:+.4f} | loss={last_loss:.5f}"
            )

        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

        if done:
            salvar_patrimonio_global(best_global)
            print(f"\n🔁 Episódio {episodio:04d} finalizado | cap_final={capital:.2f}\n")
            break

# Finalização limpa
parent_conn.send(("close", None))
env_proc.join()
print("✅ Processo simbiótico encerrado com sucesso.")
