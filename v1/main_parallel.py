# =========================================================
# 🌌 EtherSym Finance — main.py (Simbiótico Centralizado v7g)
# =========================================================
# - Replay Buffer centralizado (PER + Homeostase)
# - N-Step Buffer nos workers
# - Coleta paralela multiprocessada
# - Treinamento simbiótico contínuo com AMP + Rollback
# - Poda, regeneração e sincronização τ adaptativa
# =========================================================

import os, math, torch, multiprocessing as mp
from torch.amp import GradScaler, autocast

from v1.network import criar_modelo
from env import Env, make_feats
from v1.memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
from v1.setup import *


# =========================================================
# ⚙️ Worker — apenas coleta transições e envia para o replay central
# =========================================================
def env_worker(rank, conn, model_state_dict):
    torch.set_num_threads(1)
    torch.manual_seed(SEED + rank)

    base, price = make_feats(df)
    env = Env(base, price)
    nbuf = NStepBuffer(N_STEP, GAMMA)
    modelo_w, _, _ = criar_modelo(DEVICE, lr=LR)
    modelo_w.load_state_dict(model_state_dict, strict=False)
    modelo_w.eval()

    EPS = 1.0
    print(f"🧩 Worker {rank} ativo")

    while True:
        s = env.reset()
        done = False
        total_r = 0.0
        capital = CAPITAL_INICIAL
        pos = 0.0

        while not done:
            a, conf = escolher_acao(modelo_w, s, DEVICE, EPS, capital, pos)
            sp, r, done_env, info = env.step(a)
            y_ret = float(info.get("ret", 0.0))
            nbuf.push(s, a, r, y_ret)

            if len(nbuf.traj) == N_STEP or done_env:
                item = nbuf.flush(sp, done_env)
                if item:
                    try:
                        conn.send(item)
                    except (BrokenPipeError, EOFError):
                        return

            s, done = sp, done_env


# =========================================================
# 🎯 Função principal — replay centralizado e treino simbiótico
# =========================================================
def main():
    # ===== Preparação base =====
    base, price = make_feats(df)
    modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)
    scaler = GradScaler("cuda", enabled=AMP)

    replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
    _, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
    EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL

    modelo = torch.compile(modelo, mode="max-autotune-no-cudagraphs", fullgraph=False)
    alvo = torch.compile(alvo, mode="max-autotune-no-cudagraphs", fullgraph=False)
    set_lr(opt, LR)

    print(f"🧠 Treino simbiótico iniciado | device={DEVICE.type}")

    # ===== Setup multiprocessado =====
    mp.set_start_method("spawn", force=True)
    num_workers = 12
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_workers)])

    workers = [
        mp.Process(target=env_worker, args=(i, child_conns[i], modelo.state_dict()))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    total_steps, best_global, cooldown_until = 0, CAPITAL_INICIAL, 0
    ema_q, ema_r, rollbacks = None, None, 0
    lr_now, temp_now, beta_per = LR, TEMP_INI, BETA_PER_INI
    last_good, last_loss, last_y_pred = None, 0.0, 0.0

    # =========================================================
    # 🔁 Loop principal — integração simbiótica e aprendizado
    # =========================================================
    while True:
        # ===== Receber dados dos workers =====
        for conn in parent_conns:
            while conn.poll():
                try:
                    s0, a0, Rn, sn, dn, y0 = conn.recv()
                    replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)
                except EOFError:
                    continue

        if len(replay) < MIN_REPLAY:
            if total_steps % 1000 == 0:
                print(f"⏳ Coletando experiências... replay={len(replay)}")
            continue

        total_steps += 1
        warmup = total_steps < 3_000

        # ===== Ajustes dinâmicos =====
        eps_now = 1.0 if warmup else max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)
        beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)
        replay.beta = float(beta_per)
        if not warmup:
            EPSILON = eps_now

        # ===== Scheduler de LR =====
        if total_steps <= LR_WARMUP_STEPS:
            lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
        else:
            progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)
            lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        set_lr(opt, lr_now)

        # ===== Amostragem do replay =====
        (s_t, a_t, r_t, sn_t, d_t, idx, w, y_t) = replay.sample(BATCH)

        with torch.no_grad():
            next_q_online, _ = modelo(sn_t)
            next_q_target, _ = alvo(sn_t)
            next_q_online = next_q_online.clone().clamp_(-Q_CLAMP, Q_CLAMP)
            next_q_target = next_q_target.clone().clamp_(-Q_CLAMP, Q_CLAMP)
            next_a = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_best = next_q_target.gather(1, next_a).squeeze(1)
            alvo_q = r_t + (GAMMA ** N_STEP) * next_best * (1.0 - d_t)
            alvo_q = alvo_q.clamp_(-Q_TARGET_CLAMP, Q_TARGET_CLAMP)

        # ===== Forward + Backward =====
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=AMP):
            q_vals, y_pred = modelo(s_t)
            q_vals = q_vals.clamp_(-Q_CLAMP, Q_CLAMP)
            q_sel = q_vals.gather(1, a_t).squeeze(1)
            loss_q = loss_q_hibrida(q_sel, alvo_q)

            y_t = torch.nan_to_num(y_t, nan=0.0)
            y_target = y_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP
            do_reg = total_steps >= REG_FREEZE_STEPS
            loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)

            if ema_q is None:
                ema_q, ema_r = float(loss_q.item()), float(loss_reg.item())
            ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
            ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())

            λ = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
            loss_total = loss_q + λ * loss_reg

        # ===== Guardiões simbióticos =====
        if is_bad_number(loss_total) or abs(loss_total.item()) > LOSS_GUARD:
            cooldown_until = total_steps + COOLDOWN_STEPS
            if last_good and rollbacks < MAX_ROLLBACKS:
                modelo.load_state_dict(last_good["model"], strict=False)
                alvo.load_state_dict(last_good["target"], strict=False)
                opt.load_state_dict(last_good["opt"])
                rollbacks += 1
                print(f"⚠ Rollback simbiótico #{rollbacks} | cooldown até {cooldown_until}")
            continue

        # ===== Backprop e atualização =====
        scaler.scale(loss_total).backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).cpu().numpy()
            replay.update_priority(idx, td_error)
            soft_update(alvo, modelo, TARGET_TAU_BASE)

        last_loss = float(loss_total.item())
        last_y_pred = float(y_pred[-1].item()) if 'y_pred' in locals() else last_y_pred

        # ===== Manutenção simbiótica =====
        if total_steps % PODA_EVERY == 0:
            taxa = modelo.aplicar_poda()
            modelo.regenerar_sinapses(taxa)
            modelo.verificar_homeostase(last_loss)
            print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}%")

        if total_steps % HOMEOSTASE_EVERY == 0:
            replay.homeostase()

        # ===== Checkpoints =====
        if total_steps % PRINT_EVERY == 0:
            print(f"[{total_steps}] replay={len(replay)} | loss={last_loss:.6f} | ε={EPSILON:.3f} | τ={temp_now:.2f}")

        if total_steps % SAVE_EVERY == 0:
            salvar_estado(modelo, opt, replay, EPSILON, last_loss)
            print(f"💾 Checkpoint salvo | step={total_steps}")


# =========================================================
if __name__ == "__main__":
    main()
