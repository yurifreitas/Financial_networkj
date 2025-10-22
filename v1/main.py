from v1.network import criar_modelo
from env import Env, make_feats
from v1.memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
from v1.setup import *
import os, json, time
from v1.noise import RuidoColepax
import gc
# Função para carregar o melhor patrimônio global
def carregar_patrimonio_global():
    if os.path.exists("recorde_maximo.json"):
        with open("recorde_maximo.json", "r") as f:
            recorde_data = json.load(f)
            return recorde_data.get("patrimonio_final", CAPITAL_INICIAL)
    return CAPITAL_INICIAL  # Caso não exista o arquivo, retorna o valor inicial

base, price = make_feats(df)
env = Env(base, price)
modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)

replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
nbuf = NStepBuffer(N_STEP, GAMMA)
scaler = GradScaler("cuda", enabled=AMP)

lr_now = LR
set_lr(opt, lr_now)

_, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL
modelo = torch.compile(modelo, mode="max-autotune-no-cudagraphs", fullgraph=False)  # Alternativa para mais otimização
alvo = torch.compile(alvo, mode="max-autotune-no-cudagraphs", fullgraph=False)

print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type}")

total_steps, episodio= 0, 0
last_loss, last_y_pred = 0.0, 0.0
temp_now, beta_per = TEMP_INI, BETA_PER_INI
ema_q, ema_r = None, None
best_global = carregar_patrimonio_global()
print(f"Patrimônio global carregado: {best_global:.2f}")
# =========================================================
# 🎮 Loop principal
# =========================================================
while True:
    episodio += 1
    s = env.reset()
    done = False
    capital = CAPITAL_INICIAL
    posicao = 0.0
    max_patrimonio = CAPITAL_INICIAL
    total_reward_ep = 0.0
    melhor_patrimonio_ep = CAPITAL_INICIAL

    while not done:
        total_steps += 1
        warmup = total_steps < 3_000

        # Epsilon/temperatura/β
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

        # LR schedule: warmup -> cosine
        if total_steps <= LR_WARMUP_STEPS:
            lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
        else:
            progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)
            lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        set_lr(opt, lr_now)

        # ===== Ambiente / coleta =====
        s_cur = s
        a, conf = escolher_acao(modelo, s_cur, DEVICE, eps_now, capital, posicao)
        sp, r, done_env, info = env.step(a)

        total_reward_ep += r
        capital = float(info.get("capital", capital))
        patrimonio = float(info.get("patrimonio", capital))
        max_patrimonio = max(max_patrimonio, float(info.get("max_patrimonio", patrimonio)))
        posicao = float(env.pos)
        y_ret = float(info.get("ret", 0.0))
        melhor_patrimonio_ep = max(melhor_patrimonio_ep, patrimonio)

        nbuf.push(s_cur, a, r, y_ret)
        if len(nbuf.traj) == N_STEP or done_env:
            item = nbuf.flush(sp, done_env)
            if item:
                s0, a0, Rn, sn, dn, y0 = item
                replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)

        s = sp
        done = done_env

        FATOR_VITORIA = 2.5  # Dobra o capital inicial
        if patrimonio >= FATOR_VITORIA * CAPITAL_INICIAL:
            done_env = True
            print(f"🏆 Vitória simbiótica! Patrimônio dobrado ({patrimonio:.2f}) no episódio {episodio}")

            # Salvando a vitória
            vitoria_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "episodio": episodio,
                "capital_final": capital,
                "patrimonio_final": patrimonio,
                "max_patrimonio": max_patrimonio,
                "energia_final": float(info.get("energia", 1.0)),
                "pontuacao": float(info.get("pontuacao", 0.0)),
                "taxa_acerto": float(info.get("taxa_acerto", 0.0)),
                "trades_win": int(info.get("trades_win", 0)),
                "trades_lose": int(info.get("trades_lose", 0)),
                "trades_total": int(info.get("trades_total", 0)),
            }

            try:
                os.makedirs("runs", exist_ok=True)
                with open(f"runs/vitoria_ep{episodio}_{int(time.time())}.json", "w") as f:
                    json.dump(vitoria_data, f, indent=2)
                print("💾 Vitória simbiótica registrada em runs/")
            except Exception as e:
                print(f"[WARN] Falha ao salvar vitória simbiótica: {e}")

        if done:
            # fim do episódio
            s = env.reset()
            if capital <= 1.0:
                best_global = max(best_global, max_patrimonio)
                print(f"\n💀 Falência simbiótica | cap_final={capital:.2f} | melhor_global={best_global:.2f}")
            break
        # ===== Snapshot "estado bom" periódico p/ rollback =====
        if (total_steps % ROLLBACK_EVERY == 0) and (len(replay) >= MIN_REPLAY):
            last_good = {
                "model": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
                "target": {k: v.detach().cpu().clone() for k, v in alvo.state_dict().items()},
                "opt": opt.state_dict(),
                "eps": float(EPSILON),
                "lr": float(lr_now),
                "temp": float(temp_now),
            }

        can_train =  total_steps % 15_000 == 0

        # ==========================================================
        # 🧬 Ciclo de Treino com Ruído Colepax + Fase Temporal
        # ==========================================================
        if can_train:
            N_TREINOS = 1_000
            ruido = RuidoColepax(base_intensity=0.03, fractal_layers=4, device=DEVICE.type)
            fase_temporal = 0.0
            print(f"\n🌌 Ciclo Colepax iniciado ({N_TREINOS} iterações)\n")

            for t in range(N_TREINOS):
                ruido.step_update()
                fase_temporal += 0.0073  # Fase oscilatória simbiótica
                if t % 200 == 0:
                    torch.cuda.empty_cache()

                # Amostragem simbiótica do replay buffer
                (estados_t, acoes_t, recompensas_t, novos_estados_t, finais_t, idx, w, y_ret_t) = replay.sample(BATCH)

                # 🌊 Perturbação simbiótica leve (autoexploração)
                if torch.rand(1).item() < 0.05:
                    freq = 0.5 + 0.5 * math.sin(fase_temporal)
                    estados_t += freq * 0.001 * torch.randn_like(estados_t)
                    novos_estados_t += freq * 0.001 * torch.randn_like(novos_estados_t)

                with torch.no_grad():
                    next_q_online, _ = modelo(novos_estados_t)
                    next_q_target, _ = alvo(novos_estados_t)
                    next_q_online = next_q_online.clone().clamp_(-Q_CLAMP, Q_CLAMP)
                    next_q_target = next_q_target.clone().clamp_(-Q_CLAMP, Q_CLAMP)

                    next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                    next_best = next_q_target.gather(1, next_actions).squeeze(1)
                    alvo_q = recompensas_t + (GAMMA ** N_STEP) * next_best * (1.0 - finais_t)
                    alvo_q = alvo_q.clone().clamp_(-Q_TARGET_CLAMP, Q_TARGET_CLAMP)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=AMP):
                    q_vals, y_pred = modelo(estados_t)

                    # ==========================================================
                    # 🔮 Ruído Colepax + Ruído Temporal de Fase
                    # ==========================================================
                    fase_ruido = 1.0 + 0.25 * math.sin(fase_temporal * 2.0)
                    ruido.base_intensity = 0.03 * fase_ruido
                    q_vals = ruido.aplicar(q_vals, modo="mix")
                    y_pred = ruido.aplicar(y_pred, modo="fractal")

                    q_sel = q_vals.gather(1, acoes_t).squeeze(1).clamp_(-Q_CLAMP, Q_CLAMP)

                    do_reg = total_steps >= REG_FREEZE_STEPS
                    y_ret_t = torch.nan_to_num(y_ret_t, nan=0.0, posinf=0.0, neginf=0.0)
                    y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP

                    loss_q = loss_q_hibrida(q_sel, alvo_q)
                    loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)

                    if ema_q is None:
                        ema_q, ema_r = float(loss_q.item()), float(loss_reg.item())
                    ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
                    ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())

                    lambda_eff = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))

                    # 🌐 Penalização energética simbiótica
                    pen_energia = sum((p ** 2).mean() * 1e-5 for p in modelo.parameters())

                    # 💠 Ruído de fase temporal (entre batches)
                    fase_ruido_temporal = 0.5 * math.sin(fase_temporal * 3.14)
                    loss_total = loss_q + lambda_eff * loss_reg + pen_energia
                    loss_total = loss_total * (1.0 + fase_ruido_temporal * 0.05)

                # Proteção simbiótica contra perda anômala
                if is_bad_number(loss_total) or abs(loss_total.item()) > LOSS_GUARD:
                    cooldown_until = total_steps + COOLDOWN_STEPS
                    print(f"⚠ Reset simbiótico ativado | perda={loss_total.item():.4f}")
                    opt.zero_grad(set_to_none=True)
                    for p in modelo.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    scaler = GradScaler("cuda", enabled=AMP)
                else:
                    scaler.scale(loss_total).backward()
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()

                    # Atualiza prioridades e soft update
                    with torch.no_grad():
                        td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).detach().cpu().numpy()
                        replay.update_priority(idx, td_error)

                        loss_scalar = float(loss_q.item())
                        tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, loss_scalar)))
                        soft_update(alvo, modelo, tau)
                        if total_steps % HARD_SYNC_EVERY == 0:
                            alvo.load_state_dict(modelo.state_dict())

                # 🔭 Log simbiótico a cada 100 steps
                if t % 100 == 0:
                    print(f"[t={t:04d}] perda={loss_total.item():.6f} | "
                        f"ruído={ruido.current_intensity:.4f} | fase={fase_ruido:.3f} | λ={lambda_eff:.3f}")

            torch.cuda.empty_cache()
            total_steps += N_TREINOS

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
        # 🧾 Logs / checkpoints
        # =================================================
        if (total_steps % PRINT_EVERY == 0) or done:
            energia = float(info.get("energia", 1.0))
            print(
                f"[Ep {episodio:04d} | {total_steps:>8}] cap={capital:>9.2f} | pat={patrimonio:>9.2f} | "
                f"max={max_patrimonio:>9.2f} | ε={eps_now:.3f} | τ={temp_now:.2f} | β={beta_per:.2f} | "
                f"lr={lr_now:.6f} | enr={energia:.2f} | Δpred={last_y_pred:+.4f} | loss={last_loss:.5f}"
            )

        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

        if melhor_patrimonio_ep > best_global and len(replay) >= MIN_REPLAY:
            best_global = melhor_patrimonio_ep
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"🏆 Novo melhor patrimônio global={best_global:.2f} | step={total_steps}")
        if done:
            # fim do episódio
            s = env.reset()
            if capital <= 1.0:
                best_global = max(best_global, max_patrimonio)
                print(
                    f"\n💀 Falência simbiótica | cap_final={capital:.2f} | "
                    f"melhor_global={best_global:.2f} | ε={EPSILON:.3f}\n"
                )
            break
