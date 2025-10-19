# =========================================================
# 🌌 EtherSym Finance — main_v8f_full.py (versão ultra-otimizada)
# =========================================================
# - Double DQN + N-Step + PER + Regressão contínua
# - AMP + GradClip + Homeostase + Poda + Regeneração
# - Anneal ε, Temperatura e β-PER + Rollback simbiótico
# - Otimizações: torch.compile (max-autotune), warmup CUDA,
#   inferência sem grad, liberação VRAM/GC, checkpoints assíncronos
# =========================================================

import time, torch, gc, threading
from core.setup import setup_simbiotico
from core.scheduler import update_params
from core.collector import collect_step
from core.trainer import train_step
from core.checkpoints import check_vitoria, rollback_guard
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.maintenance import aplicar_poda, regenerar_sinapses, verificar_homeostase, homeostase_replay
from core.logger import log_status
from core.hyperparams import *
from memory import salvar_estado
import logging
# =========================================================
# 🧠 Setup simbiótico inicial
# =========================================================
env, modelo, alvo, opt, replay, nbuf, scaler = setup_simbiotico()
best_global = carregar_patrimonio_global()

EPSILON, lr_now, temp_now, beta_per = 1.0, LR, TEMP_INI, BETA_PER_INI
ema_q = ema_r = 0.0
total_steps = 0
last_loss = 0.0
last_y_pred = 0.0
episodio = 0
cooldown_until = 0
rollbacks = 0
last_good = None
loop_start = time.perf_counter()

logging.info(f"🧠 Iniciando treino simbiótico | device={DEVICE.type} | patrimônio inicial={best_global:.2f}")

# =========================================================
# 🔁 Loop principal (Episódios)
# =========================================================
while True:
    episodio += 1

    # 🧹 Limpeza simbiótica pré-reset
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    # =========================================================
    # 🔄 Reset + Warmup CUDA simbiótico
    # =========================================================
    s = env.reset()
    env.current_state = s
    try:
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            _ = modelo(torch.tensor(s, device=DEVICE).unsqueeze(0))
        torch.cuda.synchronize()
        logging.info(f"🔥 Warmup CUDA simbiótico executado (episódio {episodio})")
    except Exception as e:
        logging.info(f"[WARN] Warmup simbiótico falhou: {e}")

    # Estado inicial do episódio
    done = False
    total_reward_ep = 0.0
    capital = CAPITAL_INICIAL
    max_patrimonio = CAPITAL_INICIAL
    melhor_patrimonio_ep = CAPITAL_INICIAL
    ep_start = time.perf_counter()

    # =========================================================
    # 🔂 Loop interno (steps)
    # =========================================================
    while not done:
        total_steps += 1

        # 🔧 Atualização simbiótica de parâmetros
        lr_now, EPSILON, temp_now, beta_per, eps_now = update_params(
            total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt
        )

        # 🎯 Coleta de experiência simbiótica (inference only)
        with torch.inference_mode():
            s, done, info, total_reward_ep = collect_step(
                env, modelo, DEVICE, eps_now, replay, nbuf, total_reward_ep
            )

        # 🧠 Treinamento simbiótico (Q + Regressão)
        can_train = (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until)
        if can_train:
            loss, ema_q, ema_r = train_step(modelo, alvo, opt, replay, scaler, ema_q, ema_r, total_steps)
            last_loss = loss if loss is not None else last_loss

            # ⚠ Proteção simbiótica de rollback
            if rollback_guard(loss, total_steps, modelo, alvo, opt, replay, EPSILON):
                cooldown_until = total_steps + COOLDOWN_STEPS
                rollbacks += 1
                logging.info(f"⚠ Rollback simbiótico #{rollbacks} | cooldown até {cooldown_until}")

        # 🏆 Vitória simbiótica (melhor patrimônio)
        best_global = check_vitoria(
            info.get("patrimonio", 0.0),
            best_global, modelo, opt, replay,
            EPSILON, total_reward_ep
        )

        # 🌿 Homeostase e regeneração
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = aplicar_poda(modelo)
            regenerar_sinapses(modelo, taxa)
            verificar_homeostase(modelo, last_loss)
            logging.info(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

        if total_steps % HOMEOSTASE_EVERY == 0:
            homeostase_replay(replay)

        modelo.verificar_homeostase(last_loss)

        # 🔄 Replay trimming (mantém leve)
        if total_steps % 20000 == 0 and len(replay) > MEMORIA_MAX * 0.9:
            try:
                replay.trim(capacity=int(MEMORIA_MAX * 0.85))
                logging.info(f"🧽 Replay reduzido para {int(MEMORIA_MAX*0.85)} amostras")
            except Exception as e:
                logging.info(f"[WARN] Falha ao compactar replay: {e}")

        # 🧾 Logging simbiótico
        if total_steps % PRINT_EVERY == 0:
            energia = float(info.get("energia", 1.0))
            last_y_pred = float(info.get("ret", 0.0))
            log_status(
                episodio, total_steps,
                float(info.get("capital", capital)),
                float(info.get("patrimonio", 0.0)),
                float(info.get("max_patrimonio", max_patrimonio)),
                eps_now, temp_now, beta_per, lr_now,
                energia, last_y_pred, last_loss
            )
            loop_time = time.perf_counter() - loop_start
            logging.info(f"⏱️ Tempo loop={loop_time:.3f}s | step={total_steps}")
            loop_start = time.perf_counter()

        # 💾 Checkpoint simbiótico (thread)
        if total_steps % SAVE_EVERY == 0 and len(replay) >= MIN_REPLAY:
            threading.Thread(
                target=salvar_estado,
                args=(modelo, opt, replay, EPSILON, total_reward_ep),
                daemon=True
            ).start()
            logging.info(f"💾 Checkpoint simbiótico salvo | step={total_steps}")

        # 💾 Snapshot de rollback
        if total_steps % ROLLBACK_EVERY == 0 and len(replay) >= MIN_REPLAY:
            last_good = {
                "model": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
                "target": {k: v.detach().cpu().clone() for k, v in alvo.state_dict().items()},
                "opt": opt.state_dict(),
                "eps": float(EPSILON),
                "lr": float(lr_now),
                "temp": float(temp_now),
            }

    # =========================================================
    # 🔚 Encerramento do episódio simbiótico
    # =========================================================
    salvar_patrimonio_global(best_global)
    logging.info(f"\n🔁 Episódio {episodio:04d} finalizado | cap_final={info.get('capital', 0):.2f} | best={best_global:.2f}\n")

    # =========================================================
    # 💀 Falência simbiótica → limpeza + warmup
    # =========================================================
    if info.get("capital", 0) <= 1.0:
        logging.info("💀 Falência simbiótica — reiniciando ambiente")

        # 🔻 Limpeza de tensores órfãos e reset de scaler
        try:
            del last_good, loss, ema_q, ema_r
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        if hasattr(scaler, "update"):
            scaler._enabled = False
            scaler = torch.amp.GradScaler(enabled=True)

        # ♻ Reset e warmup pós-falência
        s = env.reset()
        env.current_state = s
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast():
                _ = modelo(torch.tensor(s, device=DEVICE).unsqueeze(0))
            torch.cuda.synchronize()
            logging.info("🔥 Warmup CUDA pós-falência executado.")
        except Exception as e:
            logging.info(f"[WARN] Warmup pós-falência falhou: {e}")
