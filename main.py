# =========================================================
# 🌌 EtherSym Finance — main_v8f_full_async.py (corrigido)
# =========================================================
# - Execução simbiótica paralela e não bloqueante
# - Correção: uso adequado de asyncio (sem SyntaxError)
# =========================================================

import torch, asyncio, concurrent.futures, gc
from core.setup import setup_simbiotico
from core.scheduler import update_params
from core.collector import collect_step
from core.trainer import train_step
from core.checkpoints import check_vitoria, rollback_guard
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.maintenance import (
    aplicar_poda, regenerar_sinapses,
    verificar_homeostase, homeostase_replay
)
from core.logger import log_status
from core.hyperparams import *
from memory import salvar_estado


# =========================================================
# 🔁 Loop principal assíncrono
# =========================================================
async def main():
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

    print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type} | patrimônio inicial={best_global:.2f}")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()

    # =====================================================
    # 🔁 Episódios simbióticos
    # =====================================================
    while True:
        episodio += 1
        s = env.reset()
        env.current_state = s
        done = False
        total_reward_ep = 0.0
        capital = CAPITAL_INICIAL
        max_patrimonio = CAPITAL_INICIAL
        melhor_patrimonio_ep = CAPITAL_INICIAL

        while not done:
            total_steps += 1

            # 🔧 Atualiza parâmetros dinâmicos
            lr_now, EPSILON, temp_now, beta_per, eps_now = update_params(
                total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt
            )

            # 🎯 Coleta de experiência (thread leve)
            s, done, info, total_reward_ep = await loop.run_in_executor(
                executor,
                collect_step,
                env, modelo, DEVICE, eps_now, replay, nbuf, total_reward_ep
            )

            # 🧠 Treino simbiótico
            can_train = (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until)
            if can_train:
                loss, ema_q, ema_r = train_step(modelo, alvo, opt, replay, scaler, ema_q, ema_r, total_steps)
                last_loss = loss if loss is not None else last_loss

                if rollback_guard(loss, total_steps, modelo, alvo, opt, replay, EPSILON):
                    cooldown_until = total_steps + COOLDOWN_STEPS
                    rollbacks += 1
                    print(f"⚠ Reset simbiótico #{rollbacks} | cooldown até {cooldown_until}")

            # 🏆 Vitória simbiótica
            best_global = check_vitoria(
                info.get("patrimonio", 0.0),
                best_global,
                modelo,
                opt,
                replay,
                EPSILON,
                total_reward_ep,
            )

            # 🌿 Poda e homeostase simbiótica
            if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
                taxa = aplicar_poda(modelo)
                regenerar_sinapses(modelo, taxa)
                verificar_homeostase(modelo, last_loss)
                print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

            if total_steps % HOMEOSTASE_EVERY == 0:
                homeostase_replay(replay)

            modelo.verificar_homeostase(last_loss)

            # 🧾 Logs
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

            # 💾 Checkpoints
            if total_steps % SAVE_EVERY == 0 and len(replay) >= MIN_REPLAY:
                torch.cuda.empty_cache()
                salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
                print(f"💾 Checkpoint salvo | step={total_steps}")

            # 💾 Snapshot rollback
            if total_steps % ROLLBACK_EVERY == 0 and len(replay) >= MIN_REPLAY:
                last_good = {
                    "model": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
                    "target": {k: v.detach().cpu().clone() for k, v in alvo.state_dict().items()},
                    "opt": opt.state_dict(),
                    "eps": float(EPSILON),
                    "lr": float(lr_now),
                    "temp": float(temp_now),
                }

        # =====================================================
        # 🔚 Fim de episódio
        # =====================================================
        salvar_patrimonio_global(best_global)
        print(f"\n🔁 Episódio {episodio:04d} finalizado | cap_final={info.get('capital', 0):.2f} | best={best_global:.2f}\n")

        if info.get("capital", 0) <= 1.0:
            print("💀 Falência simbiótica — reiniciando ambiente")
            s = env.reset()
            env.current_state = s

        # limpeza simbiótica leve
        if total_steps % 4096 == 0:
            gc.collect()
            torch.cuda.empty_cache()

# =========================================================
# 🚀 Execução
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())
