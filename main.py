# =========================================================
# 🌌 EtherSym Finance — main_v8f_turbo_simbiotico.py (universal patch)
# =========================================================
# - Compatível com PyTorch 2.6 → 2.9.1
# - torch.compile turbo + autotune persistente
# - Sem segmentation fault, sem atributos inválidos
# =========================================================

import os, warnings

# =========================================================
# 🔐 Patch simbiótico estável e compatível
# =========================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["TORCHINDUCTOR_DISABLE_FX_VALIDATION"] = "1"
os.environ["TORCHINDUCTOR_FUSE_TRIVIAL_OPS"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "cublas,triton"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torch/inductor")

# Desliga logs internos sem quebrar versões recentes
try:
    import torch._logging as _logging
    _logging.set_logs()  # limpa logs simbióticos
except Exception:
    pass

# =========================================================
# ⚙️ Configuração avançada (versão adaptativa)
# =========================================================
import torch

# Dynamo / Inductor (compatível com todas as builds)
try:
    import torch._dynamo.config as dynamo_cfg
    dynamo_cfg.verbose = False
    dynamo_cfg.suppress_errors = True
    dynamo_cfg.log_level = "ERROR"
except Exception:
    pass

try:
    import torch._inductor.config as inductor_cfg
    inductor_cfg.debug = False
    inductor_cfg.compile_threads = 1
    inductor_cfg.triton.cudagraphs = False
    inductor_cfg.max_autotune = True
    inductor_cfg.max_autotune_pointwise = True
    inductor_cfg.max_autotune_gemm_backends = "cublas,triton,aten"  # ✅ inclui fallback ATEN
    inductor_cfg.max_autotune_gemm_mode = "HEURISTIC"  # evita NoValidChoicesError

except Exception as e:
    print(f"⚠️ Patch inductor parcial: {e}")


print("🧩 Patch simbiótico turbo universal ativado | Inductor otimizado e estável")

# =========================================================
# 🔥 Imports principais
# =========================================================
import torch, asyncio, concurrent.futures, gc, queue, threading
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
# ⚙️ Logging assíncrono (não bloqueante)
# =========================================================
log_q = queue.Queue()
def _log_worker():
    while True:
        msg = log_q.get()
        if msg is None:
            break
        print(msg)
threading.Thread(target=_log_worker, daemon=True).start()


# =========================================================
# 🔁 Loop principal simbiótico turbo
# =========================================================
async def main():
    env, modelo, alvo, opt, replay, nbuf, scaler = setup_simbiotico()
    best_global = carregar_patrimonio_global()

    try:
        modelo = torch.compile(modelo, mode="max-autotune-no-cudagraphs", fullgraph=False)
        alvo = torch.compile(alvo, mode="max-autotune-no-cudagraphs", fullgraph=False)
        print("⚙️ torch.compile ativado (modo turbo)")
    except Exception as e:
        print(f"[WARN] torch.compile desativado: {e}")

    EPSILON, lr_now, temp_now, beta_per = 1.0, LR, TEMP_INI, BETA_PER_INI
    ema_q = ema_r = 0.0
    total_steps, episodio = 0, 0
    cooldown_until, rollbacks = 0, 0
    last_loss, last_y_pred, last_good = 0.0, 0.0, None

    print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type} | patrimônio inicial={best_global:.2f}")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    stream = torch.cuda.Stream()

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
            lr_now, EPSILON, temp_now, beta_per, eps_now = update_params(
                total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt
            )

            with torch.cuda.stream(stream):
                s, done, info, total_reward_ep = collect_step(
                    env, modelo, DEVICE, eps_now, replay, nbuf, total_reward_ep
                )
            torch.cuda.current_stream().wait_stream(stream)

            can_train = (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until)
            if can_train:
                with torch.cuda.stream(stream):
                    loss, ema_q, ema_r = train_step(
                        modelo, alvo, opt, replay, scaler, ema_q, ema_r, total_steps
                    )
                last_loss = loss if loss is not None else last_loss * 0.95

                if rollback_guard(loss, total_steps, modelo, alvo, opt, replay, EPSILON):
                    cooldown_until = total_steps + COOLDOWN_STEPS
                    rollbacks += 1
                    log_q.put(f"⚠ Reset simbiótico #{rollbacks} | cooldown até {cooldown_until}")

            best_global = check_vitoria(
                info.get("patrimonio", 0.0),
                best_global,
                modelo,
                opt,
                replay,
                EPSILON,
                total_reward_ep,
            )

            if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
                taxa = aplicar_poda(modelo)
                regenerar_sinapses(modelo, taxa)
                verificar_homeostase(modelo, last_loss)
                log_q.put(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

            if total_steps % HOMEOSTASE_EVERY == 0:
                homeostase_replay(replay)

            if total_steps % PRINT_EVERY == 0:
                energia = float(info.get("energia", 1.0))
                last_y_pred = float(info.get("ret", 0.0))
                msg = log_status(
                    episodio, total_steps,
                    float(info.get("capital", capital)),
                    float(info.get("patrimonio", 0.0)),
                    float(info.get("max_patrimonio", max_patrimonio)),
                    eps_now, temp_now, beta_per, lr_now,
                    energia, last_y_pred, last_loss
                )
                log_q.put(msg)

            if total_steps % SAVE_EVERY == 0 and len(replay) >= MIN_REPLAY:
                torch.cuda.synchronize()
                salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
                log_q.put(f"💾 Checkpoint salvo | step={total_steps}")

        salvar_patrimonio_global(best_global)
        log_q.put(
            f"\n🔁 Episódio {episodio:04d} finalizado | cap_final={info.get('capital', 0):.2f} | best={best_global:.2f}\n"
        )

        if info.get("capital", 0) <= 1.0:
            log_q.put("💀 Falência simbiótica — reiniciando ambiente")
            env.reset()

        if total_steps % 4096 == 0:
            torch.cuda.synchronize()
            gc.collect()


# =========================================================
# 🚀 Execução simbiótica turbo
# =========================================================
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("🚀 EtherSym Turbo Engine — inicializando loop simbiótico")
    asyncio.run(main())
