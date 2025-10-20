try:
    import torch._logging as _logging
    _logging.set_logs()  # limpa logs simbióticos
except Exception:
    pass


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

except Exception as e:
    print(f"⚠️ Patch inductor parcial: {e}")

import asyncio, gc, queue, threading, multiprocessing as mp
from core.setup import setup_simbiotico
from core.scheduler import update_params
from core.collector import collect_step
from core.trainer import train_step
from core.checkpoints import check_vitoria, rollback_guard
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.maintenance import aplicar_poda, regenerar_sinapses, verificar_homeostase, homeostase_replay
from core.logger import log_status
from core.hyperparams import *
from core.memory import salvar_estado

import os, warnings

# =========================================================
# 🔐 Patch simbiótico estável
# =========================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.update({
    "TORCH_COMPILE_DEBUG": "0",
    "TORCHINDUCTOR_DISABLE_FX_VALIDATION": "1",
    "TORCHINDUCTOR_FUSE_TRIVIAL_OPS": "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "cublas,triton",
    "TORCHINDUCTOR_CACHE_DIR": os.path.expanduser("~/.cache/torch/inductor"),
})

# =========================================================
# ⚙️ Logging assíncrono
# =========================================================
log_q = queue.Queue()
def _log_worker():
    while True:
        msg = log_q.get()
        if msg is None: break
        print(msg)
threading.Thread(target=_log_worker, daemon=True).start()

# Worker do ambiente
def env_worker(rank, conn):
    """Executa ambiente independente e envia transições"""
    env, modelo, alvo, opt, replay, nbuf, scaler, EPSILON = setup_simbiotico()

    torch.set_num_threads(1)
    total_reward_ep = 0.0  # Acumulador de recompensa do episódio
    while True:
        try:
            # Verifica se há dados para processar
            if conn.poll():
                cmd, payload = conn.recv()
                if cmd == "reset":
                    env.reset()
                    total_reward_ep = 0.0  # Resetando a recompensa no reinício
                    continue

            # Coleta os dados do ambiente
            s, done, info, r = collect_step(env, modelo, DEVICE, EPSILON, replay, nbuf, total_reward_ep)

            # Atualiza o total de recompensa
            total_reward_ep += r  # Acumula a recompensa

            conn.send((s, done, info, total_reward_ep))

        except EOFError:
            # Se a conexão for fechada, encerra o loop
            print(f"[Worker {rank}] Conexão encerrada.")
            break
        except Exception as e:
            # Caso algum outro erro ocorra, registra e tenta continuar
            print(f"[Worker {rank}] Erro: {e}")
            conn.send(("error", str(e)))
            break

# Função principal do treino
async def main():
    env, modelo, alvo, opt, replay, nbuf, scaler, EPSILON = setup_simbiotico()

    best_global = carregar_patrimonio_global()

    modelo = torch.compile(modelo, mode="max-autotune-no-cudagraphs", fullgraph=False)
    alvo = torch.compile(alvo, mode="max-autotune-no-cudagraphs", fullgraph=False)
    print(f"⚙️ torch.compile ativo | device={DEVICE.type}")

    # 🌐 Inicia múltiplos ambientes paralelos
    N_ENVS = min(4, os.cpu_count() // 2)
    ctx = mp.get_context("spawn")
    pipes, workers = [], []
    for i in range(N_ENVS):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=env_worker, args=(i, child))
        p.daemon = True
        p.start()
        pipes.append(parent)
        workers.append(p)
    print(f"🧩 {N_ENVS} ambientes simbióticos em paralelo inicializados")

    # 🔄 Streams CUDA independentes
    stream_train = torch.cuda.Stream()
    stream_sync = torch.cuda.Stream()

    EPSILON, lr_now, temp_now, beta_per = 1.0, LR, TEMP_INI, BETA_PER_INI
    ema_q = ema_r = 0.0
    total_steps, episodio = 0, 0
    cooldown_until, rollbacks = 0, 0
    last_loss = 0.0
    melhor_patrimonio_ep = 0.0

    info = {"capital": CAPITAL_INICIAL, "patrimonio": CAPITAL_INICIAL, "max_patrimonio": CAPITAL_INICIAL, "energia": 1.0, "ret": 0.0}

    print(f"🧠 Iniciando treino simbiótico | patrimônio inicial={best_global:.2f}")

    while True:
        total_steps += 1

        # 🔹 Recebe amostras dos envs paralelos
        for parent in pipes:
            if parent.poll():
                data = parent.recv()
                if isinstance(data, tuple) and not (isinstance(data[0], str) and data[0] == "error"):
                    s, done, info_recv, total_reward_ep = data
                    info.update(info_recv)
                    if replay is not None:
                        try:
                            # Atualiza o replay buffer corretamente
                            replay.push(s)  # Garante que as transições sejam armazenadas
                        except Exception:
                            pass
                    # 🔁 Finaliza e reinicia episódios
                    if done or float(info.get("capital", 0.0)) <= 1.0:
                        episodio += 1
                        melhor_patrimonio_ep = float(info.get("max_patrimonio", best_global))
                        total_reward_ep = float(info.get("reward", 0.0))
                        try:
                            best_global = check_vitoria(
                                melhor_patrimonio_ep,
                                best_global,
                                modelo,
                                opt,
                                replay,
                                EPSILON,
                                total_reward_ep
                            )
                            salvar_patrimonio_global(best_global)
                            log_q.put(f"🏁 Episódio {episodio:04d} finalizado | melhor={melhor_patrimonio_ep:.2f} | global={best_global:.2f}")
                        except Exception as e:
                            log_q.put(f"⚠️ Erro em check_vitoria(): {e}")
                        for pipe in pipes:
                            try:
                                pipe.send(("reset", None))
                            except:
                                pass
                        torch.cuda.synchronize()
                        await asyncio.sleep(0.05)

                elif isinstance(data, tuple) and data[0] == "error":
                    log_q.put(f"⚠️ Erro em worker: {data[1]}")

        # 🔹 Atualiza hiperparâmetros
        lr_now, EPSILON, temp_now, beta_per, eps_now = update_params(
            total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt
        )

        # 🔹 Treino simbiótico (GPU)
        if (len(replay) >= MIN_REPLAY) and (total_steps >= cooldown_until):
            with torch.cuda.stream(stream_train):
                loss, ema_q, ema_r = train_step(
                    modelo, alvo, opt, replay, scaler, ema_q, ema_r
                )
            last_loss = loss if loss is not None else last_loss * 0.95
            torch.cuda.current_stream().wait_stream(stream_train)

            if rollback_guard(loss, total_steps, modelo, alvo, opt, replay, EPSILON):
                cooldown_until = total_steps + COOLDOWN_STEPS
                rollbacks += 1
                log_q.put(f"⚠ Reset simbiótico #{rollbacks}")

        # 🔹 Ciclos de poda e homeostase
        print("Replay Buffer Length: ", len(replay))  # Imprime o tamanho do replay buffer
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = aplicar_poda(modelo)
            regenerar_sinapses(modelo, taxa)
            verificar_homeostase(modelo, last_loss)
            log_q.put(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}%")

        if total_steps % HOMEOSTASE_EVERY == 0:
            homeostase_replay(replay)

        # 🔹 Logging seguro
        if total_steps % PRINT_EVERY == 0 and info is not None:
            msg = log_status(
                episodio, total_steps,
                float(info.get("capital", 0.0)),
                float(info.get("patrimonio", 0.0)),
                float(info.get("max_patrimonio", 0.0)),
                eps_now, temp_now, beta_per, lr_now,
                float(info.get("energia", 1.0)),
                float(info.get("ret", 0.0)),
                last_loss
            )
            log_q.put(msg)

        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

            torch.cuda.synchronize()
            salvar_estado(modelo, opt, replay, EPSILON, total_steps)
            salvar_patrimonio_global(best_global)
            log_q.put(f"💾 Checkpoint salvo | step={total_steps}")

        # 🔹 Limpeza periódica
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

    print("🚀 EtherSym Turbo Engine — paralelismo total ativado")
    asyncio.run(main())
