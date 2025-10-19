# =========================================================
# 🌌 EtherSym Finance — env/parallel_env.py (Hybrid Multiprocess)
# =========================================================
# Executa múltiplos ambientes simbióticos em paralelo via
# multiprocessing.Pipe, mantendo compatibilidade total com Env.
# =========================================================

import multiprocessing as mp
import numpy as np
from .core_env import Env   # <-- import relativo correto


def _worker_process(base, price, conn, seed):
    """
    Processo individual que executa um ambiente EtherSym.
    """
    np.random.seed(seed)
    env = Env(base, price)
    s = env.reset()

    while True:
        try:
            cmd, data = conn.recv()

            if cmd == "step":
                a = int(data)
                sp, r, done, info = env.step_once(a)
                conn.send((sp, r, done, info))
                if done:
                    s = env.reset()

            elif cmd == "reset":
                s = env.reset()
                conn.send(s)

            elif cmd == "close":
                conn.close()
                break

            else:
                print(f"[WARN] Comando desconhecido: {cmd}")

        except EOFError:
            break
        except Exception as e:
            print(f"[ERROR] Worker exception: {e}")
            break


class ParallelEnv:
    """
    Gerencia N ambientes EtherSym rodando em paralelo.
    Cada ambiente executa em seu próprio processo.
    """
    def __init__(self, base, price, n_envs=8):
        self.n_envs = n_envs
        self.parent_pipes = []
        self.workers = []

        for i in range(n_envs):
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=_worker_process,
                args=(base, price, child_conn, 1234 + i),
                daemon=True
            )
            proc.start()
            self.parent_pipes.append(parent_conn)
            self.workers.append(proc)

        # reset inicial
        for p in self.parent_pipes:
            p.send(("reset", None))
        self.states = [p.recv() for p in self.parent_pipes]
        print(f"🚀 ParallelEnv inicializado com {n_envs} ambientes simbióticos")

    def step(self, actions):
        """
        Executa um passo em todos os ambientes simultaneamente.
        """
        assert len(actions) == self.n_envs, "Número de ações ≠ número de envs"
        for p, a in zip(self.parent_pipes, actions):
            p.send(("step", int(a)))

        results = [p.recv() for p in self.parent_pipes]
        next_states, rewards, dones, infos = zip(*results)
        self.states = list(next_states)
        return (
            np.stack(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def reset_done(self, dones):
        """
        Reseta apenas os ambientes que terminaram.
        """
        for i, done in enumerate(dones):
            if done:
                self.parent_pipes[i].send(("reset", None))
                self.states[i] = self.parent_pipes[i].recv()
        return np.stack(self.states)

    def close(self):
        """
        Fecha todos os processos filhos com segurança.
        """
        for p in self.parent_pipes:
            p.send(("close", None))
        for w in self.workers:
            w.join(timeout=1)
        print("🧩 ParallelEnv encerrado com sucesso.")
