# =========================================================
# 🌌 EtherSym Finance — core/env_parallel.py
# =========================================================
# Paraleliza o ambiente simbiótico real (core_env.Env)
# Cada worker roda uma cópia independente do ambiente
# =========================================================

import torch.multiprocessing as mp
import numpy as np
from env.core_env import Env

def worker_env(pipe, base, price):
    """Worker simbiótico: executa um ambiente isolado."""
    env = Env(base.copy(), price.copy())
    s = env.reset()
    pipe.send(("RESET_OK", s))

    while True:
        msg = pipe.recv()
        if msg == "STOP":
            break

        if isinstance(msg, dict):
            if msg.get("reset"):
                s = env.reset()
                pipe.send(("RESET", s))
            elif "action" in msg:
                a = msg["action"]
                s_next, r, done, info = env.step(a)
                pipe.send(("STEP", (s_next, r, done, info)))
            else:
                pipe.send(("ERR", "Comando inválido"))
        else:
            pipe.send(("ERR", "Mensagem desconhecida"))

    pipe.close()


class EnvPool:
    """Gerenciador simbiótico de ambientes paralelos."""
    def __init__(self, n_envs, base, price):
        self.n_envs = n_envs
        self.pipes, self.procs = [], []
        mp.set_start_method("spawn", force=True)
        for i in range(n_envs):
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(target=worker_env, args=(child_conn, base, price))
            proc.daemon = True
            proc.start()
            self.pipes.append(parent_conn)
            self.procs.append(proc)

        # espera inicialização dos workers
        self.states = []
        for pipe in self.pipes:
            tag, s = pipe.recv()
            assert tag.startswith("RESET"), f"Erro init: {tag}"
            self.states.append(s)

    def reset_all(self):
        for pipe in self.pipes:
            pipe.send({"reset": True})
        self.states = [pipe.recv()[1] for pipe in self.pipes]
        return self.states

    def step_all(self, actions):
        for pipe, a in zip(self.pipes, actions):
            pipe.send({"action": int(a)})
        results = [pipe.recv()[1] for pipe in self.pipes]
        sps, rs, dones, infos = zip(*results)
        self.states = list(sps)
        return self.states, rs, dones, infos

    def close(self):
        for pipe in self.pipes:
            pipe.send("STOP")
        for p in self.procs:
            p.join()
