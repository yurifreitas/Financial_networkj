import os, torch, pandas as pd
from torch.amp import GradScaler
from network import criar_modelo
from env import Env, make_feats
from memory import RingReplay, NStepBuffer
from core.config import turbo_cuda, reseed
from core.hyperparams import *
from env.env_parallel import EnvParallel
def setup_simbiotico():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"CSV não encontrado: {CSV}")

    df = pd.read_csv(CSV)
    base, price = make_feats(df)
    # env = Env(base, price)
    env = Env(base, price)
    modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)

    turbo_cuda()
    reseed(SEED)

    replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)
    nbuf = NStepBuffer(N_STEP, GAMMA)
    scaler = GradScaler("cuda", enabled=AMP)
    return env, modelo, alvo, opt, replay, nbuf, scaler
