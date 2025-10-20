import os, pandas as pd
from torch.amp import GradScaler
from core.network import criar_modelo
from env import Env, make_feats
from core.memory import NStepBuffer,carregar_estado
from core.config import reseed
from core.hyperparams import *
def setup_simbiotico():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"CSV não encontrado: {CSV}")

    df = pd.read_csv(CSV)
    base, price = make_feats(df)
    # env = Env(base, price)
    env = Env(base, price)
    modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)

    # turbo_cuda()
    reseed(SEED)
    # Ao carregar o estado
    replay, EPSILON_SAVED, media_saved = carregar_estado(modelo, opt, path=SAVE_PATH)
    EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL
    print(f"♻️ Estado carregado | ε={EPSILON:.3f} | média={media_saved:.3f}")

    nbuf = NStepBuffer(N_STEP, GAMMA)
    scaler = GradScaler("cuda", enabled=AMP)
    return env, modelo, alvo, opt, replay, nbuf, scaler, EPSILON
