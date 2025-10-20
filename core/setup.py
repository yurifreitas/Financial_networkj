import os, pandas as pd
from torch.amp import GradScaler
from core.network import criar_modelo
from env import Env, make_feats
from core.config import reseed
from core.hyperparams import *
from core.memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
from core.patrimonio import carregar_patrimonio_global, salvar_patrimonio_global
from core.utils import set_lr


# =========================================================
# Função de configuração do ambiente e inicialização
# =========================================================
def setup_simbiotico():
    # Verifica se o arquivo CSV existe
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"O arquivo CSV não foi encontrado: {CSV}")

    # Carrega os dados
    df = pd.read_csv(CSV)
    base, price = make_feats(df)

    # Inicializa o ambiente
    env = Env(base, price)

    # Cria o modelo, alvo e o otimizador
    modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)

    # Reseta as sementes para aleatoriedade
    reseed(SEED)

    # Inicializa o replay buffer com o estado correto
    replay = RingReplay(state_dim=base.shape[1] + 2, capacity=MEMORIA_MAX, device=DEVICE)

    # Inicializa o buffer N-step para aprendizado temporal
    nbuf = NStepBuffer(N_STEP, GAMMA)

    # Inicializa o GradScaler para AMP (Automatic Mixed Precision)
    scaler = GradScaler("cuda", enabled=AMP)

    # Configura a taxa de aprendizado
    lr_now = LR
    set_lr(opt, lr_now)

    # Carrega o estado salvo se houver
    _, EPSILON_SAVED, _ = carregar_estado(modelo, opt)

    # Define o valor de epsilon, se salvo, senão usa o valor inicial
    EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL

    print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type}")

    # Carrega o patrimônio global
    best_global = carregar_patrimonio_global()
    print(f"🏁 Patrimônio global inicial carregado: {best_global:.2f}")

    return env, modelo, alvo, opt, replay, nbuf, scaler, EPSILON
