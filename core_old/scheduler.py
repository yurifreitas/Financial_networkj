import math
from core.utils import set_lr
from core.hyperparams import *


def update_params(total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt):
    """
    Função para atualizar os parâmetros de treinamento como o learning rate (lr),
    o epsilon, a temperatura, o beta e o valor do epsilon para o treino.

    Parâmetros:
    total_steps -- Total de passos de treino realizados
    lr_now -- Taxa de aprendizado atual
    EPSILON -- Valor do epsilon, usado para controle de exploração/exploração
    temp_now -- Temperatura atual para controle de exploração/exploração
    beta_per -- Fator de ajuste do aprendizado (pode ser beta de priorização)
    replay -- Buffer de replay usado para armazenar experiências
    opt -- Otimizador usado no treino (por exemplo, Adam)

    Retorna:
    lr_now -- Nova taxa de aprendizado ajustada
    EPSILON -- Novo valor de epsilon ajustado
    temp_now -- Nova temperatura ajustada
    beta_per -- Novo valor de beta ajustado
    eps_now -- Valor de epsilon ajustado para o ciclo de warmup
    """

    # Determina se o warmup (aquecimento) está ativado, ou seja, antes de 3000 passos
    warmup = total_steps < 3000

    # Ajuste do EPSILON (exploração) baseado em uma taxa de decaimento
    if len(replay) >= MIN_REPLAY:
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    eps_now = 1.0 if warmup else EPSILON  # Durante o warmup, usa epsilon máximo (1.0)

    # Atualização dos parâmetros quando o warmup já passou
    if not warmup:
        temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)  # Decai a temperatura
        beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)  # Ajuste do beta
        replay.beta = float(beta_per)  # Atualiza a priorização do buffer de replay

    # Atualização do learning rate (LR) com warmup e progressão cíclica
    if total_steps <= LR_WARMUP_STEPS:
        lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)  # Linear warmup
    else:
        progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)  # Ajuste progressivo
        lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))  # Decaimento cíclico

    # Atualiza o learning rate no otimizador
    set_lr(opt, lr_now)

    # Retorna os parâmetros atualizados
    return lr_now, EPSILON, temp_now, beta_per, eps_now
