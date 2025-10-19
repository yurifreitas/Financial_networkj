import math
from core.utils import set_lr
from core.hyperparams import *

def update_params(total_steps, lr_now, EPSILON, temp_now, beta_per, replay, opt):
    warmup = total_steps < 3000

    if len(replay) >= MIN_REPLAY:
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    eps_now = 1.0 if warmup else EPSILON

    if not warmup:
        temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)
        beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)
        replay.beta = float(beta_per)

    if total_steps <= LR_WARMUP_STEPS:
        lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
    else:
        progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / 1_000_000)
        lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
    set_lr(opt, lr_now)

    return lr_now, EPSILON, temp_now, beta_per, eps_now
