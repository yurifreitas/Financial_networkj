import torch
from torch.amp import autocast
from core.utils import soft_update
from core.losses import loss_q_hibrida, loss_regressao
from core.hyperparams import *

def train_step(modelo, alvo, opt, replay, scaler, ema_q, ema_r, total_steps):
    (s_t, a_t, r_t, sn_t, f_t, idx, w, y_ret_t) = replay.sample(BATCH)

    with torch.no_grad():
        next_q_online, _ = modelo(sn_t)
        next_q_target, _ = alvo(sn_t)
        next_q_online = next_q_online.clamp_(-Q_CLAMP, Q_CLAMP)
        next_q_target = next_q_target.clamp_(-Q_CLAMP, Q_CLAMP)
        next_a = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_best = next_q_target.gather(1, next_a).squeeze(1)
        alvo_q = r_t + (GAMMA ** N_STEP) * next_best * (1.0 - f_t)
        alvo_q = alvo_q.clamp_(-Q_TARGET_CLAMP, Q_TARGET_CLAMP)

    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=AMP):
        q_vals, y_pred = modelo(s_t)
        q_sel = q_vals.gather(1, a_t).squeeze(1)
        y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP

        loss_q = loss_q_hibrida(q_sel, alvo_q)
        do_reg = total_steps >= REG_FREEZE_STEPS
        loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)

        ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
        ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())
        λ = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
        loss_total = loss_q + λ * loss_reg

    scaler.scale(loss_total).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
    scaler.step(opt)
    scaler.update()

    with torch.no_grad():
        td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).cpu().numpy()
        replay.update_priority(idx, td_error)
        tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, loss_q.item())))
        soft_update(alvo, modelo, tau)

    return float(loss_total.item()), ema_q, ema_r
