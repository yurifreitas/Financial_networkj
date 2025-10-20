import torch, math, logging
from torch.amp import autocast, GradScaler
from core.utils import soft_update, is_bad_number
from core.losses import loss_q_hibrida, loss_regressao
from core.hyperparams import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =========================================================
# ♻ Sanitização simbiótica segura
# =========================================================
def _sanitize_safe(t, minv=-1e9, maxv=1e9):
    if t is None:
        return None
    return torch.clamp(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), minv, maxv)

# =========================================================
# 🧬 Treinamento simbiótico principal
# =========================================================
def train_step(modelo, alvo, opt, replay, scaler, ema_q, ema_r):
    # Sample mini-batch from replay buffer
    (s_t, a_t, r_t, sn_t, f_t, idx, w, y_ret_t) = replay.sample(BATCH)

    # =====================================================
    # 🔮 Alvo Q (Double DQN)
    # =====================================================
    with torch.no_grad():
        next_q_online, _ = modelo(sn_t)
        next_q_target, _ = alvo(sn_t)
        next_q_online = _sanitize_safe(next_q_online, -Q_CLAMP, Q_CLAMP)
        next_q_target = _sanitize_safe(next_q_target, -Q_CLAMP, Q_CLAMP)

        next_a = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_best = next_q_target.gather(1, next_a).squeeze(1)
        alvo_q = r_t + (GAMMA ** N_STEP) * next_best * (1.0 - f_t)
        alvo_q = _sanitize_safe(alvo_q, -Q_TARGET_CLAMP, Q_TARGET_CLAMP)

    # =====================================================
    # ⚙️ Forward simbiótico
    # =====================================================
    opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=AMP):
        q_vals, y_pred = modelo(s_t)
        q_vals = _sanitize_safe(q_vals, -Q_CLAMP, Q_CLAMP)

        q_sel = q_vals.gather(1, a_t).squeeze(1)
        y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP

        # Função de perda híbrida
        loss_q = loss_q_hibrida(q_sel, alvo_q)
        loss_reg = loss_regressao(y_pred, y_target)

        # Médias móveis
        ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
        ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())
        λ = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))

        # Perda total
        loss_total = loss_q + λ * loss_reg

        # Suavização do gradiente no início (warmup)
        if total_steps < 4096:
            warm = total_steps / 4096.0
            loss_total = loss_total * warm

        # Amortecimento exponencial
        loss_total = loss_total / (1.0 + torch.exp(-loss_total / 25.0))

    # =====================================================
    # 🚫 Proteção contra perda anômala
    # =====================================================
    if (not torch.isfinite(loss_total)
        or is_bad_number(loss_total)
        or math.isnan(loss_total.item())
        or abs(loss_total.item()) > 100.0):

        logging.warning(f"⚠️ Loss simbiótico anômalo ({loss_total.item():.4f}) → rollback local")

        # Rollback adaptativo
        for g in opt.param_groups:
            g["lr"] = max(LR_MIN, g["lr"] * 0.7)

        # Reset GradScaler
        scaler = GradScaler("cuda", enabled=AMP)
        ema_q *= 0.9
        torch.cuda.empty_cache()
        return None, ema_q, ema_r

    # =====================================================
    # 💪 Backprop + GradScaler
    # =====================================================
    scaler.scale(loss_total).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)

    try:
        scaler.step(opt)
        scaler.update()
    except Exception as e:
        logging.warning(f"GradScaler falhou ({e}) → reset simbiótico")
        scaler = GradScaler("cuda", enabled=AMP)
        opt.zero_grad(set_to_none=True)
        return None, ema_q, ema_r


    with torch.no_grad():
        delta_loss = abs(loss_total.item() - ema_q)
        energia_t = math.exp(-delta_loss / 10.0)
        energia_prev = globals().get("energia_prev", energia_t)
        energia_suave = 0.9 * energia_prev + 0.1 * energia_t
        globals()["energia_prev"] = energia_suave

        # Ajuste da taxa de aprendizado com base na energia
        lr_base = opt.param_groups[0]["lr"]
        lr_novo = max(LR_MIN, lr_base * (0.85 + 0.25 * energia_suave))
        for g in opt.param_groups:
            g["lr"] = lr_novo

    with torch.no_grad():
        td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0)
        replay.update_priority(idx, td_error.detach().cpu())
        tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, float(loss_q.item()))))
        soft_update(alvo, modelo, tau)

    # =====================================================
    # 🧩 Retorno simbiótico
    # =====================================================
    return float(loss_total.item()), ema_q, ema_r
