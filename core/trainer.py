# =========================================================
# 🧠 EtherSym Finance — core/trainer.py (v8g_energy_stable)
# =========================================================
# - Mantém grafo do autograd intacto
# - Corrige RuntimeError: "tensor does not require grad"
# - Proteção anti-NaN + rollback simbiótico + AMP
# - Logging estruturado e energia simbiótica adaptativa
# =========================================================

import torch, math, logging
from torch.amp import autocast, GradScaler
from core.utils import soft_update, is_bad_number
from core.losses import loss_q_hibrida, loss_regressao
from core.hyperparams import *

# =========================================================
# 🪶 Configuração de Logging Simbiótico
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =========================================================
# ♻ Sanitização simbiótica segura (preserva grad_fn)
# =========================================================
def _sanitize_safe(t, minv=-1e9, maxv=1e9):
    if t is None:
        return None
    return torch.clamp(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), minv, maxv)

# =========================================================
# 🧬 Treinamento simbiótico principal
# =========================================================
def train_step(modelo, alvo, opt, replay, scaler, ema_q, ema_r, total_steps):
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
        y_target = y_ret_t.clamp(-Y_CLAMP, Y_CLAMP) / Y_CLAMP

        loss_q = loss_q_hibrida(q_sel, alvo_q)
        do_reg = total_steps >= REG_FREEZE_STEPS
        loss_reg = loss_regressao(y_pred, y_target) if do_reg else torch.zeros_like(loss_q)

        ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
        ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())

        λ = LAMBDA_REG_BASE * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
        loss_total = loss_q + λ * loss_reg

    # =====================================================
    # 🚫 Proteção simbiótica contra explosão
    # =====================================================
    if (not torch.isfinite(loss_total)
        or is_bad_number(loss_total)
        or math.isnan(loss_total.item())
        or abs(loss_total.item()) > 80.0):
        logging.warning(f"Loss simbiótico anômalo ({loss_total.item():.4f}) → rollback local")
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None, ema_q, ema_r

    # =====================================================
    # 💪 Backprop + grad clipping + scaler seguro
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

    # =====================================================
    # 🌀 Energia simbiótica adaptativa (auto-LR)
    # =====================================================
    with torch.no_grad():
        delta_loss = abs(loss_total.item() - ema_q)
        energia = math.exp(-delta_loss / 5.0)  # decai suavemente em picos

        for g in opt.param_groups:
            g["lr"] = max(1e-6, g["lr"] * (0.9 + 0.2 * energia))

        if total_steps % 1000 == 0:
            logging.info(f"Energia simbiótica adaptada | enr={energia:.2f} | Δloss={delta_loss:.3f}")

    # =====================================================
    # 🔁 Atualiza PER e alvo Q
    # =====================================================
    with torch.no_grad():
        td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).cpu().numpy()
        replay.update_priority(idx, td_error)

        tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, float(loss_q.item()))))
        soft_update(alvo, modelo, tau)

        if total_steps % 512 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logging.info(f"Sincronização simbiótica concluída | step={total_steps}")

    # =====================================================
    # 🧩 Retorno simbiótico
    # =====================================================
    logging.debug(
        f"Step={total_steps} | loss={loss_total.item():.5f} | "
        f"ema_q={ema_q:.5f} | ema_r={ema_r:.5f}"
    )

    return float(loss_total.item()), ema_q, ema_r
