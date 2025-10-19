import time, math, sys

def log_status(episodio, total_steps, capital, patrimonio, max_patrimonio,
               eps_now, temp_now, beta_per, lr_now, energia, y_pred, loss):
    """
    Logging simbiótico seguro — evita travas de stdout e NaN em métricas.
    """
    # 🔒 Sanitização numérica segura
    def _safe(v, fmt="%.5f", default="0.00000"):
        try:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return default
            return fmt % v
        except Exception:
            return default

    # Valores formatados
    cap_str   = f"{capital:>9.2f}"
    pat_str   = f"{patrimonio:>9.2f}"
    max_str   = f"{max_patrimonio:>9.2f}"
    eps_str   = _safe(eps_now, "%.3f")
    tau_str   = _safe(temp_now, "%.2f")
    beta_str  = _safe(beta_per, "%.2f")
    lr_str    = _safe(lr_now, "%.6f")
    enr_str   = _safe(energia, "%.2f")
    ypred_str = _safe(y_pred, "%+.4f")
    loss_str  = _safe(loss, "%.5f")

    # 🧠 Impressão simbiótica otimizada (sem flush travado)
    msg = (
        f"[Ep {episodio:04d} | {total_steps:>8}] "
        f"cap={cap_str} | pat={pat_str} | max={max_str} | "
        f"ε={eps_str} | τ={tau_str} | β={beta_str} | "
        f"lr={lr_str} | enr={enr_str} | Δpred={ypred_str} | loss={loss_str}"
    )
    print(msg, flush=False)
