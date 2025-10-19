import time

def log_status(episodio, total_steps, capital, patrimonio, max_patrimonio,
               eps_now, temp_now, beta_per, lr_now, energia, y_pred, loss):
    print(
        f"[Ep {episodio:04d} | {total_steps:>8}] "
        f"cap={capital:>9.2f} | pat={patrimonio:>9.2f} | max={max_patrimonio:>9.2f} | "
        f"ε={eps_now:.3f} | τ={temp_now:.2f} | β={beta_per:.2f} | "
        f"lr={lr_now:.6f} | enr={energia:.2f} | Δpred={y_pred:+.4f} | loss={loss:.5f}"
    )
