# policy_v9.py
import numpy as np, torch
ACTIONS = np.array([-1, 0, 1], dtype=np.int8)

def escolher_acao_v9(modelo, estado_np, device, eps,
                     temp_base=0.95, temp_min=0.60,
                     capital=0.0, posicao=0.0, gate_conf=0.35):
    """Política ε-greedy + temperatura."""
    if np.random.rand() < eps:
        return int(np.random.choice(ACTIONS)), 0.5

    x = torch.tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals, y = modelo(x)
        conf = float(torch.tanh(torch.abs(y)).item())
        temp = max(temp_min, temp_base * (0.8 + 0.2 * (eps / max(1e-6, eps))))
        probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).clamp_(1e-6, 1.0)
        probs = (probs / probs.sum()).cpu().numpy()

    a = int(np.random.choice(ACTIONS, p=probs))
    if conf < gate_conf: a = 0
    if a == 1 and capital <= 0: a = 0
    if a == -1 and posicao <= 0: a = 0
    return a, conf
