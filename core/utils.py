import torch, numpy as np, torch.nn.functional as F

@torch.no_grad()
def soft_update(target, online, tau=0.005):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)



def escolher_acao(modelo, estado_np, device, eps,
                  temp_base=0.95, temp_min=0.60,
                  capital=0.0, posicao=0.0, gate_conf=0.35):
    ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
    if np.random.rand() < eps:
        return int(np.random.choice(ACTIONS)), 0.5
    x = torch.tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals, y = modelo(x)
        # confiança suave (sem saturar em 1.0 tão cedo)
        conf = float(torch.sigmoid(torch.abs(y)).item())
        # temperatura vinculada ao epsilon
        temp = max(temp_min, temp_base * (0.8 + 0.2 * (eps / max(1e-6, eps))))
        probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).clamp_(1e-6, 1.0)
        probs = (probs / probs.sum()).cpu().numpy()
    a = int(np.random.choice(ACTIONS, p=probs))
    if conf < gate_conf: a = 0
    if a == 1 and capital <= 0: a = 0
    if a == -1 and posicao <= 0: a = 0
    return a, conf

def set_lr(optim, lr):
    for g in optim.param_groups:
        g.update(lr=lr)

def is_bad_number(x):
    return torch.isnan(x).any() or torch.isinf(x).any()
def a_to_idx(a: int): return int(a + 1)