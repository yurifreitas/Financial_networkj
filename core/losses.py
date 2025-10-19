import torch.nn.functional as F

def loss_q_hibrida(q_pred, q_tgt):
    mse = F.mse_loss(q_pred, q_tgt)
    hub = F.smooth_l1_loss(q_pred, q_tgt, beta=0.8)
    return 0.7 * mse + 0.3 * hub

def loss_regressao(y_pred, y_tgt, l2=1e-4):
    reg = F.smooth_l1_loss(y_pred, y_tgt, beta=0.5)
    return reg + l2 * (y_pred**2).mean()
