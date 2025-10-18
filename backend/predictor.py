import torch, numpy as np
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def prever_tendencia(modelo, df):
    base, price = make_feats(df)
    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)
    q_vals, y_pred = modelo(x)
    action = int(torch.argmax(q_vals).item()) - 1  # {-1,0,1}
    retorno_pred = float(y_pred.item())
    return {
        "preco": float(price[-1]),
        "retorno_pred": retorno_pred,
        "acao_modelo": action
    }
