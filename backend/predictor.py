# =========================================================
# 🔮 EtherSym Finance — backend/predictor.py
# =========================================================
import torch, numpy as np
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def prever_tendencia(modelo, df):
    base, price = make_feats(df)
    if len(base) == 0 or len(price) == 0:
        raise ValueError("Dados insuficientes para previsão")

    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)

    q_vals, y_pred = modelo(x)
    action = int(torch.argmax(q_vals).item()) - 1
    retorno_pred = float(y_pred.item())
    preco_atual = float(price[-1])
    preco_previsto = preco_atual * (1.0 + retorno_pred)

    return {
        "preco": preco_atual,
        "preco_previsto": preco_previsto,
        "retorno_pred": retorno_pred,
        "acao_modelo": action,
        "energia": 1.0,
    }
