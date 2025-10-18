# =========================================================
# 🧠 EtherSym Finance — backend/predictor.py
# =========================================================
# Wrapper de inferência do modelo principal:
# - Gera estado com make_feats
# - Produz q_vals (discreto) e y_pred (contínuo)
# - Retorna dict com preco atual, retorno_pred e acao_modelo
# =========================================================

import numpy as np
import torch
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def prever_tendencia(modelo, df):
    """
    Retorna:
      - preco: float (último preço do DF)
      - retorno_pred: float (predição contínua da cabeça de regressão)
      - acao_modelo: int {-1,0,1} (argmax de Q)
      - energia: float opcional (se o modelo expuser; aqui default=1.0)
    """
    base, price = make_feats(df)  # base: np.ndarray (T x feat)
    # placeholder simbiótico adicional (mantém compatibilidade com seu estado)
    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)

    x = torch.tensor(s, device=DEVICE).unsqueeze(0)  # (1, state_dim)
    q_vals, y_pred = modelo(x)                       # q_vals: (1,3), y_pred: (1,1)

    # ação {-1,0,1} pelo argmax de Q
    action = int(torch.argmax(q_vals, dim=1).item()) - 1
    retorno_pred = float(y_pred.item())

    # se seu modelo expõe 'energia' via atributo/buffer, injete aqui
    energia = 1.0

    return {
        "preco": float(price[-1]),
        "retorno_pred": retorno_pred,
        "acao_modelo": action,
        "energia": float(energia),
    }
