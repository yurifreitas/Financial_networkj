# =========================================================
# 🔮 EtherSym Finance — backend/markov_predictor.py
# =========================================================
# Gera cenários futuros (bifurcações) a partir da rede principal,
# usando política (softmax sobre Q) e retorno contínuo da cabeça Y.
# Retorna DataFrame com colunas mínimas para uso no simulador.
# =========================================================

import numpy as np
import pandas as pd
import torch
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rede_markoviana(modelo, df, profundidade=5, bifurcacoes=3, temp=0.7,
                    ruido_scale=0.25, faixa_clip=0.10):
    """
    🌌 Rede Markoviana Simbiótica
    - Propaga caminhos futuros por 'profundidade' passos
    - Em cada passo escolhe 'bifurcacoes' ações mais prováveis via softmax(Q/temp)
    - Preço futuro é simulado a partir do retorno contínuo previsto (y_pred),
      condicionado ao sinal da ação escolhida (-1, 0, +1) e ruído gaussiano.

    Retorna um DataFrame com:
      t, preco, ret, prob, data_futura, cenario
    (prob normalizada por passo t; preço é hard-clipped ±faixa_clip em torno do último preço)
    """
    # =============================
    # 🧩 Estado atual e features
    # =============================
    base, price = make_feats(df)
    ultimo_preco = float(price[-1])

    # vetor de estado (concat com placeholders)
    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)

    caminhos = [{"t": 0, "preco": ultimo_preco, "ret": 0.0, "prob": 1.0}]
    previsoes = []

    acao_space = np.array([-1, 0, 1], dtype=np.int8)

    # =============================
    # 🔁 Propagação simbiótica
    # =============================
    for t in range(profundidade):
        novos = []

        # para cada caminho corrente, bifurcar pelas ações mais prováveis
        q_vals, y_pred = modelo(x)  # q_vals: (1,3), y_pred: (1,1)
        probs = torch.softmax(q_vals / max(temp, 1e-3), dim=1).squeeze(0).cpu().numpy()
        top_idx = np.argsort(probs)[-bifurcacoes:]

        for c in caminhos:
            for i in top_idx:
                a = int(acao_space[i])
                conf = float(probs[i])

                # retorno contínuo previsto condicionado ao sinal da ação
                ret_pred = float(y_pred.item()) * float(a)

                # ruído proporcional ao módulo da previsão (mais vol => mais ruído)
                ruido = np.random.normal(0.0, abs(ret_pred) * ruido_scale + 1e-6)
                ret_sim = ret_pred + ruido

                # clipe de preço (proteção a outliers)
                preco_fut = c["preco"] * (1.0 + ret_sim)
                low, high = ultimo_preco * (1.0 - faixa_clip), ultimo_preco * (1.0 + faixa_clip)
                preco_fut = float(np.clip(preco_fut, low, high))

                novos.append({
                    "t": int(c["t"] + 1),
                    "preco": preco_fut,
                    "ret": float(ret_sim),
                    "prob": float(c["prob"] * conf),
                })

        # normaliza prob por passo (evita drift numérico)
        total_prob = sum(n["prob"] for n in novos) or 1.0
        for n in novos:
            n["prob"] = float(n["prob"] / total_prob)

        previsoes.extend(novos)
        caminhos = novos

    df_prev = pd.DataFrame(previsoes)
    if df_prev.empty:
        # garante estrutura mínima (evita KeyError no chamador)
        return pd.DataFrame(columns=["t", "preco", "ret", "prob", "data_futura", "cenario"])

    # marca tempo futuro (assumindo 1 passo = 1h; ajuste se necessário)
    base_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    df_prev["data_futura"] = base_ts + pd.to_timedelta(df_prev["t"], unit="h")

    # classificação de cenário por quantis de retorno
    try:
        q25, q75 = np.quantile(df_prev["ret"], [0.25, 0.75])
    except Exception:
        q25, q75 = -1e-9, 1e-9

    conds = [
        (df_prev["ret"] >= q75),
        (df_prev["ret"] <= q25),
        (df_prev["ret"].between(q25, q75)),
    ]
    labels = ["otimista", "pessimista", "neutro"]
    df_prev["cenario"] = np.select(conds, labels, default="neutro")

    # garante colunas mínimas
    for col in ["t", "preco", "ret", "prob", "data_futura", "cenario"]:
        if col not in df_prev.columns:
            df_prev[col] = np.nan

    return df_prev[["t", "preco", "ret", "prob", "data_futura", "cenario"]]
