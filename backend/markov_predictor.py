import torch
import numpy as np
import pandas as pd
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rede_markoviana(modelo, df, profundidade=8, bifurcacoes=3, temp=0.7):
    """
    🌌 Rede Markoviana Simbiótica
    -------------------------------------
    Gera previsões futuras do preço com base na rede neural principal,
    criando bifurcações ponderadas por probabilidade simbiótica.

    Cada passo de previsão propaga as possíveis trajetórias futuras
    do preço, classificadas em cenários otimista, neutro e pessimista.
    """

    # =============================
    # 🧩 Estado atual e features
    # =============================
    base, price = make_feats(df)
    ultimo_preco = float(price[-1])
    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)

    caminhos = [{"t": 0, "preco": ultimo_preco, "ret": 0.0, "prob": 1.0}]
    previsoes = []

    # =============================
    # 🔁 Propagação simbiótica
    # =============================
    for t in range(profundidade):
        novos_caminhos = []

        for c in caminhos:
            q_vals, y_pred = modelo(x)

            acao_space = np.array([-1, 0, 1])
            probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).cpu().numpy()
            top_idx = np.argsort(probs)[-bifurcacoes:]

            for i in top_idx:
                a = acao_space[i]
                conf = probs[i]
                ret_pred = float(y_pred.item()) * a
                ruido = np.random.normal(0, abs(ret_pred) * 0.25 + 1e-6)
                ret_sim = ret_pred + ruido

                preco_fut = c["preco"] * (1 + ret_sim)
                preco_fut = np.clip(preco_fut, ultimo_preco * 0.9, ultimo_preco * 1.1)

                novos_caminhos.append({
                    "t": c["t"] + 1,
                    "preco": preco_fut,
                    "ret": ret_sim,
                    "prob": c["prob"] * conf,
                })

        total_prob = sum(n["prob"] for n in novos_caminhos) + 1e-9
        for n in novos_caminhos:
            n["prob"] /= total_prob

        previsoes.extend(novos_caminhos)
        caminhos = novos_caminhos

    # =============================
    # 🧠 Estruturação dos dados
    # =============================
    df_prev = pd.DataFrame(previsoes)
    if df_prev.empty:
        raise ValueError("Rede Markoviana gerou um DataFrame vazio.")

    df_prev["data_futura"] = (
        pd.to_datetime(df["timestamp"].iloc[-1])
        + pd.to_timedelta(df_prev["t"], unit="h")
    )

    # =============================
    # 🧭 Classificação de cenários
    # =============================
    q25, q75 = np.quantile(df_prev["ret"], [0.25, 0.75])
    condicoes = [
        (df_prev["ret"] >= q75),
        (df_prev["ret"] <= q25),
        (df_prev["ret"].between(q25, q75)),
    ]
    valores = ["otimista", "pessimista", "neutro"]
    df_prev["cenario"] = np.select(condicoes, valores, default="neutro")

    # =============================
    # 🔹 Cálculo do preço médio ponderado
    # =============================
    df_prev["preco_ponderado"] = df_prev["preco"] * df_prev["prob"]
    df_media = (
        df_prev.groupby("t", as_index=False)
        .agg({
            "preco": "mean",
            "preco_ponderado": "sum",
            "data_futura": "first"
        })
        .rename(columns={"preco_ponderado": "preco_medio"})
    )

    df_final = pd.merge(
        df_prev,
        df_media[["t", "data_futura", "preco_medio"]],
        on=["t", "data_futura"],
        how="left",
        validate="many_to_one"
    )

    # ✅ garantir colunas mínimas para evitar KeyError
    for col in ["preco", "ret", "prob", "data_futura", "cenario", "preco_medio"]:
        if col not in df_final.columns:
            df_final[col] = np.nan

    return df_final[["t", "preco", "ret", "prob", "data_futura", "cenario", "preco_medio"]]
