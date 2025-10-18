import torch, numpy as np, pandas as pd
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rede_markoviana(modelo, df, profundidade=5, bifurcacoes=3, temp=0.7):
    base, price = make_feats(df)
    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)

    caminhos = [{"t": 0, "preco": float(price[-1]), "ret": 0.0, "prob": 1.0}]
    previsoes = []

    for t in range(profundidade):
        novos_caminhos = []
        for c in caminhos:
            q_vals, y_pred = modelo(x)
            probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).cpu().numpy()
            acao_space = np.array([-1, 0, 1])
            top_idx = np.argsort(probs)[-bifurcacoes:]
            for i in top_idx:
                a = acao_space[i]
                conf = probs[i]
                ret_pred = float(y_pred.item()) * a
                preco_fut = c["preco"] * (1 + ret_pred)
                novos_caminhos.append({
                    "t": c["t"] + 1,
                    "preco": preco_fut,
                    "ret": ret_pred,
                    "prob": c["prob"] * conf
                })
        total_prob = sum([n["prob"] for n in novos_caminhos]) + 1e-9
        for n in novos_caminhos:
            n["prob"] /= total_prob
        previsoes.extend(novos_caminhos)
        caminhos = novos_caminhos

    df_prev = pd.DataFrame(previsoes)
    df_prev["data_futura"] = pd.to_datetime(df["timestamp"].iloc[-1]) + pd.to_timedelta(df_prev["t"], unit="h")
    return df_prev
