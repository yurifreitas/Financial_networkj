# =========================================================
# 🔮 EtherSym Finance — backend/predictor.py (simbiótico v3.0)
# =========================================================
# - Escala e calibração automáticas (Y_CLAMP adaptativo)
# - Correção de polarização (viés direcional do modelo)
# - Coerência simbiótica combinada (entropia + energia)
# - Suavização dinâmica e filtragem de ruído
# - Simbiose energética e ajuste de confiança preditiva
# =========================================================

import torch, numpy as np
from env import make_feats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def prever_tendencia(modelo, df, energia=1.0, coef_suavizacao=0.7):
    """
    Executa a previsão simbiótica adaptativa.
    Mantém consistência com o treinamento, aplica correção de polarização,
    calibra o alcance Y_CLAMP dinamicamente e ajusta coerência e energia.
    """

    # === 1️⃣ Preparação dos dados ===
    base, price = make_feats(df)
    if len(base) == 0 or len(price) == 0:
        raise ValueError("Dados insuficientes para previsão simbiótica")

    s = np.concatenate([base[-1], [0, 0]]).astype(np.float32)
    x = torch.tensor(s, device=DEVICE).unsqueeze(0)

    # === 2️⃣ Forward pass ===
    q_vals, y_pred = modelo(x)
    q_vals = torch.softmax(q_vals, dim=1)
    prob_actions = q_vals.squeeze(0).cpu().numpy()
    action = int(np.argmax(prob_actions) - 1)
    y_raw = float(y_pred.item())

    # === 3️⃣ Calibração simbiótica de faixa dinâmica ===
    # inicia valor de Y_CLAMP adaptativo
    if not hasattr(prever_tendencia, "_y_clamp_adapt"):
        prever_tendencia._y_clamp_adapt = 0.05
        prever_tendencia._erro_acum = 0.0
        prever_tendencia._cont = 0

    # limite do modelo (garante simetria)
    y_raw = float(np.clip(y_raw, -1.0, 1.0))

    # === 4️⃣ Ajuste adaptativo de faixa (aprende a calibrar amplitude) ===
    erro_estimado = abs(y_raw) - 0.5
    prever_tendencia._erro_acum = 0.98 * prever_tendencia._erro_acum + 0.02 * erro_estimado
    prever_tendencia._cont += 1
    ajuste_faixa = 1.0 + np.tanh(prever_tendencia._erro_acum) * 0.2
    Y_CLAMP = float(np.clip(prever_tendencia._y_clamp_adapt * ajuste_faixa, 0.02, 0.1))

    # === 5️⃣ Aplicar conversão simbiótica para escala real ===
    retorno_pred = y_raw * Y_CLAMP * np.clip(energia, 0.5, 1.5)

    # === 6️⃣ Correção de polarização direcional ===
    if not hasattr(prever_tendencia, "_media_pred"):
        prever_tendencia._media_pred = 0.0
    prever_tendencia._media_pred = 0.995 * prever_tendencia._media_pred + 0.005 * retorno_pred
    retorno_pred -= prever_tendencia._media_pred * 0.5  # re-centra previsões

    # === 7️⃣ Suavização adaptativa com peso dinâmico ===
    if hasattr(prever_tendencia, "_ultimo_pred"):
        peso = np.clip(coef_suavizacao * (2.0 - abs(retorno_pred) / (Y_CLAMP + 1e-9)), 0.4, 0.95)
        retorno_pred = peso * prever_tendencia._ultimo_pred + (1 - peso) * retorno_pred
    prever_tendencia._ultimo_pred = retorno_pred

    # === 8️⃣ Cálculo do preço previsto ===
    preco_atual = float(price[-1])
    preco_previsto = preco_atual * (1.0 + retorno_pred)

    # === 9️⃣ Métricas simbióticas: entropia e coerência ===
    entropia = -np.sum(prob_actions * np.log2(prob_actions + 1e-9))
    entropia_norm = 1.0 - np.clip(entropia / np.log2(len(prob_actions)), 0.0, 1.0)
    coerencia_direcional = 1.0 - abs(retorno_pred) / (Y_CLAMP + 1e-9)

    # combinação simbiótica (entropia + direção)
    coerencia = float(np.clip((entropia_norm**0.7) * (coerencia_direcional**0.3), 0.0, 1.0))

    # === 🔬 10️⃣ Ajuste energético simbiótico ===
    energia_pred = energia + 0.12 * (coerencia - 0.5) * np.sign(retorno_pred)
    energia_pred *= 1.0 + (abs(y_raw) - 0.5) * 0.15
    energia_pred = float(np.clip(energia_pred, 0.25, 1.8))

    # === 11️⃣ Filtro de ruído / estabilidade simbiótica ===
    # se coerência baixa e entropia alta → reduz amplitude preditiva
    if coerencia < 0.2 and entropia_norm < 0.3:
        retorno_pred *= 0.5
        energia_pred = (energia_pred + 0.5) / 2.0

    # === 12️⃣ Log simbiótico (debug adaptativo) ===
    if abs(retorno_pred) > 0.001:
        print(
            f"[Δpred simbiótico] y_raw={y_raw:+.4f} | clamp={Y_CLAMP:.3f} "
            f"→ retorno={retorno_pred:+.4f} | coer={coerencia:.2f} | enr={energia_pred:.2f}"
        )

    # === 13️⃣ Retorno simbiótico final ===
    return {
        "preco": preco_atual,
        "preco_previsto": preco_previsto,
        "retorno_pred": retorno_pred,
        "acao_modelo": action,
        "energia": energia_pred,
        "coerencia": coerencia,
        "prob_actions": prob_actions.tolist(),
        "Y_CLAMP": Y_CLAMP,
        "entropia": entropia_norm,
    }
