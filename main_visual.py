# =========================================================
# 🌌 EtherSym Finance — main_visual.py (game sniper)
# =========================================================
# - IA joga sozinha (sem humano)
# - Acerta se erro < ERRO_MAX do env
# - Reseta streak ao errar
# - Mostra pontos, streak, melhor, erros, ação, equity, ε e erro%
# - Persiste melhor_streak em best_score.json
# =========================================================

import os, json, random, pygame, torch
import numpy as np, pandas as pd

from network import criar_modelo
from env import Env, make_feats
try:
    from env import ERRO_MAX
except Exception:
    ERRO_MAX = 0.003  # fallback

CSV = "binance_BTC_USDT_1h_2y.csv"
STATE_PATH = "estado_treinamento_finance.pth"
BEST_PATH = "best_score.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FPS = 60
LARGURA, ALTURA = 1000, 600
ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)

# -------------------------------------------------
# Seleção de ação (softmax temperado)
# -------------------------------------------------
@torch.no_grad()
def escolher_acao(modelo, s, eps=0.05):
    if random.random() < eps:
        return int(np.random.choice(ACTION_SPACE))
    x = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    p = torch.softmax(modelo(x) / 0.8, dim=1).squeeze(0).cpu().numpy()
    return int(np.random.choice(ACTION_SPACE, p=p))

# -------------------------------------------------
# Renderização de texto pygame
# -------------------------------------------------
def texto(tela, msg, x, y, cor=(255,255,255), tam=24):
    f = pygame.font.Font(None, tam)
    tela.blit(f.render(msg, True, cor), (x, y))

# -------------------------------------------------
# Persistência do recorde
# -------------------------------------------------
def carregar_recorde(path=BEST_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return int(json.load(f).get("melhor_streak", 0))
        except Exception:
            return 0
    return 0

def salvar_recorde(valor, path=BEST_PATH):
    try:
        with open(path, "w") as f:
            json.dump({"melhor_streak": int(valor)}, f, indent=2)
    except Exception as e:
        print(f"⚠️ Falha ao salvar recorde: {e}")

# -------------------------------------------------
# Main loop visual
# -------------------------------------------------
def main():
    # ==== dataset / env ====
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"❌ CSV não encontrado: {CSV}")
    df = pd.read_csv(CSV)
    base, price = make_feats(df)
    env = Env(base, price)

    # ==== modelo ====
    modelo, _, _, _ = criar_modelo(DEVICE)
    modelo.eval()
    eps = 0.05
    if os.path.exists(STATE_PATH):
        try:
            data = torch.load(STATE_PATH, map_location="cpu")
            if "modelo" in data:
                modelo.load_state_dict(data["modelo"], strict=False)
            eps = float(data.get("eps", eps))
            print(f"🧠 Estado carregado ({STATE_PATH}) | ε={eps:.3f}")
        except Exception as e:
            print(f"⚠️ Falha ao carregar estado ({type(e).__name__}): {e}")
    else:
        print(f"ℹ️ Sem estado salvo ({STATE_PATH}). Rodando com pesos atuais.")

    # ==== pygame ====
    pygame.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("🌌 EtherSym Finance — Game (Sniper Mode)")
    clock = pygame.time.Clock()

    pontos = streak = erros = 0
    melhor_streak = carregar_recorde()
    s = env.reset()
    flash = 0
    ultimos_erros = []

    print("🚀 Visual sniper iniciado (ESC para sair).")

    running = True
    while running:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        # ===== passo da IA =====
        a = escolher_acao(modelo, s, eps)
        sp, _, done, info = env.step(a)
        erro = float(info.get("erro", 1.0))
        eq   = float(info.get("eq", 1.0))

        ultimos_erros.append(erro)
        if len(ultimos_erros) > 200:
            ultimos_erros.pop(0)

        # ===== regra de acerto =====
        acertou = (erro < ERRO_MAX)
        if acertou:
            pontos += 1
            streak += 1
            if streak > melhor_streak:
                melhor_streak = streak
                salvar_recorde(melhor_streak)
        else:
            erros += 1
            streak = 0
            flash = 10
            s = env.reset()
            continue

        s = sp

        # ===== render =====
        if flash > 0:
            tela.fill((120, 20, 20))
            flash -= 1
        else:
            tela.fill((10, 12, 30))

        # Barra equity
        eq_bar = int(min(max(eq, 0.0), 2.0) * 300)
        pygame.draw.rect(tela, (0, 180, 255), (20, ALTURA - 40, eq_bar, 18))

        # Erro percentual
        err_pct = erro * 100
        alvo_pct = ERRO_MAX * 100
        texto(tela, f"Erro: {err_pct:.3f}% (alvo < {alvo_pct:.3f}%)", 20, 240, (255, 210, 120), 24)

        # Sparkline (histórico de erro)
        base_x, base_y, h = 20, 280, 80
        max_show = max(alvo_pct * 3, 1.0)
        for i, e_val in enumerate(ultimos_erros[-300:]):
            x = base_x + i
            mag = min((e_val * 100) / max_show, 1.0)
            hh = int(mag * h)
            cor = (80, 220, 80) if e_val < ERRO_MAX else (240, 90, 90)
            pygame.draw.line(tela, cor, (x, base_y + h - hh), (x, base_y + h), 1)

        # HUD principal
        texto(tela, f"Pontos: {pontos}", 20, 20, (255, 255, 0))
        texto(tela, f"Streak: {streak}", 20, 50, (0, 255, 0))
        texto(tela, f"Melhor: {melhor_streak}", 20, 80, (0, 180, 255))
        texto(tela, f"Erros: {erros}", 20, 110, (255, 80, 80))
        texto(tela, f"Ação: {a:+d}", 20, 140, (180, 180, 255))
        texto(tela, f"Equity: {eq:.3f}", 20, 170, (0, 200, 255))
        texto(tela, f"ε: {eps:.3f}", 20, 200, (150, 150, 255))

        # Legenda sparkline
        pygame.draw.line(tela, (80, 220, 80), (20, 360), (40, 360), 3)
        texto(tela, "erro < alvo", 45, 352, (200, 200, 200), 22)
        pygame.draw.line(tela, (240, 90, 90), (150, 360), (170, 360), 3)
        texto(tela, "erro >= alvo", 175, 352, (200, 200, 200), 22)

        pygame.display.flip()

    pygame.quit()
    salvar_recorde(melhor_streak)
    print(f"🕹️ Sessão encerrada — melhor streak total: {melhor_streak}")

if __name__ == "__main__":
    main()
