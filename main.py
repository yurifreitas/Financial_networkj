# =========================================================
# 🌌 EtherSym Finance — main.py (Simbiótico Avançado v7e Final)
# =========================================================
# - Double DQN + N-Step + Prioritized Replay
# - Regressão contínua normalizada (tanh-free)
# - AMP + GradClip + Homeostase + Poda + Regeneração
# - Anneal de ε, Temperatura e β-PER
# - LR Warmup + CosineDecay + Soft/Hard Target Sync
# =========================================================

import os, time, math, random, warnings
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from network import criar_modelo
from env import Env, make_feats
from memory import RingReplay, NStepBuffer, salvar_estado, carregar_estado
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 🧩 Utilitários internos (antes externos)
# =========================================================
def turbo_cuda():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_num_threads(1)
    except: pass

def reseed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"🌱 RNG reseed com seed={seed}")

@torch.no_grad()
def soft_update(target, online, tau=0.005):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)

def loss_q_hibrida(q_pred, q_tgt):
    mse = F.mse_loss(q_pred, q_tgt)
    hub = F.smooth_l1_loss(q_pred, q_tgt, beta=0.8)
    return 0.7 * mse + 0.3 * hub

def loss_regressao(y_pred, y_tgt, l2=1e-4):
    reg = F.smooth_l1_loss(y_pred, y_tgt, beta=0.5)
    return reg + l2 * (y_pred**2).mean()

def escolher_acao(modelo, estado_np, device, eps,
                  temp_base=0.95, temp_min=0.60,
                  capital=0.0, posicao=0.0, gate_conf=0.35):
    ACTIONS = np.array([-1, 0, 1], dtype=np.int8)
    if np.random.rand() < eps:
        return int(np.random.choice(ACTIONS)), 0.5
    x = torch.tensor(estado_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals, y = modelo(x)
        conf = float(torch.tanh(torch.abs(y)).item())
        temp = max(temp_min, temp_base * (0.8 + 0.2 * (eps / max(1e-6, eps))))
        probs = torch.softmax(q_vals / temp, dim=1).squeeze(0).clamp_(1e-6, 1.0)
        probs = (probs / probs.sum()).cpu().numpy()
    a = int(np.random.choice(ACTIONS, p=probs))
    if conf < gate_conf: a = 0
    if a == 1 and capital <= 0: a = 0
    if a == -1 and posicao <= 0: a = 0
    return a, conf

# =========================================================
# ⚙️ Hiperparâmetros principais
# =========================================================
CSV = "binance_BTC_USDT_1h_2y.csv"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (DEVICE.type == "cuda")
turbo_cuda(); reseed(SEED)

BATCH = 256; GAMMA = 0.995; LR = 1e-4; LR_MIN = 2e-6
LR_WARMUP_STEPS = 5_000; N_STEP = 3
MEMORIA_MAX = 250_000; MIN_REPLAY = 3_000
EPSILON_INICIAL, EPSILON_DECAY, EPSILON_MIN = 1.0, 0.9992, 0.05
TEMP_INI, TEMP_MIN, TEMP_DECAY = 0.95, 0.60, 0.9995
BETA_PER_INI, BETA_PER_MAX, BETA_PER_DECAY = 0.6, 1.0, 0.9999
LAMBDA_REG = 0.1; Y_CLAMP = 0.05
GRAD_CLIP = 0.5; LOSS_GUARD = 1e5
PRINT_EVERY, SAVE_EVERY, PODA_EVERY, HOMEOSTASE_EVERY = 400, 10_000, 5_000, 2_000
TARGET_TAU_BASE, HARD_SYNC_EVERY, CAPITAL_INICIAL = 0.005, 50_000, 1_000.0

ACTION_SPACE = np.array([-1, 0, 1], dtype=np.int8)
def a_to_idx(a: int): return int(a + 1)
def set_lr(optim, lr): [g.update(lr=lr) for g in optim.param_groups]
def is_bad_number(x): return torch.isnan(x).any() or torch.isinf(x).any()

# =========================================================
# 🚀 Setup
# =========================================================
if not os.path.exists(CSV): raise FileNotFoundError(f"CSV não encontrado: {CSV}")
df = pd.read_csv(CSV); base, price = make_feats(df)
env = Env(base, price)
modelo, alvo, opt = criar_modelo(DEVICE, lr=LR)
replay = RingReplay(state_dim=base.shape[1]+2, capacity=MEMORIA_MAX, device=DEVICE)
nbuf = NStepBuffer(N_STEP, GAMMA)
scaler = GradScaler("cuda", enabled=AMP)
lr_now = LR; set_lr(opt, lr_now)
_, EPSILON_SAVED, _ = carregar_estado(modelo, opt)
EPSILON = EPSILON_SAVED if EPSILON_SAVED is not None else EPSILON_INICIAL
print(f"🧠 Iniciando treino simbiótico | device={DEVICE.type}")

total_steps, episodio, best_global = 0, 0, CAPITAL_INICIAL
last_loss, last_y_pred = 0.0, 0.0
temp_now, beta_per = TEMP_INI, BETA_PER_INI
ema_q, ema_r = None, None

# =========================================================
# 🎮 Loop principal
# =========================================================
while True:
    episodio += 1
    s = env.reset()
    done = False
    capital = CAPITAL_INICIAL
    posicao = 0.0
    max_patrimonio = CAPITAL_INICIAL
    total_reward_ep = 0.0
    melhor_patrimonio_ep = CAPITAL_INICIAL

    while not done:
        total_steps += 1
        warmup = total_steps < 3_000
        eps_now = 1.0 if warmup else max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        if len(replay) >= MIN_REPLAY: EPSILON = eps_now

        if not warmup:
            temp_now = max(TEMP_MIN, temp_now * TEMP_DECAY)
            beta_per = min(BETA_PER_MAX, beta_per / BETA_PER_DECAY)
            try: replay.beta = float(beta_per)
            except: pass

        if total_steps <= LR_WARMUP_STEPS:
            lr_now = LR * total_steps / max(1, LR_WARMUP_STEPS)
        else:
            progress = min(1.0, (total_steps - LR_WARMUP_STEPS) / (1_000_000))
            lr_now = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        set_lr(opt, lr_now)

        s_cur = s
        a, conf = escolher_acao(modelo, s_cur, DEVICE, eps_now, capital, posicao)
        sp, r, done_env, info = env.step(a)
        total_reward_ep += r
        capital = float(info.get("capital", capital))
        patrimonio = float(info.get("patrimonio", capital))
        max_patrimonio = max(max_patrimonio, float(info.get("max_patrimonio", patrimonio)))
        posicao = float(env.pos)
        y_ret = float(info.get("ret", 0.0))
        melhor_patrimonio_ep = max(melhor_patrimonio_ep, patrimonio)

        nbuf.push(s_cur, a, r, y_ret)
        if len(nbuf.traj) == N_STEP or done_env:
            item = nbuf.flush(sp, done_env)
            if item:
                s0, a0, Rn, sn, dn, y0 = item
                replay.append(s0, a_to_idx(a0), Rn, sn, float(dn), y0)
        s = sp; done = done_env

        # =================================================
        # 🎓 Aprendizado simbiótico
        # =================================================
        if len(replay) >= MIN_REPLAY:
            (estados_t, acoes_t, recompensas_t, novos_estados_t,
             finais_t, idx, w, y_ret_t) = replay.sample(BATCH)
            with torch.no_grad():
                next_q_online, _ = modelo(novos_estados_t)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_q_target, _ = alvo(novos_estados_t)
                next_best = next_q_target.gather(1, next_actions).squeeze(1)
                alvo_q = recompensas_t + (GAMMA ** N_STEP) * next_best * (1.0 - finais_t)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=AMP):
                q_vals, y_pred = modelo(estados_t)
                q_sel = q_vals.gather(1, acoes_t).squeeze(1)
                loss_q = loss_q_hibrida(q_sel, alvo_q)
                y_target = y_ret_t.clamp_(-Y_CLAMP, Y_CLAMP) / Y_CLAMP
                loss_reg = loss_regressao(y_pred, y_target)

                if ema_q is None: ema_q, ema_r = float(loss_q.item()), float(loss_reg.item())
                ema_q = 0.98 * ema_q + 0.02 * float(loss_q.item())
                ema_r = 0.98 * ema_r + 0.02 * float(loss_reg.item())
                lambda_eff = LAMBDA_REG * max(0.3, min(2.0, (ema_q + 1e-3) / (ema_r + 1e-3)))
                loss_total = loss_q + lambda_eff * loss_reg

            if (is_bad_number(loss_total) or float(loss_total.item()) > LOSS_GUARD):
                opt.zero_grad(set_to_none=True)
                for p in modelo.parameters():
                    if p.grad is not None:
                        p.grad.detach_(); p.grad.zero_()
                # Reajusta estabilidade
                temp_now = min(TEMP_INI, temp_now * 1.01)
                EPSILON = min(1.0, EPSILON * 1.01)
                scaler = GradScaler("cuda", enabled=AMP)  # 🔁 reinicia o scaler em caso de falha

            else:
                scaler.scale(loss_total).backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()

            with torch.no_grad():
                td_error = (alvo_q - q_sel).abs().clamp_(0, 5.0).cpu().numpy()
                replay.update_priority(idx, td_error)
                loss_scalar = float(loss_q.item())
                tau = min(0.01, TARGET_TAU_BASE * (1.0 + min(2.0, loss_scalar)))
                soft_update(alvo, modelo, tau)
                if total_steps % HARD_SYNC_EVERY == 0: alvo.load_state_dict(modelo.state_dict())
            last_loss, last_y_pred = float(loss_total.item()), float(y_pred[-1].item())

        # =================================================
        # 🌿 Manutenção simbiótica
        # =================================================
        if total_steps % PODA_EVERY == 0 and len(replay) >= MIN_REPLAY:
            taxa = modelo.aplicar_poda()
            modelo.regenerar_sinapses(taxa)
            modelo.verificar_homeostase(last_loss)
            print(f"🌿 Poda simbiótica — taxa={taxa*100:.2f}% | step={total_steps}")

        if total_steps % HOMEOSTASE_EVERY == 0: replay.homeostase()
        modelo.verificar_homeostase(total_reward_ep / max(1, total_steps % 10_000))

        # =================================================
        # 🧾 Logs / checkpoints
        # =================================================
        if (total_steps % PRINT_EVERY == 0) or done:
            energia = float(info.get("energia", 1.0))
            print(f"[Ep {episodio:04d} | {total_steps:>8}] cap={capital:>9.2f} | pat={patrimonio:>9.2f} | "
                  f"max={max_patrimonio:>9.2f} | ε={eps_now:.3f} | τ={temp_now:.2f} | β={beta_per:.2f} | "
                  f"lr={lr_now:.6f} | enr={energia:.2f} | Δpred={last_y_pred:+.4f} | loss={last_loss:.5f}")

        if (total_steps % SAVE_EVERY == 0) and len(replay) >= MIN_REPLAY:
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"💾 Checkpoint salvo | step={total_steps}")

        if melhor_patrimonio_ep > best_global and len(replay) >= MIN_REPLAY:
            best_global = melhor_patrimonio_ep
            salvar_estado(modelo, opt, replay, EPSILON, total_reward_ep)
            print(f"🏆 Novo melhor patrimônio global={best_global:.2f} | step={total_steps}")

        if done:
            s = env.reset()
            if capital <= 1.0:
                best_global = max(best_global, max_patrimonio)
                print(f"\n💀 Falência simbiótica | cap_final={capital:.2f} | melhor_global={best_global:.2f} | ε={EPSILON:.3f}\n")
            break
