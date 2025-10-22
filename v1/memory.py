# ==========================================
# 💾 EtherSym Finance — Memory System (Prioritized + Homeostatic + Regressão)
# ==========================================
import os, torch, numpy as np
from collections import deque
from v1.config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL

# =========================================================
# ♻️ Replay Memory (Ring Buffer com priorização simbiótica robusta)
# =========================================================
class RingReplay:
    def __init__(self, state_dim, capacity=200_000, device="cpu", alpha=0.6, beta=0.4):
        self.state_dim = state_dim
        self.device = device
        self.cap = int(capacity)
        self.idx = 0
        self.full = False

        # buffers principais
        self.s  = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.cap, 1), dtype=np.int64)
        self.r  = np.zeros((self.cap,), dtype=np.float32)
        self.sn = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.cap,), dtype=np.float32)
        self.y  = np.zeros((self.cap,), dtype=np.float32)

        # prioridades simbióticas seguras
        self.p  = np.ones((self.cap,), dtype=np.float32)
        self.alpha = alpha
        self.beta  = beta

        # parâmetros de segurança
        self.eps_p = 1e-6
        self.p_min, self.p_max = 1e-6, 10.0

    def __len__(self):
        return self.cap if self.full else self.idx

    # ------------------------------------------------------
    # Adiciona uma nova transição simbiótica
    # ------------------------------------------------------
    def append(self, s, a_idx, r, sn, d, y_ret):
        i = self.idx
        self.s[i], self.a[i, 0], self.r[i], self.sn[i], self.d[i], self.y[i] = s, a_idx, r, sn, d, y_ret

        # prioridade inicial robusta baseada em |r|
        pri = abs(r) + self.eps_p
        if not np.isfinite(pri): pri = 1.0
        self.p[i] = np.clip(pri, self.p_min, self.p_max)

        self.idx = (i + 1) % self.cap
        if self.idx == 0:
            self.full = True

    # ------------------------------------------------------
    # Normaliza prioridades de forma robusta
    # ------------------------------------------------------
    def _get_probs(self):
        n = len(self)
        p = self.p[:n].astype(np.float64)
        p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=0.0)
        p = np.clip(p, self.p_min, self.p_max)
        p = p ** self.alpha

        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            # fallback para distribuição uniforme
            p = np.full(n, 1.0 / n, dtype=np.float32)
        else:
            p = (p / s).astype(np.float32)
        return p
    # ------------------------------------------------------
    # Reinicializa o buffer (limpeza total)
    # ------------------------------------------------------
    def clear(self):
        """Esvazia completamente o replay buffer."""
        self.idx = 0
        self.full = False
        self.s.fill(0.0)
        self.a.fill(0)
        self.r.fill(0.0)
        self.sn.fill(0.0)
        self.d.fill(0.0)
        self.y.fill(0.0)
        self.p.fill(1.0)
        print("♻️ Replay Buffer limpo com sucesso.")

    # ------------------------------------------------------
    # Mantém apenas os N% mais recentes do buffer
    # ------------------------------------------------------
    def trim(self, keep_ratio=0.5):
        """Mantém apenas parte recente do replay (por padrão 50%)."""
        n = len(self)
        if n == 0:
            return
        k = int(n * keep_ratio)
        start = n - k

        # Copia os dados mais recentes pro início
        self.s[:k]  = self.s[start:n]
        self.a[:k]  = self.a[start:n]
        self.r[:k]  = self.r[start:n]
        self.sn[:k] = self.sn[start:n]
        self.d[:k]  = self.d[start:n]
        self.y[:k]  = self.y[start:n]
        self.p[:k]  = self.p[start:n]

        self.idx = k
        self.full = False
        print(f"♻️ Replay reduzido: mantidos {k}/{n} exemplos ({keep_ratio*100:.0f}%).")

    # ------------------------------------------------------
    # Amostragem com PER e pesos de importância
    # ------------------------------------------------------
    def sample(self, batch):
        n = len(self)
        if n == 0:
            raise ValueError("Replay vazio — buffer sem amostras.")
        p = self._get_probs()

        try:
            idx = np.random.choice(n, batch, p=p)
        except ValueError:
            # fallback seguro (uniforme)
            p = np.full(n, 1.0 / n, dtype=np.float32)
            idx = np.random.choice(n, batch, p=p)

        # pesos de importância IS
        w = (n * p[idx]) ** (-self.beta)
        w /= max(self.eps_p, w.max())

        # conversão para tensores
        s  = torch.as_tensor(self.s[idx],  device=self.device)
        a  = torch.as_tensor(self.a[idx],  device=self.device)
        r  = torch.as_tensor(self.r[idx],  device=self.device)
        sn = torch.as_tensor(self.sn[idx], device=self.device)
        d  = torch.as_tensor(self.d[idx],  device=self.device)
        y  = torch.as_tensor(self.y[idx],  device=self.device)
        w  = torch.as_tensor(w,            device=self.device, dtype=torch.float32)
        return s, a, r, sn, d, idx, w, y

    # ------------------------------------------------------
    # Atualiza prioridades conforme TD-error (saneado)
    # ------------------------------------------------------
    def update_priority(self, idx, td_error):
        td = np.abs(td_error)
        td = np.nan_to_num(td, nan=1.0, posinf=1.0, neginf=0.0)
        self.p[idx] = np.clip(td + self.eps_p, self.p_min, self.p_max)

    # ------------------------------------------------------
    # Homeostase simbiótica (decadência lenta e sanitização)
    # ------------------------------------------------------
    def homeostase(self):
        self.p *= 0.9995
        self.p = np.nan_to_num(self.p, nan=1.0, posinf=1.0, neginf=1.0)
        self.p = np.clip(self.p, self.p_min, self.p_max)


# =========================================================
# 🧩 N-Step Buffer (para reforço temporal + regressão)
# =========================================================
class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.traj = deque(maxlen=n)

    # agora recebe o y_ret (retorno futuro real)
    def push(self, s, a, r, y_ret):
        self.traj.append((s, a, r, y_ret))

    def flush(self, sn, done):
        if not self.traj:
            return None
        R = sum((self.gamma ** i) * float(r) for i, (_, _, r, _) in enumerate(self.traj))
        s0, a0, _, y0 = self.traj[0]  # 🔮 y_ret do primeiro passo
        self.traj.clear()
        return s0, a0, R, sn, done, y0


def salvar_estado(modelo, opt, replay, eps, media, path=None):
    """
    Salva o estado simbiótico completo de forma atômica e compatível com torch.compile().
    """
    path = path or SAVE_PATH
    tmp_path = path + ".tmp"

    try:
        # 🔍 Corrige o caso de modelo compilado (torch.compile encapsula _orig_mod)
        to_save = modelo
        if hasattr(modelo, "_orig_mod"):
            to_save = modelo._orig_mod

        # 💾 Conteúdo simbiótico do estado
        state = {
            "modelo": to_save.state_dict(),
            "opt": opt.state_dict() if opt is not None else None,
            "eps": float(eps),
            "media": float(media or 0),
            "info": {
                "torch_version": torch.__version__,
                "device": str(next(to_save.parameters()).device),
            }
        }

        # 💽 Salvamento atômico
        torch.save(state, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
        print(f"💾 Estado simbiótico salvo com sucesso em: {path}")

    except Exception as e:
        print(f"⚠️ Erro ao salvar estado simbiótico: {e}")


def carregar_estado(modelo, opt, path=None, state_dim=10):
    """
    Carrega estado simbiótico tolerante a falhas, corrigindo prefixos _orig_mod.
    """
    path = path or SAVE_PATH
    if not os.path.exists(path):
        print("⚙️ Nenhum estado salvo encontrado, iniciando do zero.")
        return RingReplay(state_dim, device="cpu"), EPSILON_INICIAL, 0.0

    try:
        data = torch.load(path, map_location="cpu")
        state = data.get("modelo", {})

        # 🧩 Corrige prefixos do torch.compile
        fixed_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

        modelo.load_state_dict(fixed_state, strict=False)

        if opt is not None and "opt" in data and data["opt"] is not None:
            opt.load_state_dict(data["opt"])

        eps = float(data.get("eps", EPSILON_INICIAL))
        media = float(data.get("media", 0.0))

        meta = data.get("info", {})
        if meta:
            print(f"🔍 Checkpoint info: Torch {meta.get('torch_version')} | device={meta.get('device')}")

        print(f"♻️ Estado simbiótico carregado | ε={eps:.3f} | média={media:+.3f}")
        return RingReplay(state_dim, device="cpu"), eps, media

    except Exception as e:
        print(f"⚠️ Erro ao carregar estado ({type(e).__name__}): {e}")
        print("🔄 Reiniciando do zero (arquivo pode estar corrompido).")
        return RingReplay(state_dim, device="cpu"), EPSILON_INICIAL, 0.0
