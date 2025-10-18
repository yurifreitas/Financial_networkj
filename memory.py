# ==========================================
# 💾 EtherSym Finance — Memory System (Seguro e Atômico)
# ==========================================
import os, torch, numpy as np
from collections import deque
from config import SAVE_PATH, MEMORIA_MAX, EPSILON_INICIAL

# =========================================================
# ♻️ Replay Memory (Ring Buffer de alta performance)
# =========================================================
class RingReplay:
    def __init__(self, state_dim, capacity=200_000, device="cpu"):
        self.state_dim = state_dim
        self.device = device
        self.cap = int(capacity)
        self.idx = 0
        self.full = False

        # buffers fixos pré-alocados
        self.s  = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.cap, 1), dtype=np.int64)
        self.r  = np.zeros((self.cap,), dtype=np.float32)
        self.sn = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.cap,), dtype=np.float32)

    def append(self, s, a_idx, r, sn, d):
        i = self.idx
        self.s[i], self.a[i, 0], self.r[i], self.sn[i], self.d[i] = s, a_idx, r, sn, d
        self.idx = (i + 1) % self.cap
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch):
        idx = np.random.randint(0, len(self), size=batch)
        s  = torch.as_tensor(self.s[idx],  device=self.device)
        a  = torch.as_tensor(self.a[idx],  device=self.device)
        r  = torch.as_tensor(self.r[idx],  device=self.device)
        sn = torch.as_tensor(self.sn[idx], device=self.device)
        d  = torch.as_tensor(self.d[idx],  device=self.device)
        return s, a, r, sn, d


# =========================================================
# 🧩 N-Step Buffer (para reforço temporal)
# =========================================================
class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.traj = deque(maxlen=n)

    def push(self, s, a, r):
        self.traj.append((s, a, r))

    def flush(self, sn, done):
        if not self.traj:
            return None
        R = sum((self.gamma ** i) * float(r) for i, (_, _, r) in enumerate(self.traj))
        s0, a0, _ = self.traj[0]
        self.traj.clear()
        return s0, a0, R, sn, done


# =========================================================
# 💾 Estado seguro (com salvamento atômico)
# =========================================================
def salvar_estado(modelo, opt, replay, eps, media, path=None):
    path = path or SAVE_PATH
    tmp_path = path + ".tmp"

    # dump temporário atômico
    torch.save({
        "modelo": modelo.state_dict(),
        "opt": opt.state_dict(),
        "eps": float(eps),
        "media": float(media or 0)
    }, tmp_path)

    # substituição segura (sem corromper)
    os.replace(tmp_path, path)


# =========================================================
# ♻️ Carregamento tolerante a erros
# =========================================================
def carregar_estado(modelo, opt, path=None):
    path = path or SAVE_PATH
    if not os.path.exists(path):
        print("⚙️ Nenhum estado salvo encontrado, iniciando do zero.")
        return None, EPSILON_INICIAL, 0.0

    try:
        data = torch.load(path, map_location="cpu")
        modelo.load_state_dict(data.get("modelo", {}), strict=False)

        if opt is not None and "opt" in data:
            opt.load_state_dict(data["opt"])

        eps = float(data.get("eps", EPSILON_INICIAL))
        media = float(data.get("media", 0.0))
        print(f"♻️ Estado carregado | ε={eps:.3f} | média={media:+.3f}")
        return RingReplay(6, device="cpu"), eps, media

    except Exception as e:
        print(f"⚠️ Erro ao carregar estado ({type(e).__name__}): {e}")
        print("🔄 Reiniciando do zero (arquivo pode estar corrompido).")
        return None, EPSILON_INICIAL, 0.0
