import torch
from v1.network import RedeAvancada

# Caminho do estado salvo
PATH = "/home/yuri/Documents/code2/binance-model/backend/loader/estado_treinamento_finance.pth"

# =========================================================
# 🧠 Carregamento compatível (PyTorch ≥2.6 com proteção weights_only)
# =========================================================
try:
    # 🔹 Tenta o modo seguro (novo padrão)
    data = torch.load(PATH, map_location="cpu")
except Exception as e:
    print(f"⚠️ Falha no carregamento seguro ({type(e).__name__}): {e}")

    # 🧩 Detecção automática e fallback simbiótico
    try:
        from torch.serialization import safe_globals
        with safe_globals([torch.torch_version.TorchVersion]):
            data = torch.load(PATH, map_location="cpu")
        print("🔒 Carregado com safe_globals (TorchVersion liberado).")
    except Exception as e2:
        print(f"⚠️ Falha com safe_globals: {e2}")
        print("🚨 Usando fallback com weights_only=False (modo legado).")
        data = torch.load(PATH, map_location="cpu", weights_only=False)

# =========================================================
# 🔍 Extração do state_dict e comparação
# =========================================================
state = data.get("modelo", data)

modelo = RedeAvancada(state_dim=10, n_actions=3)
model_keys = set(modelo.state_dict().keys())
loaded_keys = set(state.keys())

missing = sorted(list(model_keys - loaded_keys))
extra = sorted(list(loaded_keys - model_keys))

print("🚨 Chaves faltantes:", missing[:10])
print("🚨 Chaves extras:", extra[:10])
print(f"Total faltando: {len(missing)} | Extras: {len(extra)}")
