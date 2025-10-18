import torch
from network import RedeAvancada

STATE_PATH = "estado_treinamento_finance.pth"

def carregar_modelo(device="cuda"):
    modelo = RedeAvancada(state_dim=10, n_actions=3).to(device)
    data = torch.load(STATE_PATH, map_location=device)
    if "modelo" in data:
        modelo.load_state_dict(data["modelo"], strict=False)
    else:
        modelo.load_state_dict(data, strict=False)
    modelo.eval()
    print(f"🧠 Modelo carregado: {STATE_PATH}")
    return modelo
