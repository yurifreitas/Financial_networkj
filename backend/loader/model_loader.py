# ============================================================
# 🧠 EtherSym Finance — Carregamento simbiótico robusto do modelo
# ============================================================
import os
import torch
from v1.network import RedeAvancada

STATE_PATH = "/home/yuri/Documents/code2/binance-model/backend/loader/estado_treinamento_finance.pth"


def carregar_modelo(device="cuda"):
    """
    Carrega o modelo de rede neural salvo em STATE_PATH com verificação simbiótica:
    - Compatível com PyTorch >= 2.6 (safe_globals / weights_only)
    - Remove prefixos _orig_mod de checkpoints compilados
    - Detecta chaves ausentes/extras e faz diagnósticos simbióticos
    """

    # Ajuste automático para CPU caso não haja GPU
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("⚠️ GPU não detectada — usando CPU.")
        device = "cpu"

    # Verifica se o arquivo existe
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(f"❌ Arquivo de estado não encontrado em: {STATE_PATH}")

    print(f"🧩 Carregando modelo simbiótico de: {STATE_PATH}")

    # Inicializa estrutura base
    modelo = RedeAvancada(state_dim=10, n_actions=3).to(device)

    # =========================================================
    # 🧬 Carregamento seguro compatível (PyTorch 2.9)
    # =========================================================
    try:
        # Tenta modo padrão (weights_only=True)
        checkpoint = torch.load(STATE_PATH, map_location=device)
    except Exception as e1:
        print(f"⚠️ Falha no carregamento seguro ({type(e1).__name__}): {e1}")
        try:
            from torch.serialization import safe_globals
            with safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(STATE_PATH, map_location=device)
            print("🔒 Carregado com safe_globals (TorchVersion liberado).")
        except Exception as e2:
            print(f"⚠️ Falha com safe_globals: {e2}")
            print("🚨 Usando fallback com weights_only=False (modo legado, seguro se o arquivo é seu).")
            checkpoint = torch.load(STATE_PATH, map_location=device, weights_only=False)

    # =========================================================
    # 🔍 Extrai o state_dict simbiótico
    # =========================================================
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("modelo") or checkpoint.get("state_dict") or checkpoint
    else:
        raise ValueError(f"❌ Checkpoint inválido: tipo {type(checkpoint)}")

    # 🔧 Remove prefixo _orig_mod. se existir (gerado por torch.compile)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Diagnóstico de integridade
    model_keys = set(modelo.state_dict().keys())
    loaded_keys = set(clean_state.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        print(f"⚠️ {len(missing)} camadas não encontradas no arquivo (mantidas iniciais).")
    if unexpected:
        print(f"⚠️ {len(unexpected)} chaves extras detectadas (ignoradas).")

    # =========================================================
    # 💾 Aplicação dos pesos
    # =========================================================
    modelo.load_state_dict(clean_state, strict=False)
    modelo.eval()

    # Diagnóstico final simbiótico
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"✅ Modelo restaurado com sucesso:")
    print(f"   • Parâmetros totais:   {total_params:,}")
    print(f"   • Parâmetros treináveis: {trainable_params:,}")
    print(f"   • Dispositivo ativo:   {device}")

    # Sanidade simbiótica
    with torch.no_grad():
        mean_w = modelo.fc1.weight.abs().mean().item()
        if mean_w < 1e-5:
            print("⚠️ Atenção: pesos parecem não carregados (valores muito próximos de zero).")

    return modelo


# ============================================================
# 🔍 Teste rápido
# ============================================================
if __name__ == "__main__":
    modelo = carregar_modelo()
