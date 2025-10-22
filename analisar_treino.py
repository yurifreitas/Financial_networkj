import torch

# Caminho do checkpoint simbiótico
PATH = "estado_treinamento_finance.pth"

# Carrega com total segurança (sem restrição de weights_only)
state = torch.load(PATH, map_location="cpu", weights_only=False)

# Obtém o dicionário do modelo
modelo_state = state.get("modelo", state)

print(f"🔑 Total de chaves no modelo: {len(modelo_state)}")

# Mostra uma amostra das primeiras chaves
sample_keys = list(modelo_state.keys())[:10]
print("\n🔍 Amostra de chaves:")
for k in sample_keys:
    print(" •", k)

# Tenta localizar automaticamente o peso da fc1 (ou equivalente)
target_key = None
for k in modelo_state.keys():
    if "fc1.weight" in k or "linear1.weight" in k or "dense1.weight" in k:
        target_key = k
        break

if target_key:
    peso = modelo_state[target_key]
    media = peso.abs().mean().item()
    std = peso.std().item()
    print(f"\n🧠 Camada detectada: {target_key}")
    print(f"   |peso médio| = {media:.6f}")
    print(f"   desvio padrão = {std:.6f}")
else:
    print("\n⚠️ Nenhuma camada 'fc1' encontrada. Verifique os nomes listados acima.")
