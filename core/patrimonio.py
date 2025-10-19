import os, json

PATRIMONIO_FILE = "patrimonio_global.json"

def carregar_patrimonio_global():
    if not os.path.exists(PATRIMONIO_FILE):
        return 1000.0
    try:
        with open(PATRIMONIO_FILE, "r") as f:
            data = json.load(f)
        return float(data.get("best_global", 1000.0))
    except Exception:
        return 1000.0

def salvar_patrimonio_global(valor):
    try:
        with open(PATRIMONIO_FILE, "w") as f:
            json.dump({"best_global": float(valor)}, f, indent=2)
        print(f"💾 Patrimônio global atualizado: {valor:.2f}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar patrimônio global: {e}")
