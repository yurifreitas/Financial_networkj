# =========================================================
# 💾 EtherSym Finance — persistence_env.py
# =========================================================
import json, os
from .config_env import BEST_SCORE_FILE

def load_best_score():
    try:
        if not os.path.exists(BEST_SCORE_FILE):
            return 0.0
        data = json.load(open(BEST_SCORE_FILE))
        return float(data.get("score", 0.0)) if isinstance(data, dict) else float(data)
    except Exception:
        return 0.0

def save_best_score(score: float):
    try:
        json.dump({"score": float(score)}, open(BEST_SCORE_FILE, "w"), indent=2)
    except Exception as e:
        print(f"[WARN] Falha ao salvar best_score: {e}")
