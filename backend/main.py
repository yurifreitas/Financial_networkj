# =========================================================
# 🌌 EtherSym Finance — Backend WebSocket (corrigido)
# =========================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictor import prever_tendencia
from backend.markov_predictor import rede_markoviana
from backend.strategy import EstrategiaVariacao
import asyncio, json, datetime, pandas as pd, numpy as np

# =========================================================
# 🚀 Inicialização do app e middleware
# =========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 🧠 Modelos e estratégias
# =========================================================
modelo = carregar_modelo()
estrategia = EstrategiaVariacao()

# =========================================================
# 🧩 Utilitário de serialização segura
# =========================================================
def safe_json(data):
    """Serializa objetos complexos (Timestamp, numpy, etc.)"""
    def convert(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, (pd.Timestamp, datetime.datetime)):
            return o.isoformat()
        return str(o)
    return json.dumps(data, default=convert)

# =========================================================
# 🔮 Socket de Tendência (predição contínua)
# =========================================================
@app.websocket("/ws/tendencia")
async def tendencia_socket(ws: WebSocket):
    await ws.accept()
    print("📡 Cliente conectado em /ws/tendencia")

    try:
        while True:
            df = get_recent_candles(limit=120)
            pred = prever_tendencia(modelo, df)
            sinal = estrategia.aplicar(pred)

            msg = {
                "tempo": datetime.datetime.utcnow().isoformat(),
                "preco": float(pred.get("preco", 0)),
                "retorno_pred": float(pred.get("retorno_pred", 0)),
                "acao_modelo": int(pred.get("acao_modelo", 0)),
                "sinal_final": int(sinal),
                "posicao": int(estrategia.posicao),
            }

            await ws.send_text(safe_json(msg))
            await asyncio.sleep(60)  # atualiza a cada 1 minuto

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado de /ws/tendencia")
    except asyncio.CancelledError:
        print("⚠️ Loop /ws/tendencia cancelado (provável shutdown ou reload).")
    except Exception as e:
        print(f"❌ Erro inesperado em /ws/tendencia: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

# =========================================================
# 🔁 Socket Markov (rede preditiva com bifurcações)
# =========================================================
@app.websocket("/ws/markov")
async def markov_socket(ws: WebSocket):
    await ws.accept()
    print("📡 Cliente conectado em /ws/markov")

    try:
        while True:
            df = get_recent_candles(limit=120)
            df_prev = rede_markoviana(modelo, df, profundidade=6, bifurcacoes=3)
            data = df_prev.to_dict(orient="records")

            await ws.send_text(safe_json(data))
            await asyncio.sleep(300)  # a cada 5 minutos

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado de /ws/markov")
    except asyncio.CancelledError:
        print("⚠️ Loop /ws/markov cancelado (provável shutdown ou reload).")
    except Exception as e:
        print(f"❌ Erro inesperado em /ws/markov: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
from backend.simulador_realtime import simulador

@app.websocket("/ws/robo")
async def robo_socket(ws: WebSocket):
    await ws.accept()
    print("🤖 Cliente conectado: /ws/robo")
    try:
        while True:
            dado = await simulador.tick()
            await ws.send_text(json.dumps(dado))
            await asyncio.sleep(60)  # 1 tick/minuto
    except WebSocketDisconnect:
        print("❌ Cliente desconectado de /ws/robo")
