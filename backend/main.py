# =========================================================
# 🌌 EtherSym Finance — Backend WebSocket (estável e simbiótico)
# =========================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.loader.model_loader import carregar_modelo
from backend.binance_feed import get_recent_candles
from backend.predictors.predictor import prever_tendencia
from backend.strategies.strategy import EstrategiaVariacao
from backend.simulador_realtime import simulador
import asyncio, json, datetime, pandas as pd, numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = carregar_modelo()
modelo_lock = asyncio.Lock()
estrategia = EstrategiaVariacao()

# =========================================================
# 🧩 Utilitário de serialização segura
# =========================================================
def safe_json(data):
    def convert(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, (pd.Timestamp, datetime.datetime)):
            return o.isoformat()
        return str(o)
    return json.dumps(data, default=convert)


# =========================================================
# 🔮 Socket de Tendência
# =========================================================
@app.websocket("/ws/tendencia")
async def tendencia_socket(ws: WebSocket):
    await ws.accept()
    print("📡 Cliente conectado em /ws/tendencia")

    try:
        while True:
            async with modelo_lock:
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
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("🔌 Cliente desconectado de /ws/tendencia")
    except Exception as e:
        print(f"❌ Erro em /ws/tendencia: {e}")
    finally:
        await ws.close()


# =========================================================
# 🤖 Socket do Robô (simulação simbiótica completa)
# =========================================================
@app.websocket("/ws/robo")
async def robo_socket(ws: WebSocket):
    await ws.accept()
    print("🤖 Cliente conectado: /ws/robo")

    try:
        while True:
            async with modelo_lock:
                dado = await simulador.tick()
            await ws.send_text(safe_json(dado))

            # envia sinal de reset para frontend quando patrimônio cair
            if dado.get("episode_reset"):
                await ws.send_text(safe_json({"event": "reset", "timestamp": datetime.datetime.utcnow().isoformat()}))

            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("❌ Cliente desconectado de /ws/robo")
    except Exception as e:
        print(f"❌ Erro em /ws/robo: {e}")
    finally:
        await ws.close()
