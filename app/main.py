import json
import torch
import time
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from model import DenseModel
from utils import serialize_model_weights, deserialize_model_weights, ConnectionManager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = DenseModel()
print("Modelo en el servidor:", model)
global_weights = model.state_dict()
client_weights_list = []
manager = ConnectionManager()

@app.websocket("/ws")
async def federated_client(websocket: WebSocket):
    global global_weights
    await manager.connect(websocket)
    print("Cliente conectado")
    await websocket.send_text(json.dumps({
        "type": "init_model",
        "model_config": {
            "layers": [
                {"type": "dense", "units": 32, "activation": "relu", "inputShape": [784]},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ]
        },
        "weights": serialize_model_weights(global_weights)
    }))
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "weights":
                client_weights = deserialize_model_weights(msg["weights"])
                print("Tamaños de pesos recibidos:", [v.shape for v in client_weights.values()])
                print("Cliente:", websocket.client)
                print("Hora:", time.time())
                client_weights_list.append(client_weights)
                print(f"Pesos recibidos de un cliente. Total: {len(client_weights_list)}")
                if len(client_weights_list) >= 2:
                    averaged = {k: torch.zeros_like(v) for k, v in global_weights.items()}
                    for cw in client_weights_list:
                        for k in averaged:
                            averaged[k] += cw[k]
                    for k in averaged:
                        averaged[k] /= len(client_weights_list)
                    global_weights = averaged
                    client_weights_list.clear()
                    metrics = {"accuracy": round(torch.rand(1).item(), 3), "loss": round(torch.rand(1).item(), 3)}
                    await manager.broadcast({
                        "type": "global_update",
                        "weights": serialize_model_weights(global_weights),
                        "metrics": metrics
                    })
    except Exception as e:
        print("Desconexión:", e)
        manager.disconnect(websocket)

@app.websocket("/ws/echo")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)