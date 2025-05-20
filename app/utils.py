import torch
from typing import Dict, List, Any
from fastapi import WebSocket

def serialize_model_weights(state_dict: Dict[str, torch.Tensor]) -> List[Any]:
    """
    Convierte el state_dict del modelo en una lista de listas para enviar por WebSocket.
    """
    return [
        state_dict["fc1.weight"].cpu().numpy().tolist(),
        state_dict["fc1.bias"].cpu().numpy().tolist(),
        state_dict["fc2.weight"].cpu().numpy().tolist(),
        state_dict["fc2.bias"].cpu().numpy().tolist(),
    ]

def deserialize_model_weights(weights: List[Any]) -> Dict[str, torch.Tensor]:
    """
    Convierte una lista de listas recibida por WebSocket en un state_dict para el modelo.
    """
    keys = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    return {k: torch.tensor(w) for k, w in zip(keys, weights)}

class ConnectionManager:
    """
    Maneja las conexiones WebSocket activas.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Any):
        for connection in self.active_connections:
            await connection.send_json(message)