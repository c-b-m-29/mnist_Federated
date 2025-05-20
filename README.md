# federated-mnist

Servidor de aprendizaje federado para MNIST usando FastAPI y PyTorch.

## Descripción

Este proyecto implementa un servidor para aprendizaje federado, donde múltiples clientes pueden conectarse mediante WebSocket, enviar sus pesos entrenados localmente y recibir actualizaciones globales del modelo. El modelo es una red neuronal densa simple para el dataset MNIST.

## Estructura del proyecto

```
federated-mnist/
├── app/
│   ├── __init__.py
│   ├── main.py         # Servidor FastAPI y lógica federada
│   ├── model.py        # Definición del modelo PyTorch
│   └── utils.py        # Utilidades: serialización y gestión de conexiones
├── requirements.txt    # Dependencias del proyecto
├── README.md
└── .gitignore
```

## Requisitos

- Python 3.8 o superior
- pip

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/federated-mnist.git
   cd federated-mnist
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta el servidor:

```bash
cd app
python main.py
```

El servidor escuchará por defecto en el puerto `10000`.

## Endpoints principales

- `ws://localhost:10000/ws`: WebSocket para clientes federados.  
  Envía y recibe pesos del modelo en formato JSON.
- `ws://localhost:10000/ws/echo`: WebSocket de prueba (echo).

## Documentación automática

FastAPI genera documentación automática:
- Swagger UI: [http://localhost:10000/docs](http://localhost:10000/docs)
- Redoc: [http://localhost:10000/redoc](http://localhost:10000/redoc)

## Detalles de implementación

- **Modelo:**  
  Definido en `app/model.py` como una red densa con una capa oculta de 32 unidades y salida de 10 clases.
- **Serialización:**  
  Las funciones para serializar/deserializar pesos están en `app/utils.py`.
- **Gestión de conexiones:**  
  El manejo de clientes WebSocket se realiza con la clase `ConnectionManager` en `app/utils.py`.
- **Promedio de pesos:**  
  Cuando al menos dos clientes envían sus pesos, el servidor promedia los pesos y envía la actualización global a todos los clientes conectados.
- **CORS:**  
  El servidor acepta conexiones de cualquier origen.

## Licencia

MIT