"""
Streaming
- Modo messages: la app recibe token por token a medida que el modelo los produce.
- Modo values:   el modelo comunica estado a medida que cumple pasos o tareas.
- Modo custom:   para inyectar tokens externamente a la respuesta del agente.
Se pueden combinar varios modos de streaming.
Los tokens producidos por un tool no forman parte de la observación devuelta al modelo. Son
inyectados directamente como strings crudos a la respuesta del agente.
"""

import dotenv

from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations
from langgraph.config import get_stream_writer

dotenv.load_dotenv()

model = ChatPollinations(
    model="glm",
)

agent = create_agent(
    model=model,
    system_prompt="Eres un asistente que informa del clima",
)


# Custom streaming: las tools pueden mandar información al agente antes de devolver la observación al modelo.
def get_weather(city: str) -> str:
    """Get weather for a given city"""
    # Inyecta texto en la respuesta del agente
    writer = get_stream_writer()
    writer(f"Buscando el clima para {city}")  # Info inyectada al flujo del agente
    writer(f"Obtenidos los datos para {city}")  # Info inyectada al flujo del agente
    return f"Siempre está soleado en {city}"  # Observación devuelta al modelo


agent = create_agent(
    model=model,
    tools=[get_weather],
)

print()

# Obtiene los mensajes producidos en cada paso por el agente, y los que inyecta el tool
for stream_type, payload in agent.stream(
    {"messages": [{"role": "user", "content": "Cuál es el clima en Lisboa?"}]},
    stream_mode=["values", "custom"],
):
    if stream_type == "values":
        print(payload["messages"][-1].content)  # El payload es una lista con objetos Message
    elif stream_type == "custom":
        print(payload)  # El payload es un simple string
    else:
        print(stream_type, payload)

print()

# Obtiene solo los mensajes inyectados por el tool
for text in agent.stream(
    {"messages": [{"role": "user", "content": "Cuál es el clima en Rio do Janeiro?"}]},
    stream_mode=["custom"]
):
    print(text)

print()
