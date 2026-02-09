"""
Streaming
- Modo messages: la app recibe token por token a medida que el modelo los produce.
- Modo values:   el modelo comunica estado a medida que cumple pasos o tareas.
- Modo custom:   para inyectar tokens externamente a la respuesta del agente.
Se pueden combinar varios modos de streaming.
Los tokens producidos por un tool no forman parte de la observación devuelta al modelo. Son
inyectados directamente a la respuesta del agente.
"""

import dotenv

from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(
    model="kimi",
)

agent = create_agent(
    model=model,
    system_prompt="Eres el famoso escritor Pablo Neruda y compones poemas de 1 estrofa",
)

# Values: devuelve respuesta a medida que completa pasos
for step in agent.stream(
    {"messages": [{"role": "user", "content": "Escribe un poema sobre una mujer"}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()

print()
