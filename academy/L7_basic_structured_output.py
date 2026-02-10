"""
Si los agentes van a integrarse con sistemas informáticos, entonces deben ser capaces de generar información siguiendo
una estructura o formato definido. Esto puede ser útil para aplicaciones que, por ejemplo, toman una conversación
escrita y extraen tópicos importantes de ella.
Los agentes soportan los siguientes tipos de formatos:
- pydantic BaseModel
- TypedDict
- dataclasses
- json schema (dictionary)
"""
import dotenv

from typing import TypedDict

from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()


model = ChatPollinations(
    model="deepseek",
)

recorded_conversation = """Se habló con Frank Harris. Él quiere que le envíen cuatro
discos de speed metal. Los quiere de Abbatoir, Razor, INC y Slayer.
Su número es cuatro cinco cinco, ocho nueve uno dos nueve ocho siete. Mándale mientras tanto
fotos de Lita Ford a harris@crazybad.com."""


class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str


agent = create_agent(
    model=model,
    response_format=ContactInfo
)

result = agent.invoke(
    {"messages": [
        {"role": "user", "content": recorded_conversation}
    ]},
)

print(result["structured_response"])
