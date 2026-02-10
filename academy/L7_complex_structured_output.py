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

from typing import TypedDict, Any

from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations
from typing import List
from pydantic import BaseModel, Field, constr, conint

dotenv.load_dotenv()


class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str


model = ChatPollinations(
    model="glm",
)

recorded_conversation = """Se habló con Frank Harris. Él quiere que le envíen cuatro
discos de speed metal. Los quiere de Abbatoir, Razor, INC y Slayer.
Su número es cuatro cinco cinco, ocho nueve uno dos nueve ocho siete. Mándale mientras tanto
fotos de Lita Ford a harris@crazybad.com."""


class Item(BaseModel):
    product: constr(min_length=1) = Field(..., description="Nombre del producto")
    quantity: conint(ge=1) = Field(..., description="Cantidad, entero >= 1")
    artist: List[constr(min_length=1)] = Field(..., description="Artistas solicitados (strings)")


class OrderModel(BaseModel):
    order: List[Item] = Field(..., description="Lista de items. Cada item contiene product, quantity y artist")


agent = create_agent(
    model=model,
    response_format=OrderModel
)

result = agent.invoke(
    {"messages": [
        {"role": "user", "content": recorded_conversation}
    ]},
)

print(result["structured_response"].model_dump())
