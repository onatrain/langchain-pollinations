import dotenv

from langchain_core.messages import HumanMessage

from utils import convert_to_json
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(
    model="mistral",  # ejemplo de modelo de texto
    temperature=1,
    max_tokens=100,  # CAUTION: algunos modelos en Pollinations imponen un mínimo de tokens
)

res = model.invoke([HumanMessage(content="Explica en un párrafo qué es la ciencia")])
print("Tipo de la respuesta:")
print(type(res), "\n")

print("Respuesta completa:")

print(convert_to_json(res.model_dump()), "\n")
