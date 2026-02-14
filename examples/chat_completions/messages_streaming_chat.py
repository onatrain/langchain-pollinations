import dotenv

from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(
    model="minimax",  # ejemplo de modelo de texto
    temperature=1,
    max_tokens=1000,  # CAUTION: algunos modelos en Pollinations imponen un mínimo de tokens
)

for token in model.stream(
    input=[HumanMessage(content="Explica en cuatro párrafos qué es la ciencia")],
    stream_mode="messages",
):
    print(token.content, end="", flush=True)
