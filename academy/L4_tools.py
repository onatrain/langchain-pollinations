import dotenv

from typing import Literal

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations

valid_operation = Literal['add', 'subtract', 'multiply', 'divide']


@tool
def simple_calculator(a: float, b: float, operation: valid_operation) -> float:
    """Realiza operaciones aritméticas básicas sobre dos números reales"""
    print(f"Ejecutando la operación aritmética: {operation}")

    match(operation):
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            if b == 0:
                raise ValueError("La división entre cero no está definida.")
            return a / b


dotenv.load_dotenv()

model = ChatPollinations(
    model="openai",
)

agent = create_agent(
    model=model,
    tools=[simple_calculator],
    system_prompt="Eres un asistente muy útil",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Cuanto es 3.141592 * 8.5894?"}]},
)

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
    print("*" * 100)

