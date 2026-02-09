"""
Aunque frecuentemente una descripción básica es suficiente, LangChain tiene soporte para descripciones extendidas.
Cuando el tool posee descripciones Google Style y se suministra el parámetro parse_docstring=True, se pasarán
las descripciones de argumentos al modelo. También se puede renombrar el tool y cambiar su descripción. Esto
puede ser efectivo cuando se comparte un tool estándar pero se necesitan instrucciones específicas para el agente.
"""

import dotenv

from typing import Literal

from langchain.tools import tool
from langchain_pollinations import ChatPollinations
from langchain.agents import create_agent

dotenv.load_dotenv()


@tool(
    "calculator",  # Nombre que usará el modelo para conocer al tool
    parse_docstring=True,
    description=(
        "Ejecuta operaciones aritméticas básicas sobre dos números reales."
        "Usar esto siempre que tengas operaciones sobre cualesquiera números, aún si son enteros."
    )
)
def real_number_calculator(a: float, b: float, operation: Literal['add', 'subtract', 'multiply', 'divide']) -> float:
    """Realiza operaciones aritméticas básicas sobre dos números reales

    Args:
        a (float): El primer número.
        b (float): El segundo número.
        operation (Literal['add', 'subtract', 'multiply', 'divide']): La operación aritmética a ejecutar.

        - `"add"`: Retorna la suma de `a` y `b`.
        - `"subtract"`: Retorna el resultado de `a - b`.
        - `"multiply"`: Retorna el producto de `a` y `b`.
        - `"divide"`: Retorna el resultado de `a / b`. Lanza un error si `b` es cero.

    Returns:
        float: el resultado numérico de la operación especificada.

    Raises:
        ValueError: Si se provee una operación inválida o si se intenta una división por cero.
    """
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
        case _:
            raise ValueError(f"Operación inválida: {operation}")


model = ChatPollinations(
    model="kimi",
)

agent = create_agent(
    model=model,
    tools=[real_number_calculator],
    system_prompt="Eres un asistente muy útil",
)

for step in agent.stream(
    {"messages": [{"role": "user", "content": "Cuanto es 3 * 4?"}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()

print()
