import dotenv

from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations
from langchain.tools import tool

dotenv.load_dotenv()

model = ChatPollinations(
    model="glm",
)


@tool
def check_haiku_lines(text: str) -> str:
    """
    Chequea si el texto dado de un haiku tiene exactamente 3 lineas

    :param text: Texto del haiku a chequear
    :return: Mensaje que indica si el haiku es correcto o incorrecto.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    print(f"checking haiku, it has {len(lines)} lines: \n{text}")

    if len(lines) != 3:
        return f"Incorrect! This haiku has {len(lines)} lines. A haiku must have exactly 3 lines."
    return "Correct, this haiku has 3 lines."


agent = create_agent(
    model=model,
    tools=[check_haiku_lines],
    system_prompt="Eres William Shakespeare y solo escribes haikus. Siempre verificas tu trabajo.",
)

# Envío explícito de mensaje textual de humano
result = agent.invoke(
    {"messages": {"role": "user", "content": "Escribe un haiku sobre el baseball"}}
)

for msg in result["messages"]:
    msg.pretty_print()

print("\n", result["messages"][-1].response_metadata, "\n")
