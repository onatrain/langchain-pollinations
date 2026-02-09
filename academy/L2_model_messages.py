import dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_pollinations import ChatPollinations

dotenv.load_dotenv()

model = ChatPollinations(
    model="kimi",
)

# Un agente sin tools es simplemente una interfaz con el modelo
agent = create_agent(
    model=model,
    system_prompt="Eres un gran comediante erótico",
)

# Explícitamente se crea un mensaje de humano y se envia al agente
human_message = HumanMessage("Cómo es la vida sin una mujer?")

result = agent.invoke(
    {"messages": [human_message]}
)

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}\n")

print("*" * 100)

# Si se pasa un string directamente, el agente lo toma como un mensaje de Humano
result = agent.invoke(
    {"messages": "Cuéntame un haiku sobre una mujer"}
)

print(result["messages"][-1].content, "\n")
