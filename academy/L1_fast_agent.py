import dotenv

from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations

db = SQLDatabase.from_uri("sqlite:///Chinook.db")


@dataclass
class RuntimeContext:
    db: SQLDatabase


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query and return results"""
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


system_prompt = """Eres un cuidadoso analista SQLite

Reglas:
- Piensa step by step.
- Cuando necesites datos, llama a la herramienta `execute_sql` con UN query SELECT.
- Solo Read-only. Nada de INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limita a 5 filas de salida a menos que el usuario explicitamente solicite otra cosa.
- Si la herramienta retorna 'Error:', revisa el SQL y trata nuevamente.
- Preferentemente usar lista explicita de columnas; evitar SELECT *.
"""

dotenv.load_dotenv()

model = ChatPollinations(
    model="glm",
)

agent = create_agent(
    model=model,
    tools=[execute_sql],
    system_prompt=system_prompt,
    context_schema=RuntimeContext,
)

# question = "Cual tabla tiene el mayor número de registros?"
question = "Cual género en promedio tiene las canciones más largas?"

for step in agent.stream(
    {"messages": question},
    context=RuntimeContext(db=db),
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
