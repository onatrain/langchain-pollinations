"""
A veces, los agentes necesitan de intervención humana para resolver situaciones. Con ello se puede añadir, por
ejemplo, una interrupción en after_model que se dispara cuando se invoca a un tool específico.
Cuando se define un agente, se puede especificar sobre cuáles tools se necesita human feedback. Cuando una de
estas tools es llamada, se genera una interrupción pidiendo una respuesta humana. Se pueden establecer variados
tipos de respuestas permitidos, como aprobaciones, rechazos, o ediciones.
"""

import dotenv
from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain.agents import create_agent
from langchain_pollinations import ChatPollinations
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

dotenv.load_dotenv()

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


system_prompt = """Tu eres un cuidadoso analista SQLite.

Reglas:
- Piensa step by step.
- Cuando necesites datos, llama a la herramienta `execute_sql` con UN query SELECT.
- Solo Read-only. Nada de INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limita a 5 filas de salida a menos que el usuario explicitamente solicite otra cosa.
- Si la herramienta retorna 'Error:', revisa el SQL y trata nuevamente.
- Preferentemente usar lista explicita de columnas; evitar SELECT *.
- AVERIGUA el esquema de la base de datos si un query falla por tabla o columna inexistente.
- NUNCA preguntes al usuario por nombres de tablas o campos. AVERIGUALO por tu cuenta lanzando queries a la base de datos.
- NUNCA olvides responder la pregunta original que se te hizo.
"""

model = ChatPollinations(
    model="kimi",
)

agent = create_agent(
    model=model,
    tools=[execute_sql],
    context_schema=RuntimeContext,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}}
    )]
)

question = "Dame los nombres de todos los empleados."
config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    context=RuntimeContext(db=db),
)

if "__interrupt__" in result:
    description = result["__interrupt__"][-1].value["action_requests"][-1]['description']
    print("*#" * 50)
    print(f"Interrupción:  {description}")

    result = agent.invoke(
        Command(
            resume={
                "decisions": [{
                    "type": "approve"
                }]
            }
        ),
        config=config,  # Same thread_id to resume the paused conversation
        context=RuntimeContext(db=db),
    )
    print("*#" * 50)

for msg in result["messages"]:
    msg.pretty_print()
    print("Reason:", msg.response_metadata)
