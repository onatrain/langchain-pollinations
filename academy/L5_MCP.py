"""
MCP Tools
Model Context Protocol es un estandar abierto para que tools externas y fuentes de datos se conecten y comuniquen
con un agente. El MCP Server debe estar corriendo en algún servidor local o externo para poder acceder sus tools.
"""

import asyncio
import nest_asyncio
import dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import BaseTool
from langchain_pollinations import ChatPollinations
from langchain.agents import create_agent


async def start_server() -> list[BaseTool]:
    # Conecta al MCP-server para operaciones relacionadas con el timezone
    # Provee tools para hora actual, parseo relativo de hora, conversión de timezone
    # aritmética de duración y comparación de tiempo.
    mcp_client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@theo.foobar/mcp-time"],  # Si la librería no está instalada, la descarga e instala.
            }
        },
    )

    mcp_tools = await mcp_client.get_tools()

    print(f"Cargadas {len(mcp_tools)} MCP Tools: {[t.name for t in mcp_tools]}")

    # Devuelve la lista de tools que provee el MCP Server
    return mcp_tools


async def main():
    mcp_tools = await start_server()

    model = ChatPollinations(
        model="kimi",
    )

    agent = create_agent(
        model=model,
        tools=mcp_tools,
        system_prompt="Eres un asistente muy útil",
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Qué hora es en Oporto?"}]},
    )

    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    dotenv.load_dotenv()

    nest_asyncio.apply()
    asyncio.run(main())

