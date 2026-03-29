import asyncio

from agenticcybersense.agents.telegram import TelegramAgent
from agenticcybersense.schemas.messages import AgentRequest


async def main() -> None:
    agent = TelegramAgent()
    req = AgentRequest(query="rce exploit")
    resp = await agent.process(req)
    print("Success:", resp.success)
    print("Findings:", len(resp.findings))
    print("--- Report ---")
    print(resp.content)


if __name__ == "__main__":
    asyncio.run(main())
