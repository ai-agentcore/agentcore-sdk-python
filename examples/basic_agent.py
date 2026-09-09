"""Minimal AG-UI / OpenAI Agent service using an AgentCore managed model."""

from __future__ import annotations

from agentcore import AsyncAgentCore, RequestContext
from agentcore.model import AsyncModelClient
from agentcore.server import AgentCoreServer, AgentRequest

core: AsyncAgentCore | None = None
model_client: AsyncModelClient | None = None


async def startup() -> None:
    global core, model_client
    core = AsyncAgentCore.auto()
    model_client = await core.model("test-mc", model="qwen3.8-max")


async def shutdown() -> None:
    if core is not None:
        await core.aclose()


server = AgentCoreServer(
    readiness=lambda: core is not None and model_client is not None,
    startup=startup,
    shutdown=shutdown,
)


@server.invoke
async def invoke(request: AgentRequest, _context: RequestContext) -> str:
    if core is None or model_client is None:
        raise RuntimeError("AgentCore SDK is not started")
    messages = [
        {"role": message.role.value, "content": message.content}
        for message in request.messages
    ]
    response = await model_client.invoke(messages)
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")
