"""Matrix room attachments with AgentScope 2: reuse its Read and Write tools.

Install the SDK's agentscope, collaboration and server extras. Configure test-mc
and a vision-capable model to test images. This sample uses a fresh Agent per
request; application session persistence is deliberately not part of the example.
"""

import logging
import os
from pathlib import Path

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionRule
from agentscope.state import AgentState
from agentscope.tool import Read, Toolkit, Write

from agentcore import AsyncAgentCore
from agentcore.integrations.agentscope import AgentCoreConverter, model, skill_tools, tools
from agentcore.server import AgentCoreServer, AgentRequest

logging.basicConfig(level=logging.INFO)
core = AsyncAgentCore.auto()
workspace = Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()
selected = []
worker = None


async def startup():
    global selected, worker
    worker = core.collaboration.worker()
    selected = [*tools(worker.tools()), *skill_tools(worker.skills()), Read(), Write()]


server = AgentCoreServer(
    startup=startup, shutdown=core.aclose, readiness=lambda: worker is not None
)


@server.invoke
async def invoke(request: AgentRequest, _context):
    if worker is None:
        raise RuntimeError("Agent is not initialized")
    toolkit = Toolkit()
    await toolkit.add_tool(selected)
    # Use the framework's existing permission mechanism. In production select
    # the application's sandbox/backend; a path pattern is not an OS sandbox.
    permissions = PermissionContext(
        allow_rules={
            tool.name: [
                PermissionRule(
                    tool_name=tool.name,
                    rule_content=str(workspace / "**") if tool.name in {"Read", "Write"} else None,
                    behavior=PermissionBehavior.ALLOW,
                    source="application",
                )
            ]
            for tool in selected
        }
    )
    agent = Agent(
        name="room_file_assistant",
        model=model("test-mc", model_name=os.environ.get("EXAMPLE_MODEL_NAME", "qwen3.8-max")),
        system_prompt=worker.compose_prompt(
            "按用户要求处理聊天附件。下载后用 Read 读取；需要交付文件时用 Write 生成到工作目录，"
            "再调用发送工具。不要把下载成功当作理解内容，也不要只返回本地路径。"
        ),
        toolkit=toolkit,
        state=AgentState(permission_context=permissions),
    )
    # Matrix forwarder supplies one user message; Server binds context for the
    # complete stream. Image decoding belongs to Read + the selected model.
    message = next(
        message for message in reversed(request.messages) if message.role.value == "user"
    )
    async for event in AgentCoreConverter().stream(
        agent.reply_stream(UserMsg("user", message.content))
    ):
        yield event
