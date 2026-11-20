# 集成 AgentScope

AgentScope 2.x 需要 Python 3.11 或更高版本。

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,agentscope]"
```

按[云端资源准备](../examples/README.md#云端资源准备)替换模型连接、模型、MCP 和 Skill 名称。以下代码在使用 `AsyncAgentCore` 的异步函数中运行，`core` 为已初始化的 SDK 实例。

```python
from agentscope.tool import Toolkit

from agentcore.integrations.agentscope import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
chat = model("test-mc", model_name="qwen3.8-max")
toolkit = Toolkit()
await toolkit.add_tool([*tools(await mcp.list_tools()), *skill_tools([skill])])
```

创建 AgentScope 的 `Agent` 时，将 `chat` 和 `toolkit` 分别传给 `model` 和 `toolkit`，并配置所选工具的执行权限。然后用 `await agent.reply(...)` 发送消息。完整的权限配置和调用见 [AgentScope 示例](../examples/frameworks/agentscope_agent.py)，可在仓库根目录运行。

```bash
python -m examples.frameworks.agentscope_agent
```
