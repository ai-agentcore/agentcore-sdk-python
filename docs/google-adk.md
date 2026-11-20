# 集成 Google ADK

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,google-adk]"
```

按[云端资源准备](../examples/README.md#云端资源准备)替换模型连接、模型、MCP 和 Skill 名称。以下代码在使用 `AsyncAgentCore` 的异步函数中运行，`core` 为已初始化的 SDK 实例。

```python
from google.adk.agents import Agent

from agentcore.integrations.google_adk import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
agent = Agent(
    name="resource_assistant",
    model=model("test-mc", model_name="qwen3.8-max"),
    tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
)
```

用 ADK 的会话服务创建会话，将 `agent` 和会话服务传给 `Runner`，再通过 `runner.run_async(...)` 读取响应事件。可在仓库根目录运行 [Google ADK 示例](../examples/frameworks/google_adk_agent.py)。

```bash
python -m examples.frameworks.google_adk_agent
```
