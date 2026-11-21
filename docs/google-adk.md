# 集成 Google ADK

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,google-adk]"
```

按[云端资源准备](../examples/README.md#云端资源准备)配置资源并替换示例名称。代码中的 `core` 是异步函数中已初始化的 `AsyncAgentCore` 实例。

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

按[完整示例](../examples/frameworks/google_adk_agent.py)创建会话和 `Runner`，再通过 `runner.run_async(...)` 读取响应。在仓库根目录运行示例。

```bash
python -m examples.frameworks.google_adk_agent
```
