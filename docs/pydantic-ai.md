# 集成 PydanticAI

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,pydantic-ai]"
```

按[云端资源准备](../examples/README.md#云端资源准备)配置资源并替换示例名称。代码中的 `core` 是异步函数中已初始化的 `AsyncAgentCore` 实例。

```python
from pydantic_ai import Agent

from agentcore.integrations.pydantic_ai import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
agent = Agent(
    model("test-mc", model_name="qwen3.8-max"),
    tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
)
result = await agent.run("按 Skill 的说明完成一个示范")
print(result.output)
```

在仓库根目录运行 [PydanticAI 示例](../examples/frameworks/pydantic_ai_agent.py)。

```bash
python -m examples.frameworks.pydantic_ai_agent
```
