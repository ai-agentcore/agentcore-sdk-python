# 集成 CrewAI

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,crewai]"
```

按[云端资源准备](../examples/README.md#云端资源准备)替换模型连接、模型、MCP 和 Skill 名称。以下代码在使用 `AsyncAgentCore` 的异步函数中运行，`core` 为已初始化的 SDK 实例。

```python
from crewai import Agent, Crew, Task

from agentcore.integrations.crewai import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
agent = Agent(
    role="资源助手",
    goal="使用 Skill 和 MCP 完成请求",
    backstory="根据工具返回的信息回答",
    llm=model("test-mc", model_name="qwen3.8-max"),
    tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
)
task = Task(
    description="按 Skill 的说明完成一个示范",
    expected_output="简短的执行结果说明",
    agent=agent,
)
result = await Crew(agents=[agent], tasks=[task]).akickoff()
print(result.raw)
```

使用异步的 `akickoff()` 执行带有 SDK 工具的 Crew。在仓库根目录运行 [CrewAI 示例](../examples/frameworks/crewai_agent.py)。

```bash
python -m examples.frameworks.crewai_agent
```
