# 集成 LangChain 和 LangGraph

按[云端资源准备](../examples/README.md#云端资源准备)配置资源并替换示例名称。代码中的 `core` 是异步函数中已初始化的 `AsyncAgentCore` 实例。

## LangChain

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,langchain]"
```

```python
from langchain.agents import create_agent

from agentcore.integrations.langchain import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
agent = create_agent(
    model=model("test-mc", model_name="qwen3.8-max"),
    tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "按 Skill 的说明完成一个示范"}]}
)
print(result["messages"][-1].content)
```

在仓库根目录运行 [LangChain 示例](../examples/frameworks/langchain_agent.py)。

```bash
python -m examples.frameworks.langchain_agent
```

## LangGraph

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,langgraph]"
```

将同一组工具绑定到模型和 `ToolNode`。

```python
from langgraph.prebuilt import ToolNode

from agentcore.integrations.langgraph import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
agent_tools = [*tools(await mcp.list_tools()), *skill_tools([skill])]
chat = model("test-mc", model_name="qwen3.8-max").bind_tools(agent_tools)
tool_node = ToolNode(agent_tools)
```

在仓库根目录运行 [LangGraph 示例](../examples/frameworks/langgraph_agent.py)，查看完整的图和调用方式。

```bash
python -m examples.frameworks.langgraph_agent
```
