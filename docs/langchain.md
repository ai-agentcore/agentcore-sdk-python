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

## 自定义工具参数

使用 `agentcore.BaseModel` 定义 LangChain／LangGraph 工具的 `args_schema` 时，设置 `serialize_by_alias=False`，使工具按 Python 字段名传参；输入仍支持下划线字段名和驼峰别名。

```python
from langchain_core.tools import tool
from pydantic import ConfigDict

from agentcore import BaseModel


class CharacterArgs(BaseModel):
    model_config = ConfigDict(serialize_by_alias=False)

    character_id: str


@tool(args_schema=CharacterArgs)
def describe_character(character_id: str) -> str:
    """返回指定的角色标识。"""
    return f"Character: {character_id}"


print(describe_character.invoke({"characterId": "alice"}))
print(describe_character.invoke({"character_id": "bob"}))
```

将 `describe_character` 加入 `create_agent` 或 `ToolNode` 的工具列表即可使用。需要向外输出驼峰 JSON 时，仍可对参数模型调用 `model_dump(by_alias=True)`。

此配置仅作用于该工具参数模型；其他业务模型继续默认输出驼峰。示例无需云端资源，安装 `alibabacloud-agentcore-sdk[langchain]` 后，在仓库根目录运行：

```bash
python -m examples.frameworks.langchain_tool_model
```

完整代码见 [工具参数模型示例](../examples/frameworks/langchain_tool_model.py)。

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
