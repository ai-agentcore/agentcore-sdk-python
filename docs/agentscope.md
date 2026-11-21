# 集成 AgentScope

AgentScope 2.x 需要 Python 3.11 或更高版本。

```bash
python -m pip install "alibabacloud-agentcore-sdk[mcp,agentscope]"
```

按[云端资源准备](../examples/README.md#云端资源准备)配置资源并替换示例名称。代码中的 `core` 是异步函数中已初始化的 `AsyncAgentCore` 实例。

```python
from agentscope.tool import Toolkit

from agentcore.integrations.agentscope import model, skill_tools, tools

mcp = await core.mcp("test-mcp")
skill = await core.skills.managed("test-skill")
chat = model("test-mc", model_name="qwen3.8-max")
toolkit = Toolkit()
await toolkit.add_tool([*tools(await mcp.list_tools()), *skill_tools([skill])])
```

创建 `Agent` 时传入模型和工具集，按[完整示例](../examples/frameworks/agentscope_agent.py)配置所选工具的权限，再调用 `await agent.reply(...)`。在仓库根目录运行示例。

```bash
python -m examples.frameworks.agentscope_agent
```
