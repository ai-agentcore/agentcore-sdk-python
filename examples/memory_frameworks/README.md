# Memory 框架示例

为 Agent 添加长期记忆：在回答前检索相关信息，在完成对话后保存需要记住的内容。

示例使用已有的 MemoryStore 和框架模型对象。运行前请在 AgentCore 中创建记忆空间并授予应用访问权限。

## 选择框架

示例以以下版本验证，安装时请同时满足对应 SDK extra 的依赖要求：

| 框架 | 示例 | 验证版本 |
| --- | --- | --- |
| LangChain | [Middleware](langchain_agent.py) | 1.4.0 |
| LangGraph | [记忆节点](langgraph_agent.py) | 1.2.11 |
| AgentScope | [Middleware](agentscope_agent.py) | 2.0.7，Python 3.11+ |
| Google ADK | [Memory Service](google_adk_agent.py) | 2.8.0 |
| PydanticAI | [Memory Capability](pydantic_ai_agent.py) | 2.31.1 |
| CrewAI | [Task Memory](crewai_agent.py) | 1.15.17 |

这些示例使用异步执行。模型与工具的框架示例见 [示例指南](../README.md)。

## 使用示例

以 LangChain 为例，安装 `alibabacloud-agentcore-sdk[langchain]` 后，在部署到 AgentCore 的应用中调用：

```python
import asyncio

from agentcore import AsyncAgentCore
from agentcore.integrations.langchain import model
from examples.memory_frameworks.langchain_agent import run

chat = model("test-mc", model_name="qwen3.8-max")


async def main():
    async with AsyncAgentCore.auto() as core:
        result = await run(
            core.memory_store("test-memory"),
            chat,
            partition="my-business-partition",
            session_id="session-1",
        )
        print(result["messages"][-1].content)


asyncio.run(main())
```

其他示例使用相同的调用形式：

```python
await run(store, framework_model, partition="...", session_id="...")
```

传入与目标框架匹配的模型对象，并替换资源名称。Memory 适配器不会接管模型客户端的关闭，应用应按框架提供的方式管理其生命周期。

## 选择记忆范围

- `partition` 对应逻辑 `agent_id`，用于隔离不同业务对象的记忆，不必使用平台 Agent ID。
- `session_id` 标识当前会话；示例在分区内检索，写入时附带当前会话。
- 这些值由应用的可信业务上下文提供，不应交给模型生成，也不能替代访问权限控制。

使用 `MemoryScopes` 的适配器也支持独立的 `user_id`、`agent_id` 或 `session_id`，
读写范围均须显式指定至少一个字段；写入不再强制要求会话 ID。例如按用户读写：

```python
from agentcore.integrations.memory import MemoryScopes
from agentcore.memory import MemoryScope

scopes = MemoryScopes(
    read=MemoryScope(user_id="alice"),
    write=MemoryScope(user_id="alice", session_id="session-1"),
)
```

Google ADK 仍使用 `partition_resolver(app_name, user_id)` 映射到 `agent_id`；
`add_events_to_memory` 可省略 `session_id`。

Memory 提供长期记忆，不替代框架的会话历史或检查点存储。

## 保存记忆

示例会产生真实写入，建议使用专门的测试记忆空间。

- LangChain、AgentScope 和 PydanticAI 示例显式开启 `write_back=True`；默认只检索。
- LangGraph 通过显式的检索与记录节点控制读写时机。
- ADK 示例在成功完成后提交本轮事件；仅配置 Memory Service 不会自动保存。
- CrewAI 示例通过 Task callback 写回，使用异步 `akickoff()`，不同时启用 Crew 自带的 `memory=True`。

需要业务审批后再保存时，关闭自动写回并由应用显式写入。记忆写入后可能需要一定时间才能检索到。
