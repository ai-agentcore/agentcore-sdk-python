# 框架执行事件

资源适配（模型、工具、记忆）和执行事件适配是两个入口。向 Server 只返回字符串，只能得到文本；需要展示工具调用过程时，把框架的完整事件交给对应的 `AgentCoreConverter`。每次调用创建一个转换器，不跨请求共享。

可直接运行的完整示例见 [LangChain Agent 服务](../langchain_server.py)，包含资源创建、生命周期和协议输出；启动及调用命令见 [示例指南](../README.md#调用-agent-服务)。

## LangChain / LangGraph

```python
from agentcore.integrations.langchain import AgentCoreConverter

@server.invoke
async def invoke(request, context):
    events = agent.astream_events(
        {"messages": [(m.role.value, m.content) for m in request.messages]},
        version="v2",
    )
    async for event in AgentCoreConverter().stream(events):
        yield event
```

LangGraph 使用同一个转换器，也可从 `agentcore.integrations.langgraph` 导入。输入为 `astream_events(version="v2")`，不是 `astream(stream_mode="updates")`。调用参数在模型一轮完成后输出一次，工具结果使用 `ToolMessage.tool_call_id`，不是 callback 的 `run_id`。Python 3.10 的自定义异步图节点须向模型调用显式传递 `RunnableConfig`，确保框架传递 callbacks。内置 `create_agent` 的完整服务示例要求 Python 3.11+，版本限制见 [示例指南](../README.md#选择示例)。

## 其他框架

| 框架导入 | 交给转换器的输入 |
| --- | --- |
| `agentcore.integrations.agentscope` | AgentScope 2 的 `agent.reply_stream(...)`；最终 `Msg` 快照不会重复输出 |
| `agentcore.integrations.google_adk` | `runner.run_async(...)` 的完整 Event；不能只选 `partial` 文本 |
| `agentcore.integrations.pydantic_ai` | `agent.run_stream_events(...)`；不能用 `stream_text()` 代替完整执行流 |
| `agentcore.integrations.crewai` | `converter.run(crew, inputs)`，或按发送顺序提供 CrewAI event-bus 事件 |

AgentScope / ADK 用法与上述 `converter.stream(events)` 相同。PydanticAI：

```python
from agentcore.integrations.pydantic_ai import AgentCoreConverter

async with agent.run_stream_events(user_input) as events:
    async for event in AgentCoreConverter().stream(events):
        yield event
```

若所用 PydanticAI 版本只有 `run(event_stream_handler=...)`，在 handler 中调用 `converter.convert(event)`；不要只转换最终 result。

CrewAI：

```python
from agentcore.integrations.crewai import AgentCoreConverter

# 每次请求新建 Crew / Task；不要并发复用同一个 Crew。
# 使用普通执行模式（不要设置 stream=True）。
async for event in AgentCoreConverter().run(crew, inputs):
    yield event
```

CrewAI 的普通 StreamChunk 没有工具结果。上述入口收集当前 Crew 的执行事件，并在执行完成后按 `emission_sequence` 输出完整轨迹，因此不是逐 token 实时流。工具调用使用执行事件 ID，结果通过 `started_event_id` 关联；不是模型服务商的调用 ID。该入口需要 CrewAI 1.15.20+。

ReAct 的 `step_callback` 也可调用 `converter.convert(step)`，但它是步骤级输出，且不能覆盖 CrewAI 原生 function calling；不要混用两种输入，也不要从 ReAct 正文解析工具。

## 协议表达范围

| 内容 | AG-UI | OpenAI Chat Completions |
| --- | --- | --- |
| 过程文本、最终文本 | 独立的 TEXT_MESSAGE_START / CONTENT / END，保留消息边界 | 同一 completion 的 content 增量，最终聚合到一条 assistant 消息 |
| 工具调用 | TOOL_CALL_START / ARGS / END | delta.tool_calls（按 ID / index 聚合） |
| 工具结果 | TOOL_CALL_RESULT，与调用 ID 关联 | 不输出；不是该响应协议的标准字段 |
| 框架显式 reasoning | reasoning 事件 | reasoning_content 扩展字段 |

需要完整执行过程的客户端请使用 AG-UI。普通过程说明仍是 TEXT，不能为了展示到 Thread 而改标为 REASONING。转换器保留边界与调用关系，不根据“我先查一下”等正文猜测过程或最终答案。消息在主时间线还是 Thread 展示，由消费方基于结构化事件及自己的展示策略决定。

LangChain / LangGraph 会转发框架已经处理并返回的错误 ToolMessage，保留原始调用 ID，不改变框架的错误处理策略。并行模型调用的文本分别保留消息身份；同一次调用中交替出现的文本和 reasoning 各自形成消息段。

自定义执行器可直接输出 `agentcore.events.AgentEvent`。TEXT / REASONING 可带 `message_id`；带 ID 的消息由相同 ID 的 `TEXT_END` / `REASONING_END` 结束，允许交错输出。不带 ID 的 END 结束该类型的所有当前消息；流正常完成时也会关闭剩余消息。无需依赖任何特定框架。
