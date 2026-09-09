# AgentCore SDK for Python

使用 Python 构建 Agent，连接 AgentCore 云端资源或自有服务，并与常用 Agent 框架集成。

[AgentCore 官网](https://www.aliyun.com/product/agentcore) · [官方文档](https://help.aliyun.com/zh/agentcore/) · [示例指南](examples/README.md)

## 从这里开始

| 你想做什么 | 入口 |
| --- | --- |
| 连接平台上的模型、MCP 和 Skill | [云端资源](#使用云端资源) |
| 使用自己的模型、MCP 或本地 Skill | [自定义资源](#使用自定义资源) |
| 为 Agent 添加长期记忆 | [Memory](#使用-memory) |
| 配置 MCP 凭证与固定 Header | [凭证与 Header](#凭证与-mcp-header) |
| 接入已有 Agent 框架 | [框架集成](#框架集成) |
| 将 Agent 发布为 HTTP 服务 | [服务协议](#提供-agent-服务) |

## 功能

- **模型**：访问托管模型，或连接自定义模型服务。
- **工具与 Skill**：接入 MCP 服务，使用云端或随应用分发的 Skill。
- **记忆**：检索和保存长期记忆，接入框架的 Agent 执行流程。
- **凭证**：按名称获取托管 API Key，或为 MCP 显式绑定 Header 凭证。
- **服务协议**：通过 AG-UI 或 OpenAI Chat Completions 提供 Agent 服务。

支持 LangChain、LangGraph、AgentScope、Google ADK、PydanticAI 和 CrewAI。

## 安装

需要 Python 3.10+；AgentScope 2.x 需要 Python 3.11+。

> 当前分支处于公开发布准备阶段。以下为公开发行版的安装方式，包的可用版本以正式发布为准。

```bash
pip install alibabacloud-agentcore-sdk

# 按需安装 MCP、服务端和框架支持
pip install "alibabacloud-agentcore-sdk[mcp,server,langchain]"
```

按使用场景选择 extras：`mcp`、`server`、`credentials`、`langchain`、`langgraph`、`agentscope`、`google-adk`、`pydantic-ai`、`crewai`。

## 快速开始

### 使用云端资源

先参考[官方文档](https://help.aliyun.com/zh/agentcore/)创建 Workspace，再按需准备模型连接、MCP 和 Skill，并为 Agent 授予访问权限。以下代码在部署到 AgentCore 的应用中运行；将资源名称替换为当前 Workspace 中自己的资源名称。只使用模型时，无需创建 MCP 或 Skill。

```python
import asyncio

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore.auto() as core:
        model = await core.model("test-mc", model="qwen3.8-max")
        response = await model.invoke([{"role": "user", "content": "你好"}])
        print(response["choices"][0]["message"]["content"])

        mcp = await core.mcp("test-mcp")
        print([tool.name for tool in await mcp.list_tools()])

        skill = await core.skills.managed("test-skill")
        print(skill.name)


asyncio.run(main())
```

### 凭证与 MCP Header

在平台创建凭证并授权后，使用 `await core.credentials.get(name)` 按名称获取。API Key 凭证通过 `credential.value` 读取；MCP Header 凭证通过 `credential.as_headers()` 读取。不要将凭证值写入日志或提交到代码仓库。

托管 MCP 可在创建时显式绑定 MCP Header 凭证，并追加固定 Header。只有该 MCP 在凭证允许的应用范围内时才能使用：

```python
mcp = await core.mcp(
    "test-mcp",
    credential_name="my-mcp-credential",  # 可省略；须在平台创建并允许访问该 MCP
    headers={"x-business-id": "my-app"},  # 可省略
)
```

Header 作用于该客户端的握手、工具发现和工具调用，不是请求级身份上下文。
SDK 会复制配置，不同 Header 配置使用不同客户端和 Session；不传时保持原有行为。
合并顺序为平台 Header → 凭证 Header → 自定义 Header；名称大小写不敏感，冲突直接报错，不做覆盖。
`Authorization`、平台已配置的 Header，以及 `Host`、`Mcp-Session-Id`、`Mcp-Protocol-Version` 等协议维护字段不可覆盖。
凭证在首次建连和重连时查询与获取，复用 Session 期间不重复获取，也不后台刷新。
托管 MCP 使用 Streamable HTTP；直连 MCP 保留 `headers_provider`，stdio 不承载 HTTP Header。
不要修改共享客户端来切换用户身份。Header 的内容由业务明确选择，不自动透传入站请求头。
固定 Header 不等于可信身份。完整示例见 [MCP 凭证与 Header](examples/mcp_credentials.py)。

### 使用自定义资源

连接自有模型、MCP 服务，无需在 AgentCore 注册资源：

```python
import asyncio
import os

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore() as core:
        model = core.direct_model(
            provider="openai",
            model=os.environ["CUSTOM_MODEL_NAME"],
            base_url=os.environ["CUSTOM_MODEL_BASE_URL"],
            api_key=os.environ["CUSTOM_MODEL_API_KEY"],
        )
        response = await model.invoke([{"role": "user", "content": "你好"}])
        print(response["choices"][0]["message"]["content"])

        mcp = core.direct_mcp(url=os.environ["CUSTOM_MCP_URL"])
        print([tool.name for tool in await mcp.list_tools()])


asyncio.run(main())
```

本地 Skill 加载方式及完整配置见 [自定义资源示例](examples/direct_resources.py)。

## 使用 Memory

Memory 用于保存和检索长期记忆，也支持查询、更新、删除记忆以及查看会话消息。使用前，在 AgentCore 中创建 MemoryStore 并授予应用访问权限。

```python
import asyncio

from agentcore import AsyncAgentCore
from agentcore.memory import MemoryScope


async def main():
    async with AsyncAgentCore.auto() as core:
        store = core.memory_store("test-memory")
        await store.add_memories(
            scope=MemoryScope(user_id="example-user", session_id="session-1"),
            text="用户喜欢简短的中文回答。",
        )

        # Newly added memories may take time to become searchable.
        result = await store.search_memories(
            "用户有哪些回答偏好？",
            scope=MemoryScope(user_id="example-user"),
            top_k=5,
        )
        for hit in result.memories:
            print(hit.memory.content.text)


if __name__ == "__main__":
    asyncio.run(main())
```

- `user_id`、`agent_id`、`session_id` 是彼此独立的记忆范围字段；写入时都可省略，服务端会将未传字段归入默认范围。
- 查询时只传需要精确匹配的字段，未传字段按通配范围查询；响应中的默认范围字段会表现为 `None`。
- `list_memory_session_messages` 要求 `session_id`，并且 `user_id`、`agent_id` 至少传一个。
- 分区标识由可信业务上下文提供，不替代访问权限控制。

示例会写入真实数据，建议使用测试记忆空间。新写入的记忆可能需要一定时间才能检索到，首次查询可能为空。

完整代码见 [Memory 基础示例](examples/memory.py)；让 Agent 自动检索并使用记忆，参见 [六框架 Memory 示例](examples/memory_frameworks/README.md)。

## 框架集成

将模型、MCP 和 Skill 接入现有框架，不必重写 Agent 的业务逻辑。

| 框架 | 示例 |
| --- | --- |
| LangChain | [Agent](examples/frameworks/langchain_agent.py) |
| LangGraph | [模型与工具循环](examples/frameworks/langgraph_agent.py) |
| AgentScope 2.x | [Agent](examples/frameworks/agentscope_agent.py) |
| Google ADK | [Agent 与 Runner](examples/frameworks/google_adk_agent.py) |
| PydanticAI | [Agent](examples/frameworks/pydantic_ai_agent.py) |
| CrewAI | [Agent、Task 与 Crew](examples/frameworks/crewai_agent.py) |

需要长期记忆时，参见 [Memory 框架示例](examples/memory_frameworks/README.md)。

需要展示工具执行过程时，使用 [框架执行事件转换器](examples/frameworks/execution-events.md)，保留工具调用、结果与消息边界。

## 提供 Agent 服务

`AgentCoreServer` 支持 AG-UI、OpenAI Chat Completions 和自定义 `ProtocolHandler`。

```bash
uvicorn examples.basic_agent:server --host 0.0.0.0 --port 8080
```

默认接口为 `POST /ag-ui/agent` 和 `POST /openai/v1/chat/completions`，支持流式响应。参见 [最小服务示例](examples/basic_agent.py)。

需要完整工具执行流程时，运行 [LangChain 服务示例](examples/langchain_server.py)：它将托管模型、MCP、Skill、框架事件转换器和两个协议入口串起来。此示例使用 Python 3.11+；安装 `alibabacloud-agentcore-sdk[mcp,server,langchain]` 后，在云端应用中执行：

```bash
uvicorn examples.langchain_server:server --host 0.0.0.0 --port 8080 --log-level info
```

示例使用 OpenAI/v1 模型连接，每次处理最新一条用户文本，不保存会话历史。AG-UI/OpenAI 请求及预期输出见 [调用 Agent 服务](examples/README.md#调用-agent-服务)。

框架执行流应使用对应的[事件转换器](examples/frameworks/execution-events.md)，不要只提取文本。AG-UI 可表达消息边界、工具调用和工具结果；OpenAI Chat Completions 按其标准表达文本和工具调用，不提供独立的工具结果流式事件。会话历史的保存与恢复仍由应用或所用框架负责。

## 使用提示

- 在应用生命周期内复用 Core；异步应用使用 `async with`，同步应用可使用 `with AgentCore.auto()`。
- 普通同步调用和异步文本流见 [模型调用示例](examples/model_calls.py)；Responses 用法见 [示例指南](examples/README.md#模型调用方式)。
- 模型的工具调用、Responses 和 Embedding 支持情况取决于所选模型。
- 仅加载可信 Skill；不需要执行命令时设置 `ALLOW_EXECUTE_COMMAND=false`。
- 使用 `logging.basicConfig(level=logging.INFO)` 开启日志，不要记录 API Key 等敏感信息。

更多资源配置、框架依赖与运行命令见 [示例指南](examples/README.md)。
