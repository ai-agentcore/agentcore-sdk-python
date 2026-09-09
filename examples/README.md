# 示例指南

从云端资源、自定义服务或你熟悉的框架开始。各示例可以独立阅读和运行。

[SDK 首页](../README.md) · [AgentCore 官网](https://www.aliyun.com/product/agentcore) · [官方文档](https://help.aliyun.com/zh/agentcore/)

## 云端资源准备

首次使用平台可参考[官方文档](https://help.aliyun.com/zh/agentcore/)。按所选示例创建需要的资源，并授予应用访问权限；不必为每个示例准备全部资源。示例中的名称均可替换。

| 资源 | 示例名称 |
| --- | --- |
| 模型连接 | `test-mc` |
| 模型 | `qwen3.8-max` |
| MCP | `test-mcp` |
| Skill | `test-skill` |
| MemoryStore | `test-memory`（Memory 示例需要） |
| MCP Header 凭证 | `my-mcp-credential`（凭证示例需要，应用范围须允许 `test-mcp`） |

框架示例需要模型支持工具调用。请根据 MCP 和 Skill 的用途调整示例中的问题。

## 选择示例

在仓库根目录运行，安装方式见 [README](../README.md)。以下云端示例用于部署在 AgentCore 中的应用。

| 示例 | 所需 extras | 运行命令 |
| --- | --- | --- |
| [云端资源](cloud_resources.py) | `mcp` | `python -m examples.cloud_resources` |
| [MCP 凭证与 Header](mcp_credentials.py) | `mcp,credentials` | `python -m examples.mcp_credentials` |
| [Memory 写入与检索](memory.py) | 无 | `python -m examples.memory` |
| [自定义资源](direct_resources.py) | `mcp` | `python -m examples.direct_resources` |
| [最小 Agent 服务](basic_agent.py) | `server` | `uvicorn examples.basic_agent:server --host 0.0.0.0 --port 8080` |
| [LangChain 完整服务](langchain_server.py)（Python 3.11+） | `mcp,server,langchain` | `uvicorn examples.langchain_server:server --host 0.0.0.0 --port 8080 --log-level info` |
| [模型同步调用与异步流](model_calls.py) | 无 | `python -m examples.model_calls` |
| [LangChain](frameworks/langchain_agent.py) | `mcp,langchain` | `python -m examples.frameworks.langchain_agent` |
| [LangGraph](frameworks/langgraph_agent.py) | `mcp,langgraph` | `python -m examples.frameworks.langgraph_agent` |
| [AgentScope 2.x](frameworks/agentscope_agent.py) | `mcp,agentscope` | `python -m examples.frameworks.agentscope_agent` |
| [Google ADK](frameworks/google_adk_agent.py) | `mcp,google-adk` | `python -m examples.frameworks.google_adk_agent` |
| [PydanticAI](frameworks/pydantic_ai_agent.py) | `mcp,pydantic-ai` | `python -m examples.frameworks.pydantic_ai_agent` |
| [CrewAI](frameworks/crewai_agent.py) | `mcp,crewai` | `python -m examples.frameworks.crewai_agent` |

例如，LangChain 示例需要：

```bash
pip install "alibabacloud-agentcore-sdk[mcp,langchain]"
```

AgentScope 2.x 需要 Python 3.11+。CrewAI 示例使用异步 `akickoff()`，以配合异步 MCP 工具。

LangChain 完整服务示例也使用 Python 3.11+：当前验证的 LangChain 1.4.0 / LangGraph 1.2.11 在 Python 3.10 下运行内置 `create_agent` 时，嵌套模型调用未完整传递事件回调，工具虽执行成功但文本事件可能缺失。Python 3.10 仍可使用基础 SDK 和普通模型调用；不要将普通调用成功等同于完整事件流可用。

## 模型调用方式

[model_calls.py](model_calls.py) 演示同步 `AgentCore` 与异步 `AsyncAgentCore`，均使用 OpenAI/v1 托管模型。同步代码无需手动管理事件循环；异步服务直接使用异步入口。该示例会发起两次真实模型请求。

在 `async with AsyncAgentCore.auto() as core` 内，支持 Responses 的模型还可以这样调用：

```python
model = await core.model("test-mc", model="qwen3.8-max")
response = await model.responses("用一句话介绍杭州。")
print(response["output"])

async for event in model.responses_stream("用一句话介绍杭州。"):
    if event["type"] == "response.output_text.delta":
        print(event["delta"], end="", flush=True)
```

这些是两次独立请求，按需选择一种。Responses 是否可用由连接协议和实际模型决定，SDK 不会在失败后自动切换为 Chat。Embedding 需要使用支持向量生成的模型，不应直接用聊天模型替代。以上文本解析针对 OpenAI 格式，Anthropic 使用其原生响应结构。

## 自定义模型、MCP 和 Skill

运行 [direct_resources.py](direct_resources.py) 前，通过环境变量提供自有服务的配置：

| 变量 | 含义 |
| --- | --- |
| `CUSTOM_MODEL_NAME` | 模型名称 |
| `CUSTOM_MODEL_BASE_URL` | 模型 API Base URL |
| `CUSTOM_MODEL_API_KEY` | 模型 API Key |
| `CUSTOM_MCP_URL` | MCP Endpoint |

这些变量由示例代码读取。OpenAI 兼容服务的 Base URL 通常包含 `/v1`。MCP 默认使用 Streamable HTTP，连接 SSE 服务时设置 `transport="sse"`；需要鉴权时使用 `headers_provider` 提供请求头。

示例附带 [greeting Skill](skills/greeting/SKILL.md)。用 `await core.skills.local(path)` 加载自己的 Skill 目录，再通过框架的 `skill_tools()` 接入 Agent。

云端 Skill 可显式指定版本：

```python
skill = await core.skills.managed("test-skill", version="1.0.0")
```

请将 `1.0.0` 替换为该 Skill 已发布的版本。

## 记忆

- [基础示例](memory.py)：使用 `test-memory` 演示写入和检索，不依赖框架或模型连接。
- [框架示例](memory_frameworks/README.md)：六个框架的记忆检索与写回。

运行前创建 MemoryStore 并授权；示例会写入数据，建议使用专门的测试记忆空间。分区和会话参数见 [Memory 快速开始](../README.md#使用-memory)。写入后可能需要一定时间才能检索到。

## 凭证与 Header

[凭证示例](mcp_credentials.py) 展示两种用法：创建托管 MCP 客户端时显式绑定凭证，以及通过 `core.credentials.get(name)` 单独读取凭证。API Key 使用 `.value`；MCP Header 使用 `.as_headers()`。示例只输出 Header 名称，不输出凭证值。

创建 MCP 时传入的 `headers` 是客户端级固定配置，不用于在共享客户端上逐请求切换用户身份。Header 冲突、保护字段和 Session 的作用范围见 [SDK 说明](../README.md#凭证与-mcp-header)。

## 调用 Agent 服务

服务示例提供 AG-UI 和 OpenAI Chat Completions 接口。向已部署的应用发送请求：

```bash
curl -N 'https://<agent-endpoint>/openai/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{"model":"app","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

根据部署环境补充认证信息。示例固定使用代码中的模型连接；请求的 `model` 字段不会自动切换连接。AG-UI 接口为 `POST /ag-ui/agent`，健康检查为 `GET /healthz` 和 `GET /readyz`。

AG-UI 请求示例（将 Endpoint 替换为实际地址）：

```bash
curl -N 'https://<agent-endpoint>/ag-ui/agent' \
  -H 'Content-Type: application/json' \
  -d '{
    "threadId": "example-thread-1",
    "runId": "example-run-1",
    "messages": [
      {"id": "message-1", "role": "user", "content": "加载可用 Skill，按其说明完成一个示范。"}
    ],
    "state": {},
    "tools": [],
    "context": [],
    "forwardedProps": {}
  }'
```

每次运行使用新的 `runId`；同一对话可复用 `threadId`，但它不会让服务自动保存历史。`tools: []` 不会禁用 Agent 代码中已经配置的 MCP/Skill 工具。

使用 LangChain 完整服务时，正常输出以 `RUN_STARTED` 开始、`RUN_FINISHED` 结束。发生工具调用时，可看到独立文本消息的 START/CONTENT/END，以及 `TOOL_CALL_START/ARGS/END`、`TOOL_CALL_RESULT`；调用和结果由相同 `toolCallId` 关联。是否调用工具取决于模型和任务，不保证每个请求都有工具事件。运行失败会输出 `RUN_ERROR`。

OpenAI 流式响应输出 `choices[].delta` 并以 `[DONE]` 结束；将 `stream` 改为 `false` 可获取单个 JSON 响应。它不输出独立工具结果，也不能保留多条 assistant 消息边界。模型和工具循环由此服务执行，客户端不要把返回的工具调用轨迹再次执行。

最小服务示例是等待模型完成后返回文本，不是逐 token 生成示例。要观察完整流式执行过程，请使用 LangChain 完整服务。默认不开放浏览器跨域；若浏览器与服务跨域，由应用按实际来源配置 CORS。

需要展示工具调用和工具结果时，使用[框架执行事件转换器](frameworks/execution-events.md)接入完整执行流。AG-UI 保留消息边界和工具结果；OpenAI Chat Completions 不表达完整执行轨迹。服务不会替应用保存会话历史，需要多轮对话时由应用或框架管理。
