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
curl https://<agent-endpoint>/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"app","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

根据部署环境补充认证信息。示例固定使用代码中的模型连接；请求的 `model` 字段不会自动切换连接。AG-UI 接口为 `POST /ag-ui/agent`，健康检查为 `GET /healthz` 和 `GET /readyz`。

需要展示工具调用和工具结果时，使用[框架执行事件转换器](frameworks/execution-events.md)接入完整执行流。AG-UI 保留消息边界和工具结果；OpenAI Chat Completions 不表达完整执行轨迹。服务不会替应用保存会话历史，需要多轮对话时由应用或框架管理。
