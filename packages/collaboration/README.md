# AgentCore Collaboration for Python

为 AgentCore Agent 添加 Worker 任务协作能力，包括任务执行、进度反馈和成果提交。

这是基础 SDK 的可选扩展包，可以独立安装和升级。

协作能力目前按白名单开放，需先在 AgentCore 平台开通。安装本包不会自动开通平台协作能力。

## 安装

公开发行版的安装方式（可用版本以正式发布为准）：

```bash
pip install "alibabacloud-agentcore-sdk[collaboration]"
```

协作包 0.1.x 适用于基础 SDK `>=0.1.0,<0.2.0`。

本页使用 LangChain，完整服务示例还需要 Server 支持。请额外安装：

```bash
pip install "alibabacloud-agentcore-sdk[langchain,server]"
```

使用其他框架时，按需选择对应的 [框架 extra](../../examples/README.md)。

## 使用

先在 AgentCore 中为 Agent 配置团队和 Worker 角色，再将协作指令、工具及 Skill 加入 Agent：

```python
from agentcore import AsyncAgentCore
from agentcore.integrations.langchain import skill_tools, tools

core = AsyncAgentCore.auto()
worker = core.collaboration.worker()

instructions = worker.compose_prompt("完成分配的业务任务。")
agent_tools = [*tools(worker.tools()), *skill_tools(worker.skills())]
# 将 instructions 和 agent_tools 传给 LangChain Agent。
# 应用退出时 await core.aclose()。
```

完整示例见 [LangChain 协作服务](../../examples/collaboration_agent.py)。其他框架可使用各自的工具适配器。

建议通过 `AgentCoreServer` 接收协作任务；自建服务器需要在完整的 Agent 执行范围内使用 `worker.request_context(headers)`，并确保请求来自受信任的调用方。

## 使用自己的 Server

不使用 `AgentCoreServer` 时，也可以将协作工具接入已有应用。沿用上面的 `core`、`worker`、`instructions` 和 `agent_tools`，构建一次 Agent：

```python
from langchain.agents import create_agent
from agentcore.integrations.langchain import model

agent = create_agent(
    model=model("test-mc", model_name="qwen3.8-max"),
    system_prompt=instructions,
    tools=agent_tools,
)
```

在现有 Server 的请求处理函数中，将请求头和解析后的消息传入以下接入函数。普通调用：

```python
async def invoke_agent(headers, messages):
    with worker.request_context(headers):
        return await agent.ainvoke({"messages": messages})
```

流式调用：

```python
async def stream_agent(headers, messages):
    # 绑定发生在生成器内部，覆盖完整迭代和工具执行。
    with worker.request_context(headers):
        async for update in agent.astream(
            {"messages": messages}, stream_mode="updates"
        ):
            yield update
```

这些函数返回 LangChain 的结果或更新，不是 AG-UI / OpenAI 协议响应。自建 Server 负责解析请求、按平台配置的协议编码响应，并在客户端断开时取消执行、关闭生成器。不要只在 Context 内创建生成器、退出 Context 后再消费它。

请求入口应先完成鉴权，并从可信平台入口接收 `X-AgentCore-Collaboration-Context` 和 `X-AgentCore-Session-ID`，不能直接信任任意客户端提供的协作 Header。`worker.request_context()` 只绑定协作上下文，不代替入口鉴权，也不绑定 SDK 的普通请求 Context。

应用退出时调用 `await core.aclose()`。这些接入函数不需要 `server` extra；安装协作和 LangChain extra，以及应用自己使用的 Web 框架即可。

## 协作环境配置

协作工具调用时重新读取运行时 env 文件（默认 `/var/run/agentcore/agent/env`）。任务服务地址和具名 Matrix Token 优先使用文件中的值，缺少的字段从进程环境变量补充。文件不存在时允许仅使用环境变量；必需字段仍缺失、文件无法读取或格式错误时，调用报错，不静默回退。文件中的凭证更新会在后续调用中生效，不会被旧的进程环境变量覆盖。

## Matrix 房间附件

注册上述工具和 Skills 后，会包含 `room-files` Skill 以及两个工具，无需修改 AG-UI / OpenAI 消息格式，也不需要增加附件 Header：

| 工具 | 参数 | 成功结果 `data` |
|---|---|---|
| `agentteams_download_room_file` | `event_id` | `event_id`、`local_path`、`filename`、`content_type`、`size` |
| `agentteams_send_room_file` | `local_path`、可选 `caption` | `event_id`、`filename`、`status: "sent"` |

支持未加密的普通文件、PNG/JPEG/GIF/WebP 图片收发；其他图片类型按普通文件发送。群聊可先上传文件，再回复该文件并 @Agent。转发输入只包含附件摘要及事件 ID；工具按当前请求的房间取得附件，Token 和媒体 URL 不会交给模型。房间文件不要求属于某个 Team，但需要有效 Matrix 身份、当前请求 Context 及房间访问权限。

下载保存在协作工作目录的独立 `attachments/` 子目录中，失败或取消时清理半文件。上传只允许该工作目录中的普通文件，不接受越界路径或符号链接。文件大小受 Matrix 媒体上限约束。成功下载文件由应用管理保留周期，SDK 不在会话结束时删除。

下载只完成文件传输，不代表模型已读懂文件。应用需要注册自己的文本、文档或图片读取工具；图片还需要支持视觉的模型。参见 [AgentScope 原生 Read/Write 接入示例](../../examples/collaboration_room_files.py)，SDK 不另包通用读取工具。

安装 `agentscope,collaboration,server` extras，配置运行时后可执行：

```bash
uvicorn examples.collaboration_room_files:server --host 0.0.0.0 --port 8080
```

示例使用 `test-mc` 模型连接，可通过 `EXAMPLE_MODEL_NAME` 指定该连接下的视觉模型。工作目录使用 `AGENT_WORKSPACE`（默认当前目录），必须与读取工具共享同一文件系统。

发送成功后，Matrix 房间会出现回复当前消息的原生附件；这不会提交 Task Result，也不会结束 Agent Run。只有返回 `status: "sent"` 才可确认送达。`COLLABORATION_ROOM_FILE_SEND_UNKNOWN` 表示结果无法确认，不要盲目再次调用发送工具。

自建 Server 沿用上文的 `worker.request_context(headers)`，无需预下载附件。先升级协作包和 Agent，再部署支持附件摘要的转发组件。

## 升级

```bash
pip install -U alibabacloud-agentcore-collaboration
```

保持基础 SDK 版本兼容，升级后重启应用即可使用新的协作能力。镜像部署时请重新构建镜像。
