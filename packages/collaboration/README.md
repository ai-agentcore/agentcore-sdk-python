# AgentCore Collaboration for Python

为 AgentCore Agent 添加 Worker 任务协作能力，包括任务执行、进度反馈和成果提交。

这是基础 SDK 的可选扩展包，可以独立安装和升级。

## 安装

公开发行版的安装方式（可用版本以正式发布为准）：

```bash
pip install "alibabacloud-agentcore-sdk[collaboration]"
```

协作包 0.1.x 适用于基础 SDK `>=0.2.1,<0.3.0`。

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

## 升级

```bash
pip install -U alibabacloud-agentcore-collaboration
```

保持基础 SDK 版本兼容，升级后重启应用即可使用新的协作能力。镜像部署时请重新构建镜像。
