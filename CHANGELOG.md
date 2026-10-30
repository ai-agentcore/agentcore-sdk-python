# Release notes

## SDK 0.1.5 — 2026-10-30

- 平台 MCP 调用从当前请求上下文透传 WAT，按请求隔离身份；直接连接第三方 MCP 时不自动附加平台身份。
- 支持每次 MCP 工具调用传入业务 Header，并在同一请求内复用 MCP 会话。
- 请求结束时清理会话，兼容同步和异步调用；恢复重定向兼容性，跨源重定向不转发 WAT。

## Collaboration 0.1.3 — 2026-10-22

- 新增房间附件下载与发送工具，按当前房间中的 Matrix Event ID 下载附件到工作区，并将本地文件或图片作为原生 Matrix 消息回复。
- 内置 `room-files` Skill；文件解析和图片理解继续使用应用已有的工具，无需加入 Team 即可处理房间文件。
- 支持请求取消时清理下载中的临时文件；发送结果不确定时保留事务信息，避免盲目重传导致重复消息。

## 0.1.4 — 2026-10-19

- 新增 Agent 模版读取入口 `core.templates.get()`，支持异步和同步调用，按指定版本或 `latest` 标签加载 Workspace 中的模版。
- 通过 `template.instruction` 获取主 Agent 的 Prompt，通过 `load_skills()` 按需加载随包静态 Skill 和引用的云端 Skill；模型仍由应用独立选择。
- Skill 加载支持并发复用和成功结果缓存；同一模版的静态 Skill 保持同级目录，兼容现有工具的相对路径命令执行。
- 补充 LangChain 使用示例，以及模版解析、真实 HTTP 协议和工具执行的回归测试。

## 0.1.3 — 2026-10-18

- Memory 写入默认改为异步提取，`add_memories()` 返回任务 ID。需要立即读取记忆 ID 时，显式传入 `sync=True`。
- 新增 `get_memory_task(task_id)`，用于查询提取进度、记忆 ID 和任务错误；写入支持通过 `reference_time` 指定时间参考点。
- LangChain、LangGraph、AgentScope、Google ADK、Pydantic AI 和 CrewAI 的记忆回写改为等待任务受理，不等待提取完成。
- 修复 Google ADK 适配器加载不相关扩展依赖的问题。
- 将 LiteLLM 固定为 `1.102.4`，修复已知的启动异常。
