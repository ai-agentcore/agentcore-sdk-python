# Release notes

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
