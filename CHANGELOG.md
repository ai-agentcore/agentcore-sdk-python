# Release notes

## 0.1.3 — 2026-10-18

- Memory 写入默认改为异步提取，`add_memories()` 返回任务 ID。需要立即读取记忆 ID 时，显式传入 `sync=True`。
- 新增 `get_memory_task(task_id)`，用于查询提取进度、记忆 ID 和任务错误；写入支持通过 `reference_time` 指定时间参考点。
- LangChain、LangGraph、AgentScope、Google ADK、Pydantic AI 和 CrewAI 的记忆回写改为等待任务受理，不等待提取完成。
- 修复 Google ADK 适配器加载不相关扩展依赖的问题。
- 将 LiteLLM 固定为 `1.102.4`，修复已知的启动异常。
