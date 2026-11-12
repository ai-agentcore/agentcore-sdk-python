# AgentCore Memory plugin

为自有 Agent harness 提供显式事件采集与召回。原生入口复用 core 的 Context Memory 客户端，宿主负责可信身份、source ID、持久 outbox 和重试决策。原有 Gateway 构造器继续支持已发布调用方。

## 原生 Runtime 入口

```sh
python -m pip install . ./packages/memory-plugin
export AGENTCORE_MEMORY_PLUGIN_ENABLED=true
export AGENTCORE_MEMORY_STORE_NAMES="mymemory:true,yourmemory:false"
```

core 与插件当前均为尚未发布的 0.1.7。插件依赖 `alibabacloud-agentcore-sdk>=0.1.7,<1`，必须搭配导出原生客户端的对应 core 构建。

```python
from agentcore import AsyncAgentCore
from agentcore.memory import MemoryScope, MemoryType
from agentcore_memory_plugin import MemoryPlugin, SourceEvent, SourceRecord

async def process_turn(store, trusted_scope, persisted_event, query):
    async with AsyncAgentCore.auto() as core:
        client = core.context_memory(store)
        async with MemoryPlugin.from_native(client) as plugin:
            recalled = await plugin.recall(
                query, scope=trusted_scope, types=[MemoryType.SEMANTIC], top_k=20,
            )
            captured = await plugin.capture([
                SourceRecord(persisted_event["source_id"], SourceEvent(**persisted_event["event"]))
            ], scope=trusted_scope)
            return recalled, captured
```

`from_native(client, environ=None)` 默认读取并固定下述两个环境开关，全局默认关闭，未列出的库禁用，`name:false` 保留召回并禁用采集。可传 `environ=values` 使用宿主已读取的可信 Runtime 投影。不需要 Gateway token。关闭插件不会关闭借用的客户端或凭据提供器。endpoint 与 `context` 凭据目的由 core 负责，参见 [原生 API](../../README.md#使用原生-context-memory)。

原生 `capture(records, scope=...)` 返回 `CaptureResult`，其 receipts 保留 source_id、input_index、status、event_id、request_id、code、message。`STORED` 对应 state=stored，`FAILED` 和 `REJECTED` 对应 failed，其余服务状态对应 unknown，并保留实际 status。source_id 只在本地关联，不进入云端请求，也不是 event ID；回执通过 input_index 对应原始输入。core 统一转换五类来源事件，opaque 值为键排序的紧凑 UTF-8 JSON 字符串，所有非消息事件进入 `harness.<type>;encoding=json` opaque 块，工具结果保留 role=tool 和 callId。

派发后超时、连接断开、5xx 或回执损坏会抛出 core `CreateEventsOutcomeUnknownError`，其 `outcome_unknown=True`。取消保留原生取消类型，派发后附加未知结果元数据。宿主保留原事件，并使用 `client.list_events` 分页对账后再决定是否重试。原生插件不排队、不自动重试、不维护交付状态，也不调用 AddMemories。

原生 `recall` 返回 `NativeRecallResult`。memories 保留全部原生 hit、scope、来源、版本、时间和评分，request_id 保留服务请求 ID。context 是带不可信参考标记且受 max_chars 限制的便利渲染，不改变原生 memories。types 接受四种 `MemoryType`，top_k 为 1 至 200，省略时使用服务默认。关闭能力返回空结果，服务失败明确抛错。单次 `timeout` 以秒计，覆盖默认 timeout_ms。

`MemoryScope` 的 user、team、agent、session 四维独立可选。原生插件也支持已有 session 入口，`MemoryBinding(store, None, agent)` 可省略业务 user，保持旧位置参数顺序。session capture 带 session，shared recall 省略 session；更强隔离策略由宿主通过原生方法的 scope 决定。原生 CLI [memory_plugin.py](examples/memory_plugin.py) 读取真实持久事件，capture 子命令会写入云端，仍受两个开关限制。本次没有执行云端验收。

## 已有 Gateway 入口

### 本地安装与使用

```sh
python -m pip install ./packages/memory-plugin
```

源码包版本为 0.1.7，尚未发布。运行要求 Python 3.10+；异步 HTTP 客户端需要在退出时关闭。

```python
import os
from agentcore_memory_plugin import MemoryBinding, MemoryPlugin, SourceEvent, SourceRecord

async def process_turn(trusted_identity, persisted_event, query):
    async with MemoryPlugin(
        os.environ["MEMORY_GATEWAY_ENDPOINT"],  # 完整 /v2/memory 地址
        token_provider=lambda: os.environ["MEMORY_TOKEN"],
        capture_enabled=True,
    ) as plugin:
        session = plugin.session(
            MemoryBinding(
                trusted_identity["memory_store_name"],
                trusted_identity["user_id"],
                trusted_identity["agent_id"],
                trusted_identity.get("team_id"),
            ),
            trusted_identity["session_id"],
        )
        recalled = await session.recall(query, mode="shared", max_results=3, max_chars=3600)
        # 将 recalled.context 作为带来源标记的参考数据交给业务框架。
        # persisted_event 来自已保存的真实执行记录，不来自模型生成的身份。
        receipts = await session.capture([
            SourceRecord(
                persisted_event["source_id"],
                SourceEvent(**persisted_event["event"]),
            )
        ])
        return recalled, receipts
```

以下说明适用于保留的 Gateway 构造器和 `from_env(endpoint, token_provider=...)`。

### 使用环境变量配置

```sh
export AGENTCORE_MEMORY_PLUGIN_ENABLED=true
export AGENTCORE_MEMORY_STORE_NAMES="mymemory:true,yourmemory:false,memstore-1:true"
```

使用环境配置入口创建插件：

```python
plugin = MemoryPlugin.from_env(
    gateway_endpoint,
    token_provider=lambda: os.environ["MEMORY_TOKEN"],
)
# 按原有方式创建 session，业务 binding.memory_store_name 选择其中一个库。
# await plugin.aclose()，或使用 async with MemoryPlugin.from_env(...)。
```

全局开关默认 false，关闭时不发送采集或召回请求。库名列表默认为空，未列出的库不启用；已列出的 `name:true` 开启采集和召回，`name:false` 只关闭采集，保留召回。布尔值只接受 true/false，忽略大小写及条目两侧空白；重复库名、空条目、缺失布尔值或格式非法均报错。

配置在 from_env 调用时读取并固定，修改后需重建插件。可传 `environ=values` 使用业务已读取的可信运行环境投影；默认读取进程环境。环境入口不接受 capture_enabled/recall_enabled 覆盖，避免绕过逐库策略。显式 `MemoryPlugin(...)` 仍只使用构造参数，不读取这两个变量；基础 MemoryStore API 不受影响。环境变量不提供资源授权，每个 binding 仍需对应的服务令牌。

### 事件与生命周期

支持 message、tool_call、tool_result、model_request、model_response 五种事件。timestamp 是源记录中的非负整数毫秒。消息 content 是结构化内容块数组，例如 `[{"type":"text","text":"原文"}]`；工具调用保留 callId、name、arguments，结果保留 callId、status、content。

执行前显式召回；事件完整可观察后显式采集。流式 chunk、部分答案、取消和工具执行完成由 harness 判断，不从 SDK 展示事件推断。SDK 不自动注入提示词、不维护 checkpoint，也不自动安装框架 middleware。对相同事件启用本包时，避免同时开启原 Memory middleware 的自动回写。

### 配置与结果

采集默认关闭，召回默认开启；timeout_ms 默认 1500，允许 100 至 10000。identity 由业务鉴权入口决定，每个 session 固定 store、user、agent、可选 team 和 session ID。token_provider 每次请求读取服务令牌；它不是云端 AK 或模型 API key，必须与服务授权的完整 binding 匹配。

capture 每次 1 至 100 条，单事件不超过 900000 UTF-8 字节，总请求不超过 1 MiB；不自动拆批。source_id 必须包含业务 producer 命名空间，并与 timestamp、事件内容一起持久保存。重复提交使用完全相同的记录。

回执 state 保留 queued、stored、unknown、failed；queued 表示网关已受理，stored 表示云端存储，均不代表提取完成或已能检索。`MemoryPluginError.outcome_unknown=True` 表示采集结果不确定；调用方保留原事件并核对网关，不能生成新 source ID 盲目重发。本包没有本地持久队列或后台重试，网关受理前的可靠性由调用方负责。取消可能中断等待，也不能当作服务端未受理的证明。

recall 的 shared 模式在相同授权范围跨 session 检索，session 模式限定当前 session。返回 memories 保留 source_event_ids；context 标记为不可信参考数据，max_chars 包括标记和 JSON 格式开销，按 Unicode 码点计数。空查询或关闭召回返回空结果；服务错误明确抛出，不伪装成无记忆。准备 context 不证明模型实际使用。

服务必须支持 `host=sdk` 的 `/v2/memory/capture` 与 `/search`。需要升级旧版仅允许三个原生宿主的接入服务。首稿未完成真实云端端到端验收。
