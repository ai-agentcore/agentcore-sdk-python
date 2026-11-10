# AgentCore Memory plugin（首稿）

为自有 Agent harness 提供显式事件采集与召回。此可选包连接单独部署的 Memory 接入服务，不调用 AddMemories 模拟事件采集，也不依赖 harness 源码路径。原有 SDK MemoryStore 和框架中间件保持原来的行为。

## 本地安装与使用

```sh
python -m pip install ./packages/memory-plugin
```

源码包版本为 0.1.0，尚未发布。运行要求 Python 3.10+；异步 HTTP 客户端需要在退出时关闭。

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

`examples/memory_plugin.py` 提供读取业务配置、召回和提交持久事件文件的命令行入口。它不生成测试数据；capture 是真实写操作，只有调用该子命令且环境中的全局与对应库采集开关均开启时才会提交。

## 使用环境变量配置

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

## 事件与生命周期

支持 message、tool_call、tool_result、model_request、model_response 五种事件。timestamp 是源记录中的非负整数毫秒。消息 content 是结构化内容块数组，例如 `[{"type":"text","text":"原文"}]`；工具调用保留 callId、name、arguments，结果保留 callId、status、content。

执行前显式召回；事件完整可观察后显式采集。流式 chunk、部分答案、取消和工具执行完成由 harness 判断，不从 SDK 展示事件推断。SDK 不自动注入提示词、不维护 checkpoint，也不自动安装框架 middleware。对相同事件启用本包时，避免同时开启原 Memory middleware 的自动回写。

## 配置与结果

采集默认关闭，召回默认开启；timeout_ms 默认 1500，允许 100 至 10000。identity 由业务鉴权入口决定，每个 session 固定 store、user、agent、可选 team 和 session ID。token_provider 每次请求读取服务令牌；它不是云端 AK 或模型 API key，必须与服务授权的完整 binding 匹配。

capture 每次 1 至 100 条，单事件不超过 900000 UTF-8 字节，总请求不超过 1 MiB；不自动拆批。source_id 必须包含业务 producer 命名空间，并与 timestamp、事件内容一起持久保存。重复提交使用完全相同的记录。

回执 state 保留 queued、stored、unknown、failed；queued 表示网关已受理，stored 表示云端存储，均不代表提取完成或已能检索。`MemoryPluginError.outcome_unknown=True` 表示采集结果不确定；调用方保留原事件并核对网关，不能生成新 source ID 盲目重发。本包没有本地持久队列或后台重试，网关受理前的可靠性由调用方负责。取消可能中断等待，也不能当作服务端未受理的证明。

recall 的 shared 模式在相同授权范围跨 session 检索，session 模式限定当前 session。返回 memories 保留 source_event_ids；context 标记为不可信参考数据，max_chars 包括标记和 JSON 格式开销，按 Unicode 码点计数。空查询或关闭召回返回空结果；服务错误明确抛出，不伪装成无记忆。准备 context 不证明模型实际使用。

服务必须支持 `host=sdk` 的 `/v2/memory/capture` 与 `/search`。需要升级旧版仅允许三个原生宿主的接入服务。首稿未完成真实云端端到端验收。
