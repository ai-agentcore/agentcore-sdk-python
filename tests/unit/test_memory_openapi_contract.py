from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agentcore.auth.resource_sts import ResourceCredential
from agentcore.memory import AsyncMemoryStore, MemoryMessage, MemoryScope
from agentcore.memory._transport import _MemoryRuntime


@dataclass(frozen=True)
class WireCall:
    params: dict[str, Any]
    request: dict[str, Any]
    runtime: dict[str, Any]


class CapturingOpenAPIClient:
    def __init__(self) -> None:
        self.calls: list[WireCall] = []

    async def call_api_async(self, params: Any, request: Any, runtime: Any) -> Any:
        self.calls.append(WireCall(params.to_map(), request.to_map(), runtime.to_map()))
        action = params.action
        body: dict[str, Any] = {"success": True, "httpStatusCode": 200}
        if action == "AddMemories":
            body["data"] = {"memories": []}
        elif action == "SearchMemories":
            body["data"] = {"memories": []}
        elif action == "GetMemory":
            body["data"] = _memory_map("memory-get", "get text")
        elif action == "UpdateMemory":
            body["data"] = _memory_map("memory-update", "updated text")
        elif action in {
            "ListMemories",
            "ListMemorySessions",
            "ListMemorySessionMessages",
        }:
            body["items"] = []
        return {"statusCode": 200, "headers": {}, "body": body}


class WireSTS:
    def __init__(self) -> None:
        self.purposes: list[str | None] = []

    async def get(self, purpose: str | None = None) -> ResourceCredential:
        self.purposes.append(purpose)
        return ResourceCredential(
            access_key_id="wire-ak",
            access_key_secret="wire-sk",
            security_token="wire-token",
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )


def _memory_map(memory_id: str, text: str) -> dict[str, Any]:
    return {
        "memoryId": memory_id,
        "content": {"text": text},
        "scope": {
            "agentId": "agent-1",
            "sessionId": "session-1",
            "userId": "user-1",
        },
    }


@pytest.mark.asyncio
async def test_public_calls_need_only_common_request_with_exact_wire_contract() -> None:
    client = CapturingOpenAPIClient()
    sts = WireSTS()
    runtime = _MemoryRuntime(
        workspace_id="workspace-runtime",
        region_id="cn-hangzhou",
        endpoint=None,
        resource_sts=sts,
    )

    async def runtime_provider() -> _MemoryRuntime:
        return runtime

    store = AsyncMemoryStore(
        "customer_memory",
        _runtime_provider=runtime_provider,
        _client_factory=lambda resolved, credential: client,
    )
    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")

    await store.add_memories(
        scope=scope,
        messages=[
            MemoryMessage(
                role="user",
                content="hello",
            )
        ],
        metadata={"kind": "dialogue"},
    )
    await store.search_memories(
        "find this",
        scope=MemoryScope(session_id="session-1", user_id="user-1"),
        top_k=5,
        metadata={"kind": "dialogue"},
        enable_rerank=False,
        min_similarity=0.0,
        min_score=0.2,
    )
    await store.list_memories(
        user_id="user-1",
        agent_id="agent-1",
        session_id="session-1",
        max_results=10,
        next_token="memory-token",
    )
    await store.get_memory("memory-get")
    await store.update_memory(
        "memory-update",
        text="new text",
        metadata={"kind": "updated"},
    )
    await store.delete_memory("memory-delete")
    await store.list_memory_sessions(
        user_id="user-1",
        agent_id="agent-1",
        max_results=20,
        next_token="session-token",
    )
    await store.list_memory_session_messages(
        "session-1",
        user_id="user-1",
        agent_id="agent-1",
        max_results=30,
        next_token="message-token",
    )

    actual_routes = [
        (call.params["action"], call.params["method"], call.params["pathname"])
        for call in client.calls
    ]
    assert actual_routes == [
        (
            "AddMemories",
            "POST",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories",
        ),
        (
            "SearchMemories",
            "POST",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories/search",
        ),
        (
            "ListMemories",
            "GET",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories",
        ),
        (
            "GetMemory",
            "GET",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories/memory-get",
        ),
        (
            "UpdateMemory",
            "PUT",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories/memory-update",
        ),
        (
            "DeleteMemory",
            "DELETE",
            "/workspaces/workspace-runtime/memorystores/customer_memory/memories/memory-delete",
        ),
        (
            "ListMemorySessions",
            "GET",
            "/workspaces/workspace-runtime/memorystores/customer_memory/sessions",
        ),
        (
            "ListMemorySessionMessages",
            "GET",
            "/workspaces/workspace-runtime/memorystores/customer_memory/messages",
        ),
    ]

    add_body = json.loads(client.calls[0].request["body"]["body"])
    assert add_body == {
        "messages": [
            {
                "content": "hello",
                "role": "user",
            }
        ],
        "metadata": {"kind": "dialogue"},
        "scope": {
            "agentId": "agent-1",
            "sessionId": "session-1",
            "userId": "user-1",
        },
    }
    search_body = json.loads(client.calls[1].request["body"]["body"])
    assert search_body == {
        "enableRerank": False,
        "metadata": {"kind": "dialogue"},
        "minScore": 0.2,
        "minSimilarity": 0.0,
        "query": "find this",
        "scope": {"sessionId": "session-1", "userId": "user-1"},
        "topK": 5,
    }
    assert client.calls[2].request["query"] == {
        "agentId": "agent-1",
        "maxResults": "10",
        "nextToken": "memory-token",
        "sessionId": "session-1",
        "userId": "user-1",
    }
    assert client.calls[3].request == {"headers": {}}
    update_body = json.loads(client.calls[4].request["body"]["body"])
    assert update_body == {"metadata": {"kind": "updated"}, "text": "new text"}
    assert client.calls[5].request == {"headers": {}}
    assert client.calls[6].request["query"] == {
        "agentId": "agent-1",
        "maxResults": "20",
        "nextToken": "session-token",
        "userId": "user-1",
    }
    assert client.calls[7].request["query"] == {
        "agentId": "agent-1",
        "maxResults": "30",
        "nextToken": "message-token",
        "sessionId": "session-1",
        "userId": "user-1",
    }
    assert client.calls[0].runtime == {
        "autoretry": False,
        "max_attempts": 1,
        "readTimeout": 120_000,
    }
    assert all(
        call.runtime == {"autoretry": False, "max_attempts": 1} for call in client.calls[1:]
    )
    assert sts.purposes == ["highcode_sdk"] * 8
