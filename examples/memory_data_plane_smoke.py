"""Optional pre-release smoke test for an existing READY MemoryStore."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from agentcore import AsyncAgentCore
from agentcore.auth import AccessKeyCredential
from agentcore.errors import AgentCoreError
from agentcore.memory import MemoryMessage, MemoryScope


async def run_smoke(
    core: Any,
    memory_store_name: str,
    *,
    search_attempts: int = 6,
    search_delay: float = 2.0,
) -> dict[str, Any]:
    suffix = uuid.uuid4().hex
    user_id = f"sdk-smoke-user-{suffix}"
    agent_id = f"sdk-smoke-agent-{suffix}"
    session_id = f"sdk-smoke-session-{suffix}"
    scope = MemoryScope(user_id=user_id, agent_id=agent_id, session_id=session_id)
    store = core.memory_store(memory_store_name)
    tracked_ids: set[str] = set()
    operations: list[str] = []
    search_visible = False
    search_metadata = {"source": "agentcore-sdk-smoke"}
    if any(len(key) + len(value) > 29 for key, value in search_metadata.items()):
        raise AssertionError("Search metadata exceeds the temporary OTS key/value limit")
    report: dict[str, Any] | None = None
    cleanup_errors: list[dict[str, Any]] = []
    try:
        added = await store.add_memories(
            scope=scope,
            messages=[
                MemoryMessage(
                    role="user",
                    content=f"Remember coffee preference {suffix}",
                ),
                MemoryMessage(role="assistant", content="Preference acknowledged"),
            ],
            metadata=search_metadata,
        )
        operations.append("AddMemories")
        tracked_ids.update(added.memory_ids)

        for attempt in range(search_attempts):
            searched = await store.search_memories(
                suffix,
                scope=scope,
                top_k=10,
                metadata=search_metadata,
                enable_rerank=False,
                min_similarity=0.0,
                min_score=0.0,
            )
            operations.append("SearchMemories")
            if searched.memories:
                search_visible = True
                break
            if attempt + 1 < search_attempts:
                await asyncio.sleep(search_delay)

        page = await store.list_memories(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            max_results=10,
        )
        operations.append("ListMemories")
        tracked_ids.update(item.memory_id for item in page.items)
        if not tracked_ids:
            raise RuntimeError("smoke write produced no memory to verify")

        memory_id = sorted(tracked_ids)[0]
        await store.get_memory(memory_id)
        operations.append("GetMemory")
        await store.update_memory(memory_id, metadata={"source": "agentcore-sdk-smoke-updated"})
        operations.append("UpdateMemory")
        await store.list_memory_sessions(
            user_id=user_id,
            agent_id=agent_id,
            max_results=10,
        )
        operations.append("ListMemorySessions")
        await store.list_memory_session_messages(
            session_id,
            user_id=user_id,
            agent_id=agent_id,
            max_results=10,
        )
        operations.append("ListMemorySessionMessages")
        await store.delete_memory(memory_id)
        operations.append("DeleteMemory")
        tracked_ids.remove(memory_id)
        report = {
            "success": True,
            "operations": operations,
            "searchVisible": search_visible,
        }
    finally:
        for memory_id in tuple(tracked_ids):
            try:
                await store.delete_memory(memory_id)
                tracked_ids.remove(memory_id)
            except AgentCoreError as exc:
                cleanup_errors.append(
                    {
                        "memoryId": memory_id,
                        "code": exc.code,
                        "requestId": getattr(exc, "request_id", None),
                    }
                )
    assert report is not None
    report["remainingMemoryIds"] = len(tracked_ids)
    report["cleanupErrors"] = cleanup_errors
    report["success"] = not cleanup_errors
    return report


async def _main() -> None:
    memory_store_name = os.environ.get("AGENTCORE_MEMORY_STORE_NAME", "").strip()
    if not memory_store_name:
        raise SystemExit("AGENTCORE_MEMORY_STORE_NAME must name an existing READY MemoryStore")
    workspace_id = os.environ.get("AGENTCORE_E2E_WORKSPACE_ID", "").strip()
    region_id = os.environ.get("AGENTCORE_E2E_REGION_ID", "cn-hangzhou").strip()
    endpoint = os.environ.get(
        "AGENTCORE_E2E_ENDPOINT", "https://agentcore-pre.aliyuncs.com"
    ).strip()
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "").strip()
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "").strip()
    security_token = os.environ.get("ALIBABA_CLOUD_SECURITY_TOKEN", "").strip() or None
    if bool(access_key_id) != bool(access_key_secret):
        raise SystemExit(
            "ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET "
            "must be provided together"
        )
    if (access_key_id or workspace_id) and not (access_key_id and workspace_id):
        raise SystemExit(
            "AGENTCORE_E2E_WORKSPACE_ID and explicit AK/SK must be provided together"
        )
    core_options: dict[str, Any] = {}
    if access_key_id:
        core_options = {
            "workspace_id": workspace_id,
            "region_id": region_id,
            "control_plane_endpoint": endpoint,
            "access_key_credential": AccessKeyCredential(
                access_key_id,
                access_key_secret,
                security_token,
            ),
        }
    try:
        async with AsyncAgentCore.auto(**core_options) as core:
            report = await run_smoke(core, memory_store_name)
    except AgentCoreError as exc:
        report = {
            "success": False,
            "code": exc.code,
            "requestId": getattr(exc, "request_id", None),
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
