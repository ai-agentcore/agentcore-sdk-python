"""Native Context Memory calls through the shared signed POP requester."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from functools import partial
from typing import Any, NoReturn

import anyio
from anyio.from_thread import BlockingPortal

from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.resource_sts import ResourceSTSProvider
from agentcore.controlplane.client import _normalize_endpoint
from agentcore.errors import (
    CreateEventsOutcomeUnknownError,
    MemoryAPIError,
    MemoryContractError,
    MemoryValidationError,
)
from agentcore.memory._transport import (
    _ErrorDetails,
    _field,
    _MemoryRuntime,
    _MemoryTransport,
    _RuntimeProvider,
    _scope_body,
)
from agentcore.memory.client import (
    _bounded_integer,
    _metadata,
    _next_token,
    _optional_boolean,
    _optional_scope,
    _required_string,
)
from agentcore.memory.context_models import (
    ContextEvent,
    ContextMemory,
    ContextMemorySearchHit,
    ContextMemorySearchResult,
    CreateEventsResult,
    EventPage,
    EventReceipt,
    MemorySource,
    MemoryType,
)
from agentcore.memory.models import MemoryContent, MemoryScope


class AsyncContextMemoryClient:
    """Three native operations bound to one existing workspace and memory store."""

    def __init__(self, memory_store_name: str, *, _runtime_provider: _RuntimeProvider) -> None:
        self.memory_store_name = _required_string(memory_store_name, "memory_store_name")
        self._transport = _MemoryTransport(
            self.memory_store_name, _runtime_provider, purpose="context"
        )

    @classmethod
    def from_credentials(
        cls,
        *,
        workspace_id: str,
        memory_store_name: str,
        region_id: str,
        endpoint: str | None = None,
        resource_sts: ResourceSTSProvider | None = None,
        access_key_credential: AccessKeyCredential | None = None,
    ) -> AsyncContextMemoryClient:
        """Borrow public credentials; their lifecycle remains with the caller."""
        _required_string(workspace_id, "workspace_id")
        _required_string(region_id, "region_id")
        if endpoint is None:
            endpoint = (
                os.getenv("AGENTCORE_MEMORY_ENDPOINT", "").strip()
                or f"https://agentcore.{region_id}.aliyuncs.com"
            )
        _required_string(endpoint, "endpoint")
        _normalize_endpoint(endpoint)
        if (resource_sts is None) == (access_key_credential is None):
            raise MemoryValidationError("provide exactly one credential provider")

        async def runtime() -> _MemoryRuntime:
            return _MemoryRuntime(
                workspace_id, region_id, endpoint, resource_sts, access_key_credential
            )

        return cls(memory_store_name, _runtime_provider=runtime)

    async def create_events(
        self,
        events: Sequence[ContextEvent],
        *,
        scope: MemoryScope | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> CreateEventsResult:
        scope = _optional_scope(scope)
        if (
            not isinstance(events, Sequence)
            or isinstance(events, (str, bytes))
            or not 1 <= len(events) <= 100
            or not all(isinstance(event, ContextEvent) for event in events)
        ):
            raise MemoryValidationError("events must contain 1 to 100 ContextEvent values")
        body, details = await self._call(
            "CreateEvents",
            "POST",
            "/events",
            timeout=timeout,
            body={"scope": _scope_body(scope), "events": [event.to_dict() for event in events]},
        )
        raw = _field(_field(body, "data"), "results")
        if not isinstance(raw, list) or len(raw) != len(events):
            self._invalid_write("data.results must contain one receipt per input", details)
        receipts: list[EventReceipt] = []
        seen: set[int] = set()
        for row in raw:
            index = _field(row, "input_index")
            if type(index) is not int or index not in range(len(events)) or index in seen:
                self._invalid_write(
                    "receipt inputIndex is missing, duplicated, or out of range", details
                )
            seen.add(index)
            try:
                receipts.append(
                    EventReceipt(
                        index,
                        _field(row, "status"),
                        _field(row, "event_id"),
                        _optional_string(_field(row, "code"), "CreateEvents", details),
                        _optional_string(_field(row, "message"), "CreateEvents", details),
                    )
                )
            except (MemoryValidationError, MemoryContractError):
                self._invalid_write("receipt contains invalid fields", details)
        return CreateEventsResult(tuple(receipts), details.request_id)

    async def list_events(
        self,
        *,
        scope: MemoryScope | None = None,
        next_token: str | None = None,
        max_results: int | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> EventPage:
        scope = _optional_scope(scope)
        _next_token(next_token)
        _bounded_integer(max_results, "max_results", maximum=100)
        body, details = await self._call(
            "ListEvents",
            "GET",
            "/events",
            timeout=timeout,
            query={
                **(_scope_body(scope) or {}),
                "nextToken": next_token,
                "maxResults": max_results,
            },
        )
        raw = _field(body, "items")
        if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
            _contract("ListEvents", "body.items must be a list of event envelopes", details)
        return EventPage(
            tuple(raw),
            _optional_string(_field(body, "next_token"), "ListEvents", details),
            _optional_integer(_field(body, "max_results"), "ListEvents", details),
            _optional_integer(_field(body, "total_count"), "ListEvents", details),
            details.request_id,
        )

    async def search_memories(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        types: Sequence[MemoryType] | None = None,
        top_k: int | None = None,
        metadata: Mapping[str, str] | None = None,
        enable_rerank: bool | None = None,
        min_similarity: float | None = None,
        min_score: float | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> ContextMemorySearchResult:
        _required_string(query, "query")
        scope = _optional_scope(scope)
        _bounded_integer(top_k, "top_k", maximum=200)
        validated_metadata = _metadata(metadata, "metadata")
        _optional_boolean(enable_rerank, "enable_rerank")
        for name, value in (("min_similarity", min_similarity), ("min_score", min_score)):
            if value is not None and not _finite(value):
                raise MemoryValidationError(f"{name} must be a finite number")
        if types is not None and (
            not isinstance(types, Sequence)
            or isinstance(types, (str, bytes))
            or not all(isinstance(item, MemoryType) for item in types)
        ):
            raise MemoryValidationError("types must contain MemoryType values")
        selected_types = list(dict.fromkeys(item.value for item in types)) if types else None
        body, details = await self._call(
            "SearchMemories",
            "POST",
            "/memories/search",
            timeout=timeout,
            body={
                "query": query,
                "scope": _scope_body(scope),
                "types": selected_types,
                "topK": top_k,
                "metadata": validated_metadata,
                "enableRerank": enable_rerank,
                "minSimilarity": min_similarity,
                "minScore": min_score,
            },
        )
        raw = _field(_field(body, "data"), "memories")
        if not isinstance(raw, list):
            _contract("SearchMemories", "data.memories must be a list", details)
        hits = tuple(_hit(row, details) for row in raw)
        for hit in hits:
            if scope is not None:
                for name in ("user_id", "team_id", "agent_id", "session_id"):
                    expected = getattr(scope, name)
                    if expected is not None and getattr(hit.memory.scope, name) != expected:
                        _contract("SearchMemories", f"memory scope does not match {name}", details)
            if (
                selected_types
                and hit.memory.type is not None
                and hit.memory.type.value not in selected_types
            ):
                _contract("SearchMemories", "memory type does not match requested types", details)
        return ContextMemorySearchResult(hits, details.request_id)

    async def _call(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        timeout: float | None,  # noqa: ASYNC109
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> tuple[Any, _ErrorDetails]:
        if timeout is not None and (not _finite(timeout) or timeout <= 0):
            raise MemoryValidationError("timeout must be a positive finite number of seconds")
        dispatched = False

        def on_dispatch() -> None:
            nonlocal dispatched
            dispatched = True

        write = operation == "CreateEvents"
        try:
            with anyio.fail_after(timeout):
                response = await self._transport._request(
                    operation,
                    method,
                    path,
                    body=body,
                    query=query,
                    add=write,
                    on_dispatch=on_dispatch,
                )
        except TimeoutError as exc:
            error = CreateEventsOutcomeUnknownError if write and dispatched else MemoryAPIError
            raise error(operation, service_code="Timeout") from exc
        return self._transport._success_body(response, operation, add=write)

    @staticmethod
    def _invalid_write(detail: str, details: _ErrorDetails) -> NoReturn:
        raise CreateEventsOutcomeUnknownError(
            "CreateEvents",
            request_id=details.request_id,
            http_status_code=details.http_status_code,
            service_code=details.service_code,
            service_message=detail,
        )


def _contract(operation: str, detail: str, details: _ErrorDetails) -> NoReturn:
    raise MemoryContractError(operation, detail, request_id=details.request_id)


def _optional_string(value: Any, operation: str, details: _ErrorDetails) -> str | None:
    if value is not None and not isinstance(value, str):
        _contract(operation, "optional field must be a string", details)
    return value


def _optional_integer(value: Any, operation: str, details: _ErrorDetails) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        _contract(operation, "optional count must be a non-negative integer", details)
    return value


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _hit(row: Any, details: _ErrorDetails) -> ContextMemorySearchHit:
    operation = "SearchMemories"
    memory = _field(row, "memory")
    memory_id = _field(memory, "memory_id")
    text = _field(_field(memory, "content"), "text")
    raw_scope = _field(memory, "scope")
    if (
        not isinstance(memory_id, str)
        or not memory_id.strip()
        or not isinstance(text, str)
        or not isinstance(raw_scope, Mapping)
    ):
        _contract(operation, "memory requires memoryId, content.text, and scope", details)
    scope = MemoryScope(
        **{
            name: _optional_string(_field(raw_scope, name), operation, details)
            for name in ("agent_id", "session_id", "user_id", "team_id")
        }
    )
    metadata = _field(memory, "metadata")
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or not all(
            isinstance(key, str) and key.strip() and isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        _contract(operation, "memory.metadata must contain string keys and values", details)
    kind = _field(memory, "type")
    source = _field(memory, "source")
    try:
        kind = MemoryType(kind) if kind is not None else None
        source = MemorySource(source) if source is not None else None
    except (ValueError, TypeError):
        _contract(operation, "memory has an unsupported type or source", details)
    valid = _field(memory, "valid")
    version = _field(memory, "version")
    if valid is not None and type(valid) is not bool:
        _contract(operation, "memory.valid must be a boolean", details)
    if version is not None and type(version) not in (str, int):
        _contract(operation, "memory.version must be a string or integer", details)
    ids = _field(memory, "source_event_ids")
    if ids is not None and (
        not isinstance(ids, list) or not all(isinstance(item, str) and item.strip() for item in ids)
    ):
        _contract(operation, "memory.sourceEventIds must contain non-empty strings", details)
    score, similarity = _field(row, "score"), _field(row, "similarity")
    if not _finite(score) or not _finite(similarity):
        _contract(operation, "score and similarity must be finite numbers", details)
    return ContextMemorySearchHit(
        ContextMemory(
            memory_id,
            MemoryContent(text),
            scope,
            metadata,
            kind,
            source,
            valid,
            tuple(ids) if ids is not None else None,
            version,
            _optional_string(_field(memory, "created_at"), operation, details),
            _optional_string(_field(memory, "updated_at"), operation, details),
        ),
        float(score),
        float(similarity),
    )


class ContextMemoryClient:
    """Synchronous facade using the owning AgentCore portal and credential cache."""

    def __init__(
        self,
        client: AsyncContextMemoryClient,
        portal: BlockingPortal,
        operation: Callable[[], AbstractContextManager[None]],
    ) -> None:
        self._client, self._portal, self._operation = client, portal, operation

    @property
    def memory_store_name(self) -> str:
        return self._client.memory_store_name

    def create_events(
        self,
        events: Sequence[ContextEvent],
        *,
        scope: MemoryScope | None = None,
        timeout: float | None = None,
    ) -> CreateEventsResult:
        with self._operation():
            return self._portal.call(
                partial(self._client.create_events, events, scope=scope, timeout=timeout)
            )

    def list_events(
        self,
        *,
        scope: MemoryScope | None = None,
        next_token: str | None = None,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> EventPage:
        with self._operation():
            return self._portal.call(
                partial(
                    self._client.list_events,
                    scope=scope,
                    next_token=next_token,
                    max_results=max_results,
                    timeout=timeout,
                )
            )

    def search_memories(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        types: Sequence[MemoryType] | None = None,
        top_k: int | None = None,
        metadata: Mapping[str, str] | None = None,
        enable_rerank: bool | None = None,
        min_similarity: float | None = None,
        min_score: float | None = None,
        timeout: float | None = None,
    ) -> ContextMemorySearchResult:
        with self._operation():
            return self._portal.call(
                partial(
                    self._client.search_memories,
                    query,
                    scope=scope,
                    types=types,
                    top_k=top_k,
                    metadata=metadata,
                    enable_rerank=enable_rerank,
                    min_similarity=min_similarity,
                    min_score=min_score,
                    timeout=timeout,
                )
            )
