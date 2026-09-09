"""Private POP transport and response projection for Memory data-plane calls."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol, TypeGuard
from urllib.parse import quote

from alibabacloud_agentcore20260804.client import Client
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_openapi import utils_models as openapi_utils
from alibabacloud_tea_util.models import RuntimeOptions  # type: ignore[import-untyped]

from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.resource_sts import HIGH_CODE_SDK_PURPOSE, ResourceCredential
from agentcore.controlplane.client import _normalize_endpoint
from agentcore.errors import (
    AddMemoriesOutcomeUnknownError,
    MemoryAPIError,
    MemoryContractError,
    ResourceNotConfiguredError,
)
from agentcore.memory.models import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    MemorySession,
    Page,
    SearchMemoriesResult,
)

logger = logging.getLogger(__name__)


class _STSProvider(Protocol):
    async def get(self, purpose: str | None = None) -> ResourceCredential: ...


@dataclass(frozen=True)
class _MemoryRuntime:
    workspace_id: str
    region_id: str
    endpoint: str | None
    resource_sts: _STSProvider | None
    access_key_credential: AccessKeyCredential | None = None


@dataclass(frozen=True)
class _ErrorDetails:
    service_code: str | None = None
    http_status_code: int | None = None
    request_id: str | None = None


_RuntimeProvider = Callable[[], Awaitable[_MemoryRuntime]]
_MemoryCredential = AccessKeyCredential | ResourceCredential
_ClientFactory = Callable[[_MemoryRuntime, _MemoryCredential], Any]
_OpenAPICall = Callable[[], Awaitable[Any]]


class _MemoryTransport:
    def __init__(
        self,
        memory_store_name: str,
        runtime_provider: _RuntimeProvider,
        *,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        self._memory_store_name = memory_store_name
        self._runtime_provider = runtime_provider
        self._client_factory = client_factory or self._new_client

    async def add_memories(
        self,
        *,
        scope: MemoryScope | None,
        text: str | None,
        messages: Sequence[MemoryMessage] | None,
        metadata: Mapping[str, str] | None,
    ) -> AddMemoriesResult:
        response = await self._request(
            "AddMemories",
            "POST",
            "/memories",
            body={
                "scope": _scope_body(scope),
                "text": text,
                "messages": (
                    [{"role": m.role, "content": m.content} for m in messages]
                    if messages is not None
                    else None
                ),
                "metadata": dict(metadata) if metadata is not None else None,
            },
            add=True,
        )
        body, details = self._success_body(response, "AddMemories", add=True)
        data = _field(body, "data")
        raw_memories = _field(data, "memories") if data is not None else None
        if raw_memories is None:
            return AddMemoriesResult()
        if not _is_sequence(raw_memories):
            self._invalid_response("AddMemories", "data.memories must be a list", details)
        memory_ids: list[str] = []
        for item in raw_memories:
            memory_id = _field(item, "memory_id")
            if not _non_empty_string(memory_id):
                self._invalid_response(
                    "AddMemories",
                    "data.memories[].memory_id must be a non-empty string",
                    details,
                )
            memory_ids.append(memory_id)
        return AddMemoriesResult(tuple(memory_ids))

    async def search_memories(
        self,
        query: str,
        *,
        scope: MemoryScope | None,
        top_k: int | None,
        metadata: Mapping[str, str] | None,
        enable_rerank: bool | None,
        min_similarity: float | None,
        min_score: float | None,
    ) -> SearchMemoriesResult:
        response = await self._request(
            "SearchMemories",
            "POST",
            "/memories/search",
            body={
                "query": query,
                "scope": _scope_body(scope),
                "topK": top_k,
                "metadata": dict(metadata) if metadata is not None else None,
                "enableRerank": enable_rerank,
                "minSimilarity": min_similarity,
                "minScore": min_score,
            },
        )
        body, _ = self._success_body(response, "SearchMemories")
        data = _field(body, "data")
        raw_hits = _field(data, "memories") if data is not None else None
        if raw_hits is None:
            return SearchMemoriesResult()
        if not _is_sequence(raw_hits):
            self._contract("SearchMemories", "data.memories must be a list")
        hits: list[MemorySearchHit] = []
        for item in raw_hits:
            score = _field(item, "score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                self._contract("SearchMemories", "data.memories[].score must be a number")
            similarity = _field(item, "similarity")
            if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
                self._contract("SearchMemories", "data.memories[].similarity must be a number")
            hits.append(
                MemorySearchHit(
                    memory=self._memory(
                        _field(item, "memory"),
                        "SearchMemories",
                        "data.memories[].memory",
                    ),
                    score=float(score),
                    similarity=float(similarity),
                )
            )
        return SearchMemoriesResult(tuple(hits))

    async def list_memories(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        max_results: int | None,
        next_token: str | None,
    ) -> Page[Memory]:
        response = await self._request(
            "ListMemories",
            "GET",
            "/memories",
            query={
                "agentId": agent_id,
                "sessionId": session_id,
                "userId": user_id,
                "maxResults": max_results,
                "nextToken": next_token,
            },
        )
        body, _ = self._success_body(response, "ListMemories")
        return self._page(
            body,
            "ListMemories",
            lambda item: self._memory(item, "ListMemories", "items[]"),
        )

    async def get_memory(self, memory_id: str) -> Memory:
        response = await self._request(
            "GetMemory",
            "GET",
            f"/memories/{quote(memory_id, safe='')}",
            memory_id=memory_id,
        )
        body, _ = self._success_body(response, "GetMemory", memory_id=memory_id)
        return self._memory(_field(body, "data"), "GetMemory", "data", memory_id)

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None,
        metadata: Mapping[str, str] | None,
    ) -> Memory:
        response = await self._request(
            "UpdateMemory",
            "PUT",
            f"/memories/{quote(memory_id, safe='')}",
            body={"text": text, "metadata": dict(metadata) if metadata is not None else None},
            memory_id=memory_id,
        )
        body, _ = self._success_body(response, "UpdateMemory", memory_id=memory_id)
        return self._memory(_field(body, "data"), "UpdateMemory", "data", memory_id)

    async def delete_memory(self, memory_id: str) -> None:
        response = await self._request(
            "DeleteMemory",
            "DELETE",
            f"/memories/{quote(memory_id, safe='')}",
            memory_id=memory_id,
        )
        self._success_body(response, "DeleteMemory", memory_id=memory_id)

    async def list_memory_sessions(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        max_results: int | None,
        next_token: str | None,
    ) -> Page[MemorySession]:
        response = await self._request(
            "ListMemorySessions",
            "GET",
            "/sessions",
            query={
                "agentId": agent_id,
                "userId": user_id,
                "maxResults": max_results,
                "nextToken": next_token,
            },
        )
        body, _ = self._success_body(response, "ListMemorySessions")
        return self._page(body, "ListMemorySessions", self._memory_session)

    async def list_memory_session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None,
        agent_id: str | None,
        max_results: int | None,
        next_token: str | None,
    ) -> Page[MemoryMessage]:
        response = await self._request(
            "ListMemorySessionMessages",
            "GET",
            "/messages",
            query={
                "agentId": agent_id,
                "userId": user_id,
                "sessionId": session_id,
                "maxResults": max_results,
                "nextToken": next_token,
            },
        )
        body, _ = self._success_body(response, "ListMemorySessionMessages")
        return self._page(body, "ListMemorySessionMessages", self._memory_message)

    async def _request(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        add: bool = False,
        memory_id: str | None = None,
    ) -> Any:
        runtime, client = await self._request_client()
        params = openapi_utils.Params(
            action=operation,
            version="2026-08-04",
            protocol="HTTPS",
            pathname=(
                f"/workspaces/{quote(runtime.workspace_id, safe='')}"
                f"/memorystores/{quote(self._memory_store_name, safe='')}{path}"
            ),
            method=method,
            auth_type="AK",
            style="ROA",
            req_body_type="formData" if body is not None else "json",
            body_type="json",
        )
        # Match the OpenAPI contract: a form field named body contains JSON.
        request = openapi_utils.OpenApiRequest(
            headers={},
            body={"body": json.dumps(_without_none(body))} if body is not None else None,
            query=(
                {key: str(value) for key, value in query.items() if value is not None}
                if query is not None
                else None
            ),
        )
        return await self._invoke(
            operation,
            lambda: client.call_api_async(params, request, _runtime_options(add=add)),
            add=add,
            memory_id=memory_id,
        )

    async def _request_client(self) -> tuple[_MemoryRuntime, Any]:
        runtime = await self._runtime_provider()
        credential: _MemoryCredential
        if runtime.access_key_credential is not None:
            credential = runtime.access_key_credential
        elif runtime.resource_sts is not None:
            credential = await runtime.resource_sts.get(HIGH_CODE_SDK_PURPOSE)
        else:
            raise ResourceNotConfiguredError(
                "AgentCore Memory data-plane credentials are not configured"
            )
        return runtime, self._client_factory(runtime, credential)

    async def _invoke(
        self,
        operation: str,
        call: _OpenAPICall,
        *,
        add: bool = False,
        memory_id: str | None = None,
    ) -> Any:
        response: Any = None
        failure: MemoryAPIError | None = None
        failure_type: str | None = None
        try:
            response = await call()
        except Exception as exc:
            details = _exception_details(exc)
            failure_type = type(exc).__name__
            if add and (details.http_status_code is None or details.http_status_code >= 500):
                failure = AddMemoriesOutcomeUnknownError(
                    operation,
                    service_code=details.service_code,
                    http_status_code=details.http_status_code,
                    request_id=details.request_id,
                )
            else:
                failure = MemoryAPIError(
                    operation,
                    service_code=details.service_code,
                    http_status_code=details.http_status_code,
                    request_id=details.request_id,
                )
        if failure is not None:
            self._log_failure(failure, memory_id=memory_id, exception_type=failure_type)
            raise failure
        return response

    def _success_body(
        self,
        response: Any,
        operation: str,
        *,
        add: bool = False,
        memory_id: str | None = None,
    ) -> tuple[Any, _ErrorDetails]:
        if response is None:
            return self._malformed(operation, "response is missing", add, memory_id)
        body = _field(response, "body")
        if body is None:
            return self._malformed(operation, "response body is missing", add, memory_id)
        details = _response_details(response, body)
        response_status = _optional_int(_field(response, "status_code"))
        if response_status is not None and response_status >= 400:
            details = _ErrorDetails(
                service_code=details.service_code,
                http_status_code=details.http_status_code or response_status,
                request_id=details.request_id,
            )
            self._raise_api_failure(operation, details, add=add, memory_id=memory_id)
        success = _field(body, "success")
        if success is False:
            self._raise_api_failure(operation, details, add=add, memory_id=memory_id)
        if success is not True:
            return self._malformed(
                operation,
                "body.success must be true",
                add,
                memory_id,
                details,
            )
        return body, details

    def _raise_api_failure(
        self,
        operation: str,
        details: _ErrorDetails,
        *,
        add: bool,
        memory_id: str | None,
    ) -> NoReturn:
        if add and details.http_status_code is not None and details.http_status_code >= 500:
            error: MemoryAPIError = AddMemoriesOutcomeUnknownError(
                operation,
                service_code=details.service_code,
                http_status_code=details.http_status_code,
                request_id=details.request_id,
            )
        else:
            error = MemoryAPIError(
                operation,
                service_code=details.service_code,
                http_status_code=details.http_status_code,
                request_id=details.request_id,
            )
        self._log_failure(error, memory_id=memory_id)
        raise error

    def _malformed(
        self,
        operation: str,
        detail: str,
        add: bool,
        memory_id: str | None,
        details: _ErrorDetails | None = None,
    ) -> NoReturn:
        details = details or _ErrorDetails()
        if add:
            self._invalid_response(operation, detail, details, memory_id)
        self._contract(operation, detail, memory_id)

    def _invalid_response(
        self,
        operation: str,
        detail: str,
        details: _ErrorDetails,
        memory_id: str | None = None,
    ) -> NoReturn:
        error = AddMemoriesOutcomeUnknownError(
            operation,
            service_code=details.service_code,
            http_status_code=details.http_status_code,
            request_id=details.request_id,
        )
        self._log_failure(error, memory_id=memory_id, exception_type="InvalidResponse")
        raise error

    def _contract(
        self,
        operation: str,
        detail: str,
        memory_id: str | None = None,
    ) -> NoReturn:
        error = MemoryContractError(operation, detail)
        self._log_failure(error, memory_id=memory_id, exception_type="InvalidResponse")
        raise error

    def _memory(
        self,
        value: Any,
        operation: str,
        path: str,
        memory_id_for_log: str | None = None,
    ) -> Memory:
        if value is None:
            self._contract(operation, f"{path} is missing", memory_id_for_log)
        memory_id = self._required_string(
            _field(value, "memory_id"), operation, f"{path}.memory_id", memory_id_for_log
        )
        content = _field(value, "content")
        if content is None:
            self._contract(operation, f"{path}.content is missing", memory_id_for_log)
        text = self._string(
            _field(content, "text"), operation, f"{path}.content.text", memory_id_for_log
        )
        scope_value = _field(value, "scope")
        if scope_value is None:
            self._contract(operation, f"{path}.scope is missing", memory_id_for_log)
        scope = MemoryScope(
            agent_id=self._optional_string(
                _field(scope_value, "agent_id"),
                operation,
                f"{path}.scope.agent_id",
                memory_id_for_log,
            ),
            session_id=self._optional_string(
                _field(scope_value, "session_id"),
                operation,
                f"{path}.scope.session_id",
                memory_id_for_log,
            ),
            user_id=self._optional_string(
                _field(scope_value, "user_id"),
                operation,
                f"{path}.scope.user_id",
                memory_id_for_log,
            ),
        )
        return Memory(
            memory_id=memory_id,
            content=MemoryContent(text=text),
            scope=scope,
            metadata=self._metadata(
                _field(value, "metadata"), operation, f"{path}.metadata", memory_id_for_log
            ),
            created_at=self._optional_string(
                _field(value, "created_at"),
                operation,
                f"{path}.created_at",
                memory_id_for_log,
            ),
            updated_at=self._optional_string(
                _field(value, "updated_at"),
                operation,
                f"{path}.updated_at",
                memory_id_for_log,
            ),
        )

    def _memory_session(self, value: Any) -> MemorySession:
        session = MemorySession(
            agent_id=self._optional_string(
                _field(value, "agent_id"),
                "ListMemorySessions",
                "items[].agent_id",
            ),
            session_id=self._optional_string(
                _field(value, "session_id"),
                "ListMemorySessions",
                "items[].session_id",
            ),
            user_id=self._optional_string(
                _field(value, "user_id"),
                "ListMemorySessions",
                "items[].user_id",
            ),
        )
        if session.user_id is None and session.agent_id is None and session.session_id is None:
            self._contract(
                "ListMemorySessions",
                "items[] must contain at least one scope field",
            )
        return session

    def _memory_message(self, value: Any) -> MemoryMessage:
        return MemoryMessage(
            role=self._required_string(
                _field(value, "role"),
                "ListMemorySessionMessages",
                "items[].role",
            ),
            content=self._required_string(
                _field(value, "content"),
                "ListMemorySessionMessages",
                "items[].content",
            ),
        )

    def _page(
        self,
        body: Any,
        operation: str,
        item_mapper: Callable[[Any], Any],
    ) -> Page[Any]:
        raw_items = _field(body, "items")
        if raw_items is None:
            raw_items = []
        if not _is_sequence(raw_items):
            self._contract(operation, "body.items must be a list")
        items = tuple(item_mapper(item) for item in raw_items)
        max_results = self._optional_integer(
            _field(body, "max_results"), operation, "body.max_results"
        )
        total_count = self._optional_integer(
            _field(body, "total_count"), operation, "body.total_count"
        )
        raw_next_token = _field(body, "next_token")
        if raw_next_token is not None and not isinstance(raw_next_token, str):
            self._contract(operation, "body.next_token must be a string")
        next_token = raw_next_token or None
        return Page(
            items=items,
            max_results=max_results,
            next_token=next_token,
            total_count=total_count,
        )

    def _required_string(
        self,
        value: Any,
        operation: str,
        path: str,
        memory_id: str | None = None,
    ) -> str:
        if not _non_empty_string(value):
            self._contract(operation, f"{path} must be a non-empty string", memory_id)
        return value

    def _string(
        self,
        value: Any,
        operation: str,
        path: str,
        memory_id: str | None = None,
    ) -> str:
        if not isinstance(value, str):
            self._contract(operation, f"{path} must be a string", memory_id)
        return value

    def _optional_string(
        self,
        value: Any,
        operation: str,
        path: str,
        memory_id: str | None = None,
    ) -> str | None:
        if value is not None and not isinstance(value, str):
            self._contract(operation, f"{path} must be a string", memory_id)
        return value

    def _optional_integer(self, value: Any, operation: str, path: str) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            self._contract(operation, f"{path} must be an integer")
        return value

    def _metadata(
        self,
        value: Any,
        operation: str,
        path: str,
        memory_id: str | None = None,
    ) -> Mapping[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            self._contract(operation, f"{path} must contain string keys and values", memory_id)
        return dict(value)

    def _log_failure(
        self,
        error: Exception,
        *,
        memory_id: str | None = None,
        exception_type: str | None = None,
    ) -> None:
        logger.warning(
            "agentcore.memory.request.failed operation=%s memory_store_name=%s "
            "memory_id=%s status=%s service_code=%s request_id=%s error_type=%s",
            getattr(error, "operation", "unknown"),
            self._memory_store_name,
            memory_id or "-",
            getattr(error, "http_status_code", None) or "-",
            getattr(error, "service_code", None) or "-",
            getattr(error, "request_id", None) or "-",
            exception_type or type(error).__name__,
        )

    @staticmethod
    def _new_client(runtime: _MemoryRuntime, credential: _MemoryCredential) -> Client:
        endpoint, protocol = _normalize_endpoint(runtime.endpoint)
        values: dict[str, Any] = {
            "access_key_id": credential.access_key_id,
            "access_key_secret": credential.access_key_secret,
            "region_id": runtime.region_id,
            "endpoint": endpoint,
            "protocol": protocol,
        }
        if credential.security_token is not None:
            values["security_token"] = credential.security_token
        return Client(openapi_models.Config(**values))


def _runtime_options(*, add: bool = False) -> RuntimeOptions:
    return RuntimeOptions(
        autoretry=False,
        max_attempts=1,
        read_timeout=120_000 if add else None,
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        head, *tail = name.split("_")
        return value.get(head + "".join(part.title() for part in tail))
    return getattr(value, name, None)


def _without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _scope_body(scope: MemoryScope | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    return _without_none(
        {"agentId": scope.agent_id, "sessionId": scope.session_id, "userId": scope.user_id}
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _non_empty_string(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _safe_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _exception_details(exc: Exception) -> _ErrorDetails:
    service_code: str | None = None
    status: int | None = None
    request_id: str | None = None
    current: object | None = exc
    seen: set[int] = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        service_code = service_code or _safe_string(getattr(current, "code", None))
        status = status or _optional_int(getattr(current, "status_code", None))
        status = status or _optional_int(getattr(current, "statusCode", None))
        request_id = request_id or _safe_string(getattr(current, "request_id", None))
        request_id = request_id or _safe_string(getattr(current, "requestId", None))
        data = getattr(current, "data", None)
        if isinstance(data, Mapping):
            service_code = service_code or _safe_string(data.get("code"))
            service_code = service_code or _safe_string(data.get("Code"))
            status = status or _optional_int(data.get("statusCode"))
            status = status or _optional_int(data.get("StatusCode"))
            request_id = request_id or _safe_string(data.get("requestId"))
            request_id = request_id or _safe_string(data.get("RequestId"))
        response = getattr(current, "response", None)
        status = status or _optional_int(_field(response, "status_code"))
        headers = _field(response, "headers")
        if request_id is None and isinstance(headers, Mapping):
            request_id = _safe_string(
                headers.get("x-acs-request-id") or headers.get("X-Acs-Request-Id")
            )
        current = getattr(current, "inner_exception", None)
    return _ErrorDetails(
        service_code=service_code,
        http_status_code=status,
        request_id=request_id,
    )


def _response_details(response: Any, body: Any) -> _ErrorDetails:
    status = _optional_int(_field(body, "http_status_code"))
    if status is None:
        status = _optional_int(_field(response, "status_code"))
    return _ErrorDetails(
        service_code=_safe_string(_field(body, "code")),
        http_status_code=status,
        request_id=_safe_string(_field(body, "request_id")),
    )
