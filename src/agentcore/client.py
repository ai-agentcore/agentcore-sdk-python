"""Async-first AgentCore client and its long-lived synchronous portal."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Condition, RLock
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal

from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.agent_sa_token import AgentSATokenSource
from agentcore.auth.bound_credentials import AsyncBoundCredentials
from agentcore.auth.resource_sts import ResourceSTSProvider
from agentcore.auth.workload_access_token import WorkloadAccessTokenProvider
from agentcore.controlplane import AgentCoreControlPlane
from agentcore.controlplane.client import _normalize_endpoint as _normalize_control_plane_endpoint
from agentcore.errors import ResourceNotConfiguredError
from agentcore.integrations.common import Tool
from agentcore.mcp import AsyncMCPClient
from agentcore.mcp._headers import merge_mcp_headers
from agentcore.mcp.client import HeadersProvider as MCPHeadersProvider
from agentcore.memory._transport import _MemoryRuntime
from agentcore.memory.client import AsyncMemoryStore
from agentcore.memory.models import (
    AddMemoriesResult,
    Memory,
    MemoryMessage,
    MemoryScope,
    MemorySession,
    Page,
    SearchMemoriesResult,
)
from agentcore.model import AsyncModelClient
from agentcore.model.client import APIKeyProvider
from agentcore.model.client import HeadersProvider as ModelHeadersProvider
from agentcore.runtime.config import AgentConfig
from agentcore.runtime.startup import (
    DEBUG_TOKEN_ENV,
    DebugRuntimeSource,
    RuntimeBindings,
    RuntimeSource,
    UnconfiguredRuntimeSource,
    create_runtime_source,
)
from agentcore.skill import AsyncSkills

if TYPE_CHECKING:
    from agentcore_collaboration import Collaboration
    from agentcore_collaboration.sync import SyncCollaboration

T = TypeVar("T")
OperationScope = Callable[[], AbstractContextManager[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResourceContext:
    workspace_id: str
    region_id: str
    controller_endpoint: str | None
    control_plane_endpoint: str | None
    agent_sa_tokens: AgentSATokenSource | None


def _explicit_resource_context(
    workspace_id: str | None,
    region_id: str | None,
    *,
    control_plane_endpoint: str | None,
) -> _ResourceContext | None:
    if workspace_id is None and region_id is None:
        return None
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace_id and region_id must be provided together")
    if not isinstance(region_id, str) or not region_id.strip():
        raise ValueError("workspace_id and region_id must be provided together")
    return _ResourceContext(
        workspace_id=workspace_id.strip(),
        region_id=region_id.strip(),
        controller_endpoint=None,
        control_plane_endpoint=control_plane_endpoint,
        agent_sa_tokens=None,
    )


class AsyncCoreProtocol(Protocol):
    skills: Any
    credentials: Any
    collaboration: Any

    async def model(self, resource_name: str, *, model: str | None = None) -> Any: ...

    def direct_model(self, **kwargs: Any) -> Any: ...

    async def mcp(
        self,
        name: str,
        *,
        headers: Mapping[str, str] | None = None,
        credential_name: str | None = None,
    ) -> Any: ...

    def direct_mcp(self, **kwargs: Any) -> Any: ...

    def memory_store(self, memory_store_name: str) -> Any: ...

    async def aclose(self) -> None: ...


class AsyncAgentCore:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        env_path: str | Path | None = None,
        control_plane_endpoint: str | None = None,
        skill_workspace_dir: str | Path | None = None,
        teams_path: str | Path | None = None,
        collaboration_workspace_dir: str | Path | None = None,
        workspace_id: str | None = None,
        region_id: str | None = None,
        access_key_credential: AccessKeyCredential | None = None,
    ) -> None:
        explicit_context = _explicit_resource_context(
            workspace_id,
            region_id,
            control_plane_endpoint=control_plane_endpoint,
        )
        if explicit_context is not None:
            if config_path is not None or env_path is not None:
                raise ValueError(
                    "workspace_id and region_id cannot be combined with runtime file paths"
                )
            if os.getenv(DEBUG_TOKEN_ENV, "").strip():
                raise ValueError(
                    "workspace_id and region_id cannot be combined with AGENTCORE_DEBUG_TOKEN"
                )
            self._runtime_source: RuntimeSource = UnconfiguredRuntimeSource()
        else:
            self._runtime_source = create_runtime_source(
                config_path,
                env_path=env_path,
                control_plane_endpoint=control_plane_endpoint,
            )
        self._runtime: RuntimeBindings | None = self._runtime_source.initial
        self._runtime_lock = anyio.Lock()
        self.config: AgentConfig | None = (
            self._runtime.config if self._runtime is not None else None
        )
        self._resource_context = explicit_context
        self._access_key_credential = access_key_credential
        self._model_clients: dict[tuple[str, str | None], AsyncModelClient] = {}
        self._model_lock = anyio.Lock()
        self._direct_model_clients: list[AsyncModelClient] = []
        self._mcp_clients: dict[
            tuple[str, str | None, tuple[tuple[str, str], ...]], AsyncMCPClient
        ] = {}
        self._mcp_lock = anyio.Lock()
        self._direct_mcp_clients: list[AsyncMCPClient] = []
        self._wat: WorkloadAccessTokenProvider | None = None
        self._resource_sts: ResourceSTSProvider | None = None
        self._control_plane: AgentCoreControlPlane | None = None
        if self._runtime is not None:
            self._install_runtime(self._runtime)
        elif self._resource_context is not None:
            self._install_resource_context(self._resource_context)
        self._skills = AsyncSkills(
            None,
            None,
            workspace_dir=skill_workspace_dir,
            _runtime_provider=self._skill_runtime,
        )
        self._credentials = AsyncBoundCredentials(
            "",
            "",
            None,
            None,
            _runtime_provider=self._credential_runtime,
        )
        self._collaboration: Collaboration | None = None
        self._teams_path = teams_path
        self._collaboration_env_path = env_path
        self._collaboration_workspace_dir = collaboration_workspace_dir
        self._closed = False
        logger.info(
            "agentcore.client.created runtime_ready=%s resource_context_ready=%s "
            "explicit_access_key=%s",
            self._runtime is not None,
            self._resource_context is not None,
            access_key_credential is not None,
        )

    @classmethod
    def auto(
        cls,
        config_path: str | Path | None = None,
        *,
        env_path: str | Path | None = None,
        control_plane_endpoint: str | None = None,
        skill_workspace_dir: str | Path | None = None,
        teams_path: str | Path | None = None,
        collaboration_workspace_dir: str | Path | None = None,
        workspace_id: str | None = None,
        region_id: str | None = None,
        access_key_credential: AccessKeyCredential | None = None,
    ) -> AsyncAgentCore:
        return cls(
            config_path,
            env_path=env_path,
            control_plane_endpoint=control_plane_endpoint,
            skill_workspace_dir=skill_workspace_dir,
            teams_path=teams_path,
            collaboration_workspace_dir=collaboration_workspace_dir,
            workspace_id=workspace_id,
            region_id=region_id,
            access_key_credential=access_key_credential,
        )

    async def __aenter__(self) -> AsyncAgentCore:
        self._ensure_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def model(
        self,
        resource_name: str,
        *,
        model: str | None = None,
    ) -> AsyncModelClient:
        self._ensure_open()
        runtime = await self._ensure_runtime()
        config = runtime.config
        if self._control_plane is None:
            raise ResourceNotConfiguredError("AgentCore control-plane runtime is not configured")
        key = (resource_name, model)
        cached = self._model_clients.get(key)
        if cached is not None:
            logger.debug(
                "agentcore.client.model.cache.hit resource_name=%s model=%s",
                resource_name,
                model or "auto",
            )
            return cached
        async with self._model_lock:
            cached = self._model_clients.get(key)
            if cached is not None:
                logger.debug(
                    "agentcore.client.model.cache.hit resource_name=%s model=%s",
                    resource_name,
                    model or "auto",
                )
                return cached
            logger.info(
                "agentcore.client.model.create.started resource_name=%s model=%s",
                resource_name,
                model or "auto",
            )
            descriptor = await self._control_plane.resolve_model(resource_name, model)
            client = AsyncModelClient.platform(config, descriptor)
            self._model_clients[key] = client
            logger.info(
                "agentcore.client.model.create.succeeded resource_name=%s model=%s",
                resource_name,
                descriptor.model_name,
            )
            return client

    def direct_model(
        self,
        *,
        model: str,
        base_url: str,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_provider: APIKeyProvider | None = None,
        headers_provider: ModelHeadersProvider | None = None,
    ) -> AsyncModelClient:
        self._ensure_open()
        client = AsyncModelClient.direct(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            api_key_provider=api_key_provider,
            headers_provider=headers_provider,
        )
        self._direct_model_clients.append(client)
        logger.info(
            "agentcore.client.direct_model.created provider=%s model=%s",
            provider,
            model,
        )
        return client

    @property
    def skills(self) -> AsyncSkills:
        self._ensure_open()
        return self._skills

    @property
    def credentials(self) -> AsyncBoundCredentials:
        self._ensure_open()
        return self._credentials

    @property
    def collaboration(self) -> Collaboration:
        self._ensure_open()
        if self._collaboration is None:
            from agentcore.collaboration import Collaboration
            from agentcore.collaboration.debug import DebugCollaborationRuntime

            debug_collaboration = (
                DebugCollaborationRuntime(self._runtime_source)
                if isinstance(self._runtime_source, DebugRuntimeSource)
                else None
            )
            self._collaboration = Collaboration(
                teams_path=self._teams_path or "/var/run/agentcore/agent/teams.yaml",
                env_path=self._collaboration_env_path,
                workspace_dir=self._collaboration_workspace_dir,
                _debug_runtime=debug_collaboration,
                _use_debug_teams=self._teams_path is None,
            )
        return self._collaboration

    async def mcp(
        self,
        name: str,
        *,
        headers: Mapping[str, str] | None = None,
        credential_name: str | None = None,
    ) -> AsyncMCPClient:
        self._ensure_open()
        custom_headers = merge_mcp_headers({}, headers)
        if credential_name is not None:
            credential_name = credential_name.strip()
            if not credential_name:
                raise ValueError("credential name must not be empty")
        runtime = await self._ensure_runtime()
        config = runtime.config
        key = (name, credential_name, tuple(sorted(custom_headers.items())))
        cached = self._mcp_clients.get(key)
        if cached is not None:
            logger.debug("agentcore.client.mcp.cache.hit name=%s", name)
            return cached
        if self._control_plane is None:
            raise ResourceNotConfiguredError("AgentCore control-plane runtime is not configured")
        async with self._mcp_lock:
            cached = self._mcp_clients.get(key)
            if cached is not None:
                logger.debug("agentcore.client.mcp.cache.hit name=%s", name)
                return cached
            logger.info("agentcore.client.mcp.create.started name=%s", name)
            descriptor = await self._control_plane.resolve_mcp(name)

            async def credential_headers() -> Mapping[str, str]:
                assert credential_name is not None
                credential = await self._credentials._get(
                    credential_name,
                    mcp_server_id=descriptor.mcp_server_id,
                )
                return credential.as_headers()

            client = AsyncMCPClient.platform(
                config,
                descriptor,
                headers=custom_headers,
                credential_headers_provider=credential_headers
                if credential_name is not None
                else None,
            )
            self._mcp_clients[key] = client
            logger.info(
                "agentcore.client.mcp.create.succeeded name=%s mcp_server_id=%s",
                name,
                descriptor.mcp_server_id,
            )
            return client

    def direct_mcp(
        self,
        *,
        url: str,
        transport: str = "streamable-http",
        headers_provider: MCPHeadersProvider | None = None,
    ) -> AsyncMCPClient:
        self._ensure_open()
        client = AsyncMCPClient.direct(
            url=url,
            transport=transport,
            headers_provider=headers_provider,
        )
        self._direct_mcp_clients.append(client)
        logger.info("agentcore.client.direct_mcp.created transport=%s", transport)
        return client

    def memory_store(self, memory_store_name: str) -> AsyncMemoryStore:
        self._ensure_open()
        return AsyncMemoryStore(
            memory_store_name,
            _runtime_provider=self._memory_runtime,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info(
            "agentcore.client.close.started model_clients=%s mcp_clients=%s "
            "direct_models=%s direct_mcps=%s",
            len(self._model_clients),
            len(self._mcp_clients),
            len(self._direct_model_clients),
            len(self._direct_mcp_clients),
        )
        errors: list[BaseException] = []
        closers = [client.aclose for client in self._mcp_clients.values()]
        closers.extend(client.aclose for client in self._model_clients.values())
        closers.extend(client.aclose for client in self._direct_mcp_clients)
        closers.extend(client.aclose for client in self._direct_model_clients)
        if self._collaboration is not None:
            closers.append(self._collaboration.aclose)
        if self._resource_sts is not None:
            closers.append(self._resource_sts.aclose)
        if self._wat is not None:
            closers.append(self._wat.aclose)
        closers.append(self._runtime_source.aclose)
        for close in closers:
            try:
                await close()
            except BaseException as exc:
                errors.append(exc)
                logger.warning(
                    "agentcore.client.close.component_failed error_type=%s",
                    type(exc).__name__,
                )
        if errors:
            raise errors[0]
        logger.info("agentcore.client.close.succeeded")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentCore client is closed")

    async def _ensure_runtime(self) -> RuntimeBindings:
        if self._runtime is not None:
            return self._runtime
        async with self._runtime_lock:
            if self._runtime is None:
                logger.info("agentcore.client.runtime.resolve.started")
                runtime = await self._runtime_source.resolve()
                self._install_runtime(runtime)
                self._runtime = runtime
                logger.info("agentcore.client.runtime.resolve.succeeded")
            return self._runtime

    def _install_runtime(self, runtime: RuntimeBindings) -> None:
        self._install_resource_context(
            _ResourceContext(
                workspace_id=runtime.config.workspace_id,
                region_id=runtime.config.region_id,
                controller_endpoint=runtime.controller_endpoint,
                control_plane_endpoint=runtime.control_plane_endpoint,
                agent_sa_tokens=runtime.agent_sa_tokens,
            )
        )
        self.config = runtime.config

    def _install_resource_context(self, context: _ResourceContext) -> None:
        _normalize_control_plane_endpoint(context.control_plane_endpoint)
        auth_values = (context.controller_endpoint, context.agent_sa_tokens)
        if any(auth_values) and not all(auth_values):
            raise ValueError("AgentCore runtime bindings are incomplete")
        wat: WorkloadAccessTokenProvider | None = None
        resource_sts: ResourceSTSProvider | None = None
        if all(auth_values):
            assert context.controller_endpoint is not None
            assert context.agent_sa_tokens is not None
            wat = WorkloadAccessTokenProvider(
                context.controller_endpoint,
                context.agent_sa_tokens,
            )
            resource_sts = ResourceSTSProvider(
                context.controller_endpoint,
                context.agent_sa_tokens,
            )
        control_plane: AgentCoreControlPlane | None = None
        if self._access_key_credential is not None or resource_sts is not None:
            control_plane = AgentCoreControlPlane(
                workspace_id=context.workspace_id,
                region_id=context.region_id,
                resource_sts=resource_sts,
                access_key_credential=self._access_key_credential,
                endpoint=context.control_plane_endpoint,
            )
        self._resource_context = context
        self._wat = wat
        self._resource_sts = resource_sts
        self._control_plane = control_plane
        if control_plane is None:
            logger.info(
                "agentcore.client.resource_context.installed workspace_id=%s region_id=%s "
                "authenticated=false",
                context.workspace_id,
                context.region_id,
            )
            return
        logger.info(
            "agentcore.client.resource_context.installed workspace_id=%s region_id=%s "
            "access_key_configured=%s automatic_auth_configured=%s "
            "control_endpoint_override=%s",
            context.workspace_id,
            context.region_id,
            self._access_key_credential is not None,
            resource_sts is not None,
            context.control_plane_endpoint is not None,
        )

    async def _ensure_resource_context(self) -> _ResourceContext:
        if self._resource_context is not None:
            return self._resource_context
        await self._ensure_runtime()
        assert self._resource_context is not None
        return self._resource_context

    async def _skill_runtime(
        self,
    ) -> tuple[AgentCoreControlPlane | None, str | None]:
        context = await self._ensure_resource_context()
        return self._control_plane, context.workspace_id

    async def _credential_runtime(
        self,
    ) -> tuple[
        str,
        str,
        WorkloadAccessTokenProvider | None,
        ResourceSTSProvider | None,
        AgentCoreControlPlane | None,
    ]:
        context = await self._ensure_resource_context()
        return (
            context.workspace_id,
            context.region_id,
            self._wat,
            self._resource_sts,
            self._control_plane,
        )

    async def _memory_runtime(self) -> _MemoryRuntime:
        self._ensure_open()
        context = await self._ensure_resource_context()
        return _MemoryRuntime(
            workspace_id=context.workspace_id,
            region_id=context.region_id,
            endpoint=context.control_plane_endpoint,
            resource_sts=self._resource_sts,
            access_key_credential=self._access_key_credential,
        )


class AgentCore:
    """Synchronous facade backed by one AnyIO BlockingPortal."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        env_path: str | Path | None = None,
        control_plane_endpoint: str | None = None,
        skill_workspace_dir: str | Path | None = None,
        teams_path: str | Path | None = None,
        collaboration_workspace_dir: str | Path | None = None,
        workspace_id: str | None = None,
        region_id: str | None = None,
        access_key_credential: AccessKeyCredential | None = None,
    ) -> None:
        self._factory: Callable[[], AsyncCoreProtocol] = cast(
            Callable[[], AsyncCoreProtocol],
            partial(
                AsyncAgentCore.auto,
                config_path,
                env_path=env_path,
                control_plane_endpoint=control_plane_endpoint,
                skill_workspace_dir=skill_workspace_dir,
                teams_path=teams_path,
                collaboration_workspace_dir=collaboration_workspace_dir,
                workspace_id=workspace_id,
                region_id=region_id,
                access_key_credential=access_key_credential,
            ),
        )
        self._portal_context: Any = None
        self._portal: BlockingPortal | None = None
        self._async_core: AsyncCoreProtocol | None = None
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._active_calls = 0
        self._closing = False
        self._closed = False

    @classmethod
    def auto(
        cls,
        config_path: str | Path | None = None,
        *,
        env_path: str | Path | None = None,
        control_plane_endpoint: str | None = None,
        skill_workspace_dir: str | Path | None = None,
        teams_path: str | Path | None = None,
        collaboration_workspace_dir: str | Path | None = None,
        workspace_id: str | None = None,
        region_id: str | None = None,
        access_key_credential: AccessKeyCredential | None = None,
    ) -> AgentCore:
        return cls(
            config_path,
            env_path=env_path,
            control_plane_endpoint=control_plane_endpoint,
            skill_workspace_dir=skill_workspace_dir,
            teams_path=teams_path,
            collaboration_workspace_dir=collaboration_workspace_dir,
            workspace_id=workspace_id,
            region_id=region_id,
            access_key_credential=access_key_credential,
        )

    def __enter__(self) -> AgentCore:
        self._start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def model(self, resource_name: str, *, model: str | None = None) -> _SyncModel:
        with self._lock:
            async_core = self._get()
            resolved = self._portal_required().call(
                partial(async_core.model, resource_name, model=model)
            )
            return _SyncModel(resolved, self._portal_required(), self._operation)

    def direct_model(
        self,
        *,
        model: str,
        base_url: str,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_provider: APIKeyProvider | None = None,
        headers_provider: ModelHeadersProvider | None = None,
    ) -> _SyncModel:
        with self._lock:
            return _SyncModel(
                self._get().direct_model(
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                    api_key_provider=api_key_provider,
                    headers_provider=headers_provider,
                ),
                self._portal_required(),
                self._operation,
            )

    def mcp(
        self,
        name: str,
        *,
        headers: Mapping[str, str] | None = None,
        credential_name: str | None = None,
    ) -> _SyncMCP:
        with self._lock:
            async_core = self._get()
            resolved = self._portal_required().call(
                partial(
                    async_core.mcp,
                    name,
                    headers=headers,
                    credential_name=credential_name,
                )
            )
            return _SyncMCP(resolved, self._portal_required(), self._operation)

    def direct_mcp(self, **kwargs: Any) -> _SyncMCP:
        with self._lock:
            return _SyncMCP(
                self._get().direct_mcp(**kwargs),
                self._portal_required(),
                self._operation,
            )

    def memory_store(self, memory_store_name: str) -> _SyncMemoryStore:
        with self._lock:
            store = self._get().memory_store(memory_store_name)
            return _SyncMemoryStore(
                store,
                self._portal_required(),
                self._operation,
            )

    @property
    def skills(self) -> _SyncSkills:
        with self._lock:
            return _SyncSkills(self._get().skills, self._portal_required(), self._operation)

    @property
    def credentials(self) -> _SyncCredentials:
        with self._lock:
            return _SyncCredentials(
                self._get().credentials, self._portal_required(), self._operation
            )

    @property
    def collaboration(self) -> SyncCollaboration:
        with self._lock:
            collaboration = self._get().collaboration
            from agentcore_collaboration.sync import SyncCollaboration

            return SyncCollaboration(
                collaboration,
                self._portal_required(),
                self._operation,
            )

    def close(self) -> None:
        with self._condition:
            while self._closing:
                self._condition.wait()
            if self._closed:
                return
            self._closed = True
            if self._portal is None or self._async_core is None:
                return
            self._closing = True
            logger.info("agentcore.sync_client.close.started active_calls=%s", self._active_calls)
            while self._active_calls:
                self._condition.wait()
            context = self._portal_context
            portal = self._portal
            async_core = self._async_core
        try:
            portal.call(async_core.aclose)
        finally:
            try:
                context.__exit__(None, None, None)
            finally:
                with self._condition:
                    self._portal = None
                    self._async_core = None
                    self._portal_context = None
                    self._closing = False
                    self._condition.notify_all()
                    logger.info("agentcore.sync_client.close.succeeded")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._condition:
            if self._closed or self._closing or self._portal is None:
                raise RuntimeError("AgentCore client is closed")
            self._active_calls += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._condition.notify_all()

    def _start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("AgentCore client is closed")
            if self._portal is not None:
                return
            context = start_blocking_portal(backend="asyncio", name="agentcore-sdk")
            logger.info("agentcore.sync_client.portal.started")
            portal = context.__enter__()
            try:
                async_core = portal.call(_create_async_core, self._factory)
            except BaseException as exc:
                logger.warning(
                    "agentcore.sync_client.portal.failed error_type=%s",
                    type(exc).__name__,
                )
                context.__exit__(type(exc), exc, exc.__traceback__)
                raise
            self._portal_context = context
            self._portal = portal
            self._async_core = async_core

    def _get(self) -> AsyncCoreProtocol:
        self._start()
        assert self._async_core is not None
        return self._async_core

    def _portal_required(self) -> BlockingPortal:
        self._start()
        assert self._portal is not None
        return self._portal


async def _create_async_core(factory: Callable[[], AsyncCoreProtocol]) -> AsyncCoreProtocol:
    value = factory()
    if inspect.isawaitable(value):
        value = await value
    return value


class _SyncMemoryStore:
    def __init__(
        self,
        store: AsyncMemoryStore,
        portal: BlockingPortal,
        operation: OperationScope,
    ) -> None:
        self._store = store
        self._portal = portal
        self._operation = operation

    @property
    def memory_store_name(self) -> str:
        return self._store.memory_store_name

    def add_memories(
        self,
        *,
        scope: MemoryScope | None = None,
        text: str | None = None,
        messages: Sequence[MemoryMessage] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> AddMemoriesResult:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.add_memories,
                    scope=scope,
                    text=text,
                    messages=messages,
                    metadata=metadata,
                )
            )

    def search_memories(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        top_k: int | None = None,
        metadata: Mapping[str, str] | None = None,
        enable_rerank: bool | None = None,
        min_similarity: float | None = None,
        min_score: float | None = None,
    ) -> SearchMemoriesResult:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.search_memories,
                    query,
                    scope=scope,
                    top_k=top_k,
                    metadata=metadata,
                    enable_rerank=enable_rerank,
                    min_similarity=min_similarity,
                    min_score=min_score,
                )
            )

    def list_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[Memory]:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.list_memories,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    max_results=max_results,
                    next_token=next_token,
                )
            )

    def get_memory(self, memory_id: str) -> Memory:
        with self._operation():
            return self._portal.call(self._store.get_memory, memory_id)

    def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Memory:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.update_memory,
                    memory_id,
                    text=text,
                    metadata=metadata,
                )
            )

    def delete_memory(self, memory_id: str) -> None:
        with self._operation():
            self._portal.call(self._store.delete_memory, memory_id)

    def list_memory_sessions(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[MemorySession]:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.list_memory_sessions,
                    user_id=user_id,
                    agent_id=agent_id,
                    max_results=max_results,
                    next_token=next_token,
                )
            )

    def list_memory_session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[MemoryMessage]:
        with self._operation():
            return self._portal.call(
                partial(
                    self._store.list_memory_session_messages,
                    session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    max_results=max_results,
                    next_token=next_token,
                )
            )


class _SyncModel:
    def __init__(self, model: Any, portal: BlockingPortal, operation: OperationScope) -> None:
        self._model = model
        self._portal = portal
        self._operation = operation

    @property
    def descriptor(self) -> Any:
        return getattr(self._model, "descriptor", None)

    def invoke(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        with self._operation():
            return cast(
                dict[str, Any],
                self._portal.call(partial(self._model.invoke, messages, **kwargs)),
            )

    def completion(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        with self._operation():
            return cast(
                dict[str, Any],
                self._portal.call(partial(self._model.completion, messages, **kwargs)),
            )

    def responses(self, input: Any, **kwargs: Any) -> dict[str, Any]:
        with self._operation():
            return cast(
                dict[str, Any],
                self._portal.call(partial(self._model.responses, input, **kwargs)),
            )

    def responses_stream(self, input: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        with self._operation():
            iterator = self._model.responses_stream(input, **kwargs)
            return _SyncAsyncIterator(iterator, self._portal, self._operation)

    def embedding(self, input: Any, **kwargs: Any) -> dict[str, Any]:
        with self._operation():
            return cast(
                dict[str, Any],
                self._portal.call(partial(self._model.embedding, input, **kwargs)),
            )

    def stream(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        with self._operation():
            iterator = self._model.stream(messages, **kwargs)
            return _SyncAsyncIterator(iterator, self._portal, self._operation)


class _SyncMCP:
    def __init__(self, client: Any, portal: BlockingPortal, operation: OperationScope) -> None:
        self._client = client
        self._portal = portal
        self._operation = operation

    @property
    def descriptor(self) -> Any:
        return getattr(self._client, "descriptor", None)

    def tools(self) -> list[Any]:
        with self._operation():
            values = cast(list[Any], self._portal.call(self._client.tools))
        return [
            value.with_sync_call(partial(self.call_tool, value.name))
            if isinstance(value, Tool)
            else value
            for value in values
        ]

    def list_tools(self) -> list[Any]:
        return self.tools()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        with self._operation():
            return self._portal.call(partial(self._client.call_tool, name, arguments))


class _SyncSkills:
    def __init__(self, skills: Any, portal: BlockingPortal, operation: OperationScope) -> None:
        self._skills = skills
        self._portal = portal
        self._operation = operation

    def local(self, root: str | Path) -> list[Any]:
        with self._operation():
            return cast(list[Any], self._portal.call(self._skills.local, root))

    def managed(self, name: str, *, version: str | None = None) -> Any:
        with self._operation():
            return self._portal.call(partial(self._skills.managed, name, version=version))


class _SyncCredentials:
    def __init__(self, credentials: Any, portal: BlockingPortal, operation: OperationScope) -> None:
        self._credentials = credentials
        self._portal = portal
        self._operation = operation

    def get(self, credential_name: str) -> Any:
        with self._operation():
            return self._portal.call(self._credentials.get, credential_name)


class _SyncAsyncIterator(Iterator[T]):
    def __init__(
        self, iterator: AsyncIterator[T], portal: BlockingPortal, operation: OperationScope
    ) -> None:
        self._iterator = iterator
        self._portal = portal
        self._operation = operation
        self._closed = False

    def __iter__(self) -> _SyncAsyncIterator[T]:
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        with self._operation():
            try:
                return self._portal.call(self._iterator.__anext__)
            except StopAsyncIteration:
                self.close()
                raise StopIteration from None

    def close(self) -> None:
        if self._closed:
            return
        with self._operation():
            self._closed = True
            close = getattr(self._iterator, "aclose", None)
            if close is not None:
                self._portal.call(close)

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
