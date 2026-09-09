"""Managed AgentCore Gateway and user-supplied direct model clients."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from agentcore._logging import safe_url as _safe_url
from agentcore.controlplane import ModelDescriptor
from agentcore.errors import ConfigError, InvocationError
from agentcore.runtime.config import AgentConfig

APIKeyProvider = Callable[[], str | Awaitable[str]]
HeadersProvider = Callable[[], Mapping[str, str] | Awaitable[Mapping[str, str]]]
logger = logging.getLogger(__name__)


class _ModelBackend(Protocol):
    async def completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]: ...

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def responses(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]: ...

    def responses_stream(
        self,
        input: Any,
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def embedding(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class AsyncModelClient:
    """One model API backed by either AgentCore Gateway or a direct LiteLLM route."""

    def __init__(
        self,
        config: AgentConfig,
        descriptor: ModelDescriptor,
        *,
        timeout: float = 600.0,
    ) -> None:
        self.descriptor: ModelDescriptor | None = descriptor
        self._backend: _ModelBackend = _managed_gateway_backend(
            config,
            descriptor,
            timeout=timeout,
        )

    @classmethod
    def platform(
        cls,
        config: AgentConfig,
        descriptor: ModelDescriptor,
        *,
        timeout: float = 600.0,
    ) -> AsyncModelClient:
        return cls(config, descriptor, timeout=timeout)

    @classmethod
    def direct(
        cls,
        *,
        model: str,
        base_url: str,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_provider: APIKeyProvider | None = None,
        headers_provider: HeadersProvider | None = None,
    ) -> AsyncModelClient:
        client = cls.__new__(cls)
        client.descriptor = None
        client._backend = _LiteLLMDirectBackend(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            api_key_provider=api_key_provider,
            headers_provider=headers_provider,
        )
        return client

    async def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        return await self.completion(messages, **parameters)

    async def completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        return await self._backend.completion(messages, **parameters)

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._backend.stream(messages, **parameters):
            yield chunk

    async def responses(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        return await self._backend.responses(input, **parameters)

    async def responses_stream(
        self,
        input: Any,
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in self._backend.responses_stream(input, **parameters):
            yield event

    async def embedding(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        return await self._backend.embedding(input, **parameters)

    async def aclose(self) -> None:
        await self._backend.aclose()

    def __repr__(self) -> str:
        return f"AsyncModelClient(backend={self._backend!r})"


@dataclass(frozen=True)
class _ManagedProtocolSpec:
    value: str
    framework_provider: str
    runtime_base_path: str


_OPENAI_V1 = _ManagedProtocolSpec(
    value="OpenAI/v1",
    framework_provider="openai",
    runtime_base_path="/v1",
)
_ANTHROPIC = _ManagedProtocolSpec(
    value="Anthropic",
    framework_provider="anthropic",
    runtime_base_path="",
)
_MANAGED_PROTOCOL_SPECS = {
    "openai/v1": _OPENAI_V1,
    "anthropic": _ANTHROPIC,
}


def _managed_gateway_backend(
    config: AgentConfig,
    descriptor: ModelDescriptor,
    *,
    timeout: float,
) -> _ModelBackend:
    protocol = _managed_model_protocol(descriptor.protocol)
    if protocol is _OPENAI_V1:
        return _OpenAIV1GatewayBackend(
            config,
            descriptor,
            protocol=protocol,
            timeout=timeout,
        )
    if protocol is _ANTHROPIC:
        return _AnthropicGatewayBackend(
            config,
            descriptor,
            protocol=protocol,
            timeout=timeout,
        )
    raise ConfigError(f"managed model protocol has no SDK adapter: {descriptor.protocol}")


class _ManagedGatewayBackendBase:
    _base_url: str

    def __init__(
        self,
        descriptor: ModelDescriptor,
        protocol: _ManagedProtocolSpec,
    ) -> None:
        self._protocol = protocol
        self.descriptor = descriptor

    async def _request(
        self,
        operation: str,
        request: Awaitable[Any],
    ) -> dict[str, Any]:
        logger.debug(
            "agentcore.model.request.started mode=managed operation=%s "
            "connection_id=%s model=%s base_url=%s",
            operation,
            self.descriptor.connection_id,
            self.descriptor.model_name,
            _safe_url(self._base_url),
        )
        try:
            response = await request
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=managed operation=%s "
                "connection_id=%s error_type=%s status=%s base_url=%s",
                operation,
                self.descriptor.connection_id,
                type(exc).__name__,
                _exception_status(exc),
                _safe_url(self._base_url),
            )
            raise InvocationError("model request failed") from exc
        try:
            result = _response_dict(response)
        except InvocationError:
            logger.warning(
                "agentcore.model.request.failed mode=managed operation=%s "
                "connection_id=%s reason=invalid_response",
                operation,
                self.descriptor.connection_id,
            )
            raise
        logger.debug(
            "agentcore.model.request.succeeded mode=managed operation=%s connection_id=%s",
            operation,
            self.descriptor.connection_id,
        )
        return result

    def _unsupported_operation(self, operation: str) -> InvocationError:
        return InvocationError(
            f"managed model protocol {self._protocol.value} does not publish {operation}"
        )

    @staticmethod
    def _reject_parameters(parameters: Mapping[str, Any], reserved: set[str]) -> None:
        conflicts = reserved.intersection(parameters)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"reserved model request parameters: {names}")


class _OpenAIV1GatewayBackend(_ManagedGatewayBackendBase):
    _RESERVED_PARAMETERS = {
        "extra_headers",
        "model",
        "resource_name",
    }

    def __init__(
        self,
        config: AgentConfig,
        descriptor: ModelDescriptor,
        *,
        protocol: _ManagedProtocolSpec,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(descriptor, protocol)
        self._base_url = agentcore_model_base_url(
            config.spec.model.gateway_url,
            descriptor.connection_id,
            descriptor.protocol,
        )
        self._api_key, self._gateway_headers = _gateway_credentials(config.spec.gateway_headers)
        self._http = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            event_hooks={"request": [_log_model_http_request]},
        )
        self._openai_client = _openai_client(
            api_key=self._api_key,
            base_url=self._base_url,
            http_client=self._http,
            timeout=timeout,
        )
        logger.info(
            "agentcore.model.managed.created connection_name=%s connection_id=%s "
            "model=%s protocol=%s base_url=%s",
            descriptor.connection_name,
            descriptor.connection_id,
            descriptor.model_name,
            descriptor.protocol,
            _safe_url(self._base_url),
        )

    async def completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"messages", "stream"},
        )
        return await self._request(
            "completion",
            self._openai_client.chat.completions.create(
                messages=[dict(message) for message in messages],
                model=self.descriptor.model_name,
                stream=False,
                extra_headers=dict(self._gateway_headers) or None,
                **parameters,
            ),
        )

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"messages", "stream"},
        )
        logger.debug(
            "agentcore.model.request.started mode=managed operation=stream "
            "connection_id=%s model=%s",
            self.descriptor.connection_id,
            self.descriptor.model_name,
        )
        try:
            response = await self._openai_client.chat.completions.create(
                messages=[dict(message) for message in messages],
                model=self.descriptor.model_name,
                stream=True,
                extra_headers=dict(self._gateway_headers) or None,
                **parameters,
            )
            async for chunk in response:
                yield _response_dict(chunk)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=managed operation=stream "
                "connection_id=%s error_type=%s status=%s base_url=%s",
                self.descriptor.connection_id,
                type(exc).__name__,
                _exception_status(exc),
                _safe_url(self._base_url),
            )
            raise InvocationError("model stream request failed") from exc
        logger.debug(
            "agentcore.model.request.succeeded mode=managed operation=stream connection_id=%s",
            self.descriptor.connection_id,
        )

    async def responses(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"input", "stream"},
        )
        return await self._request(
            "responses",
            self._openai_client.responses.create(
                input=input,
                model=self.descriptor.model_name,
                stream=False,
                extra_headers=dict(self._gateway_headers) or None,
                **parameters,
            ),
        )

    async def responses_stream(
        self,
        input: Any,
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"input", "stream"},
        )
        logger.debug(
            "agentcore.model.request.started mode=managed operation=responses_stream "
            "connection_id=%s model=%s",
            self.descriptor.connection_id,
            self.descriptor.model_name,
        )
        try:
            response = await self._openai_client.responses.create(
                input=input,
                model=self.descriptor.model_name,
                stream=True,
                extra_headers=dict(self._gateway_headers) or None,
                **parameters,
            )
            async for event in response:
                yield _response_dict(event)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=managed operation=responses_stream "
                "connection_id=%s error_type=%s status=%s base_url=%s",
                self.descriptor.connection_id,
                type(exc).__name__,
                _exception_status(exc),
                _safe_url(self._base_url),
            )
            raise InvocationError("model responses stream request failed") from exc
        logger.debug(
            "agentcore.model.request.succeeded mode=managed operation=responses_stream "
            "connection_id=%s",
            self.descriptor.connection_id,
        )

    async def embedding(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        raise self._unsupported_operation("Embedding API")

    async def aclose(self) -> None:
        await self._http.aclose()
        logger.debug(
            "agentcore.model.managed.closed connection_id=%s",
            self.descriptor.connection_id,
        )

    def __repr__(self) -> str:
        return (
            "OpenAIV1GatewayBackend("
            f"model={self.descriptor.model_name!r}, protocol={self._protocol.value!r}, "
            f"base_url={_safe_url(self._base_url)!r}, credentials=<redacted>)"
        )


class _AnthropicGatewayBackend(_ManagedGatewayBackendBase):
    _RESERVED_PARAMETERS = {
        "extra_headers",
        "messages",
        "model",
        "resource_name",
        "stream",
    }

    def __init__(
        self,
        config: AgentConfig,
        descriptor: ModelDescriptor,
        *,
        protocol: _ManagedProtocolSpec,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(descriptor, protocol)
        self._base_url = agentcore_model_base_url(
            config.spec.model.gateway_url,
            descriptor.connection_id,
            descriptor.protocol,
        )
        auth_token, gateway_headers = _gateway_credentials(config.spec.gateway_headers)
        self._anthropic_client = anthropic_gateway_client(
            auth_token=auth_token,
            base_url=self._base_url,
            default_headers=gateway_headers,
            timeout=timeout,
        )
        logger.info(
            "agentcore.model.managed.created connection_name=%s connection_id=%s "
            "model=%s protocol=%s base_url=%s",
            descriptor.connection_name,
            descriptor.connection_id,
            descriptor.model_name,
            descriptor.protocol,
            _safe_url(self._base_url),
        )

    async def completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        request_parameters = self._request_parameters(messages, parameters)
        return await self._request(
            "completion",
            self._anthropic_client.messages.create(
                messages=request_parameters.pop("messages"),
                model=self.descriptor.model_name,
                stream=False,
                **request_parameters,
            ),
        )

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        request_parameters = self._request_parameters(messages, parameters)
        logger.debug(
            "agentcore.model.request.started mode=managed operation=stream "
            "connection_id=%s model=%s",
            self.descriptor.connection_id,
            self.descriptor.model_name,
        )
        try:
            response = await self._anthropic_client.messages.create(
                messages=request_parameters.pop("messages"),
                model=self.descriptor.model_name,
                stream=True,
                **request_parameters,
            )
            async for event in response:
                yield _response_dict(event)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=managed operation=stream "
                "connection_id=%s error_type=%s status=%s base_url=%s",
                self.descriptor.connection_id,
                type(exc).__name__,
                _exception_status(exc),
                _safe_url(self._base_url),
            )
            raise InvocationError("model stream request failed") from exc
        logger.debug(
            "agentcore.model.request.succeeded mode=managed operation=stream connection_id=%s",
            self.descriptor.connection_id,
        )

    async def responses(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        raise self._unsupported_operation("Responses API")

    async def responses_stream(
        self,
        input: Any,
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise self._unsupported_operation("Responses API")

    async def embedding(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        raise self._unsupported_operation("Embedding API")

    async def aclose(self) -> None:
        await self._anthropic_client.close()
        logger.debug(
            "agentcore.model.managed.closed connection_id=%s",
            self.descriptor.connection_id,
        )

    def _request_parameters(
        self,
        messages: Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._reject_parameters(parameters, self._RESERVED_PARAMETERS)
        request_parameters = dict(parameters)
        max_tokens = request_parameters.pop("max_tokens", self.descriptor.max_tokens)
        if max_tokens is None:
            raise ConfigError("managed Anthropic model requires max_tokens")
        request_messages, system = _anthropic_messages(messages)
        if system is not None:
            if "system" in request_parameters:
                raise ValueError("system is already set by a system message")
            request_parameters["system"] = system
        request_parameters["max_tokens"] = max_tokens
        request_parameters["messages"] = request_messages
        return request_parameters

    def __repr__(self) -> str:
        return (
            "AnthropicGatewayBackend("
            f"model={self.descriptor.model_name!r}, protocol={self._protocol.value!r}, "
            f"base_url={_safe_url(self._base_url)!r}, credentials=<redacted>)"
        )


class _LiteLLMDirectBackend:
    _RESERVED_PARAMETERS = {
        "api_base",
        "api_key",
        "base_url",
        "caching",
        "client",
        "custom_llm_provider",
        "extra_headers",
        "max_retries",
        "model",
        "num_retries",
        "resource_name",
        "shared_session",
    }

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        base_url: str,
        api_key: str | None,
        api_key_provider: APIKeyProvider | None,
        headers_provider: HeadersProvider | None,
    ) -> None:
        if not model:
            raise ConfigError("Direct model name must be non-empty")
        if not provider:
            raise ConfigError("Direct model provider must be non-empty")
        try:
            parsed = urlsplit(base_url)
        except ValueError as exc:
            raise ConfigError("Direct model base_url must be a valid HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError("Direct model base_url must be an absolute HTTP(S) URL")
        if api_key is not None and api_key_provider is not None:
            raise ConfigError("Direct model api_key and api_key_provider are mutually exclusive")
        self._model = model
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self._headers_provider = headers_provider
        logger.info(
            "agentcore.model.direct.created provider=%s model=%s host=%s base_url=%s",
            provider,
            model,
            urlsplit(self._base_url).hostname or "",
            _safe_url(self._base_url),
        )

    async def completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        self._reject_parameters(parameters, self._RESERVED_PARAMETERS | {"messages"})
        from litellm import acompletion

        kwargs = await self._kwargs(parameters=parameters)
        logger.debug(
            "agentcore.model.request.started mode=direct operation=completion provider=%s model=%s",
            self._provider,
            self._model,
        )
        try:
            response = await acompletion(messages=list(messages), **kwargs)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=completion "
                "provider=%s model=%s error_type=%s base_url=%s",
                self._provider,
                self._model,
                type(exc).__name__,
                _safe_url(self._base_url),
            )
            raise InvocationError("direct model completion failed") from exc
        try:
            result = _response_dict(response)
        except InvocationError:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=completion "
                "provider=%s model=%s reason=invalid_response",
                self._provider,
                self._model,
            )
            raise
        logger.debug(
            "agentcore.model.request.succeeded mode=direct operation=completion "
            "provider=%s model=%s",
            self._provider,
            self._model,
        )
        return result

    async def responses_stream(
        self,
        input: Any,
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"input", "stream"},
        )
        from litellm import aresponses

        kwargs = await self._kwargs(
            parameters={**parameters, "stream": True},
            api_base=True,
        )
        logger.debug(
            "agentcore.model.request.started mode=direct operation=responses_stream "
            "provider=%s model=%s",
            self._provider,
            self._model,
        )
        try:
            response = await aresponses(input=input, **kwargs)
            async for event in response:
                yield _response_dict(event)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=responses_stream "
                "provider=%s model=%s error_type=%s base_url=%s",
                self._provider,
                self._model,
                type(exc).__name__,
                _safe_url(self._base_url),
            )
            raise InvocationError("direct model responses stream failed") from exc
        logger.debug(
            "agentcore.model.request.succeeded mode=direct operation=responses_stream "
            "provider=%s model=%s",
            self._provider,
            self._model,
        )

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self._reject_parameters(
            parameters,
            self._RESERVED_PARAMETERS | {"messages", "stream"},
        )
        from litellm import acompletion

        kwargs = await self._kwargs(parameters={**parameters, "stream": True})
        logger.debug(
            "agentcore.model.request.started mode=direct operation=stream provider=%s model=%s",
            self._provider,
            self._model,
        )
        try:
            response = await acompletion(messages=list(messages), **kwargs)
            async for chunk in response:
                yield _response_dict(chunk)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=stream "
                "provider=%s model=%s error_type=%s base_url=%s",
                self._provider,
                self._model,
                type(exc).__name__,
                _safe_url(self._base_url),
            )
            raise InvocationError("direct model stream failed") from exc
        logger.debug(
            "agentcore.model.request.succeeded mode=direct operation=stream provider=%s model=%s",
            self._provider,
            self._model,
        )

    async def responses(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        self._reject_parameters(parameters, self._RESERVED_PARAMETERS | {"input"})
        from litellm import aresponses

        kwargs = await self._kwargs(parameters=parameters, api_base=True)
        logger.debug(
            "agentcore.model.request.started mode=direct operation=responses provider=%s model=%s",
            self._provider,
            self._model,
        )
        try:
            response = await aresponses(input=input, **kwargs)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=responses "
                "provider=%s model=%s error_type=%s base_url=%s",
                self._provider,
                self._model,
                type(exc).__name__,
                _safe_url(self._base_url),
            )
            raise InvocationError("direct model responses call failed") from exc
        try:
            result = _response_dict(response)
        except InvocationError:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=responses "
                "provider=%s model=%s reason=invalid_response",
                self._provider,
                self._model,
            )
            raise
        logger.debug(
            "agentcore.model.request.succeeded mode=direct operation=responses "
            "provider=%s model=%s",
            self._provider,
            self._model,
        )
        return result

    async def embedding(
        self,
        input: Any,
        **parameters: Any,
    ) -> dict[str, Any]:
        self._reject_parameters(parameters, self._RESERVED_PARAMETERS | {"input"})
        from litellm import aembedding

        kwargs = await self._kwargs(parameters=parameters, api_base=True)
        logger.debug(
            "agentcore.model.request.started mode=direct operation=embedding provider=%s model=%s",
            self._provider,
            self._model,
        )
        try:
            response = await aembedding(input=input, **kwargs)
        except Exception as exc:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=embedding "
                "provider=%s model=%s error_type=%s base_url=%s",
                self._provider,
                self._model,
                type(exc).__name__,
                _safe_url(self._base_url),
            )
            raise InvocationError("direct model embedding failed") from exc
        try:
            result = _response_dict(response)
        except InvocationError:
            logger.warning(
                "agentcore.model.request.failed mode=direct operation=embedding "
                "provider=%s model=%s reason=invalid_response",
                self._provider,
                self._model,
            )
            raise
        logger.debug(
            "agentcore.model.request.succeeded mode=direct operation=embedding "
            "provider=%s model=%s",
            self._provider,
            self._model,
        )
        return result

    async def aclose(self) -> None:
        return None

    async def _kwargs(
        self,
        *,
        parameters: Mapping[str, Any],
        api_base: bool = False,
    ) -> dict[str, Any]:
        api_key = await self._resolve_api_key()
        headers = await self._resolve_headers()
        kwargs = {
            "api_key": api_key,
            "api_base" if api_base else "base_url": self._base_url,
            "model": self._model,
            "custom_llm_provider": self._provider,
            "caching": False,
            "max_retries": 0,
            **parameters,
        }
        if headers:
            kwargs["extra_headers"] = headers
        return kwargs

    async def _resolve_api_key(self) -> str:
        value: str | Awaitable[str] | None = self._api_key
        if self._api_key_provider is not None:
            value = self._api_key_provider()
        if inspect.isawaitable(value):
            value = await value
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ConfigError("Direct model api_key_provider must return a string")
        return value

    async def _resolve_headers(self) -> dict[str, str]:
        if self._headers_provider is None:
            return {}
        value = self._headers_provider()
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, Mapping):
            raise ConfigError("Direct model headers_provider must return a mapping")
        headers = dict(value)
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in headers.items()):
            raise ConfigError("Direct model headers must contain string keys and values")
        return headers

    @staticmethod
    def _reject_parameters(parameters: Mapping[str, Any], reserved: set[str]) -> None:
        conflicts = reserved.intersection(parameters)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"reserved direct model request parameters: {names}")

    def __repr__(self) -> str:
        return (
            "LiteLLMDirectBackend("
            f"model={self._model!r}, provider={self._provider!r}, "
            f"base_url={_safe_url(self._base_url)!r}, credentials=<redacted>)"
        )


def _gateway_credentials(headers: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    authorization = next(
        (value for key, value in headers.items() if key.lower() == "authorization"),
        None,
    )
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise ConfigError("managed model requires a Bearer gateway credential")
    api_key = authorization[7:].strip()
    if not api_key:
        raise ConfigError("managed model requires a Bearer gateway credential")
    extra_headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
    return api_key, extra_headers


def _openai_client(
    *,
    api_key: str,
    base_url: str,
    http_client: httpx.AsyncClient,
    timeout: float,
) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        timeout=timeout,
        max_retries=0,
    )


def anthropic_gateway_client(
    *,
    auth_token: str,
    base_url: str,
    default_headers: Mapping[str, str],
    timeout: float,
) -> Any:
    from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

    return AsyncAnthropic(
        auth_token=auth_token,
        base_url=base_url,
        default_headers=dict(default_headers) or None,
        http_client=DefaultAsyncHttpxClient(
            timeout=timeout,
            follow_redirects=False,
            event_hooks={"request": [_log_model_http_request]},
        ),
        max_retries=0,
    )


def _anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Any | None]:
    request_messages: list[dict[str, Any]] = []
    system_values: list[Any] = []
    for message in messages:
        value = dict(message)
        if value.get("role") == "system":
            system_values.append(value.get("content"))
        else:
            request_messages.append(value)
    if not system_values:
        return request_messages, None
    if len(system_values) == 1:
        return request_messages, system_values[0]
    if all(isinstance(value, str) for value in system_values):
        return request_messages, "\n\n".join(system_values)
    raise ValueError("multiple Anthropic system messages must contain text")


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        if isinstance(value, dict):
            return value
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    raise InvocationError("model endpoint returned an invalid response")


async def _log_model_http_request(request: Any) -> None:
    # Both OpenAI (httpx) and Anthropic (httpx2) support this request hook.
    logger.info(
        "agentcore.model.http.request method=%s url=%s",
        request.method,
        _safe_url(str(request.url)),
    )


def _exception_status(exc: Exception) -> int | str:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else "-"


def agentcore_model_base_url(
    gateway_url: str,
    connection_id: str,
    protocol: str,
) -> str:
    spec = _managed_model_protocol(protocol)
    gateway = gateway_url.rstrip("/")
    if gateway.endswith("/v1"):
        gateway = gateway[:-3]
    return (
        f"{gateway}/model-connection/{quote(connection_id, safe='')}"
        f"{spec.runtime_base_path}"
    )


def require_managed_model_adapter(protocol: str) -> None:
    _managed_model_protocol(protocol)


def managed_framework_provider(protocol: str) -> str:
    return _managed_model_protocol(protocol).framework_provider


def _managed_model_protocol(protocol: str) -> _ManagedProtocolSpec:
    normalized = protocol.strip().lower()
    try:
        return _MANAGED_PROTOCOL_SPECS[normalized]
    except KeyError as exc:
        raise ConfigError(f"managed model protocol has no SDK adapter: {protocol}") from exc
