"""Validate fixed MCP headers without replacing platform or transport headers."""

import re
from collections.abc import Mapping

from agentcore.errors import ConfigError

_TRANSPORT_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "upgrade",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
    "content-type",
    "accept",
}


def merge_mcp_headers(
    base: Mapping[str, str],
    custom: Mapping[str, str] | None,
) -> dict[str, str]:
    result = dict(base)
    if custom is None:
        return result
    if not isinstance(custom, Mapping):
        raise ConfigError("MCP headers must be a mapping")
    protected = {name.lower() for name in base} | _TRANSPORT_HEADERS | {"authorization"}
    seen: set[str] = set()
    for name, value in custom.items():
        if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            raise ConfigError("Invalid MCP header name")
        if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ConfigError("MCP header values must be strings without control characters")
        folded = name.lower()
        if folded in protected:
            raise ConfigError(f"MCP header is protected: {folded}")
        if folded in seen:
            raise ConfigError(f"Duplicate MCP header: {folded}")
        seen.add(folded)
        result[folded] = value
    return result
