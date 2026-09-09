"""Explicit Alibaba Cloud AccessKey credentials for local SDK use."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class AccessKeyCredential:
    """An AK/SK pair, optionally with an STS security token."""

    access_key_id: str
    access_key_secret: str
    security_token: str | None = None

    def __post_init__(self) -> None:
        if not self.access_key_id.strip() or not self.access_key_secret.strip():
            raise ValueError("AccessKey ID and AccessKey secret must not be empty")
        if self.security_token is not None and not self.security_token.strip():
            raise ValueError("Security token must not be empty")

    def __repr__(self) -> str:
        credential_type = "sts" if self.security_token is not None else "access_key"
        return f"AccessKeyCredential(type={credential_type!r}, <redacted>)"
