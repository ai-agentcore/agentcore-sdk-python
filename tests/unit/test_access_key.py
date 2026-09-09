from __future__ import annotations

import pytest

from agentcore.auth import AccessKeyCredential


def test_access_key_credential_supports_ak_and_sts_without_secret_repr() -> None:
    access_key = AccessKeyCredential("ak", "sk")
    sts = AccessKeyCredential("sts-ak", "sts-sk", "sts-token")

    assert access_key.security_token is None
    assert sts.security_token == "sts-token"
    assert "ak" not in repr(access_key)
    assert "sk" not in repr(access_key)
    assert "sts-token" not in repr(sts)


@pytest.mark.parametrize(
    ("access_key_id", "access_key_secret", "security_token"),
    [
        ("", "sk", None),
        ("ak", "", None),
        ("ak", "sk", ""),
    ],
)
def test_access_key_credential_rejects_empty_values(
    access_key_id: str,
    access_key_secret: str,
    security_token: str | None,
) -> None:
    with pytest.raises(ValueError):
        AccessKeyCredential(access_key_id, access_key_secret, security_token)
