"""Collaboration diagnostics kept local so the package can upgrade independently."""

import logging
import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx


def safe_error_message(value: object) -> str | None:
    """Bound a service's message field; never pass an exception or response dump.

    Omit any suffix that contains credential or echoed request fields rather
    than attempting to parse arbitrary nested payloads in diagnostic text.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    text = re.sub(
        r"(?i)\b(?:authorization|api[ _-]?key(?:\s+provided)?|"
        r"access[_-]?key(?:[_-]?(?:id|secret))?|"
        r"(?:security|access|refresh|jwt)[_-]?token|token|password|secret|"
        r"text|content|query|messages|metadata|headers)[\"']?\s*[:=].*",
        "<redacted>", text,
    )
    text = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s,;]+", "<redacted>", text)
    text = re.sub(
        r"\b(?:LTAI[A-Za-z0-9]+|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b",
        "<redacted>", text,
    )
    text = re.sub(r"https?://\S+", "<url>", text)
    return text[:512]


def safe_url(value: str) -> str:
    """Keep scheme, host, port and path; omit userinfo, query and fragment."""
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))


logger = logging.getLogger("agentcore.collaboration.task_service")


def log_response_failure(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
    except (ValueError, RuntimeError):
        body = {}
    body = body if isinstance(body, Mapping) else {}
    request_id = (body.get("requestId") or body.get("RequestId")
                  or response.headers.get("x-request-id")
                  or response.headers.get("x-acs-request-id")
                  or response.headers.get("x-oss-request-id"))
    try:
        url = safe_url(str(response.request.url))
        method = response.request.method
    except RuntimeError:
        url, method = "-", "-"
    logger.warning(
        "agentcore.collaboration.task_service.request.failed method=%s url=%s "
        "status=%s code=%s request_id=%s message=%s",
        method, url, response.status_code,
        safe_error_message(body.get("code") or body.get("Code")) or "-",
        safe_error_message(request_id) or "-",
        safe_error_message(body.get("message") or body.get("Message")) or "-",
    )
