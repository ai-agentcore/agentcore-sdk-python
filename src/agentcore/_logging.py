"""Safe rendering of URLs and service messages for diagnostic logs."""

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


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


def _diagnostics(body: Any, headers: Any, status: Any, error: Any = None) -> str:
    body = body if isinstance(body, Mapping) else {}
    detail = body.get("error")
    detail = detail if isinstance(detail, Mapping) else body
    headers = {str(k).lower(): v for k, v in headers.items()} if headers else {}
    request_id = (getattr(error, "request_id", None) or body.get("RequestId")
                  or body.get("requestId") or body.get("requestid")
                  or headers.get("x-acs-request-id") or headers.get("x-request-id")
                  or headers.get("request-id") or headers.get("x-oss-request-id"))
    code = (detail.get("code") or detail.get("Code") or detail.get("type")
            or getattr(error, "code", None))
    message = (detail.get("message") or detail.get("Message")
               or getattr(error, "message", None))
    status = status or body.get("statusCode") or body.get("StatusCode")
    return (f"status={status or '-'} code={safe_error_message(code) or '-'} "
            f"request_id={safe_error_message(request_id) or '-'} "
            f"message={safe_error_message(message) or '-'}")


def response_diagnostics(response: Any) -> str:
    """Read only diagnostic fields from an unsuccessful HTTP response."""
    try:
        body = response.json()
    except (ValueError, RuntimeError):
        body = None
    return _diagnostics(body, response.headers, response.status_code)


def exception_diagnostics(error: Exception) -> str:
    """Read SDK error fields without rendering its arbitrary body or exception chain."""
    response = getattr(error, "response", None)
    body = getattr(error, "body", None) or getattr(error, "data", None)
    if not body and response is not None:
        try:
            body = response.json()
        except (ValueError, RuntimeError):
            pass
    return _diagnostics(body, getattr(response, "headers", None),
                        getattr(error, "status_code", None)
                        or getattr(error, "statusCode", None)
                        or getattr(response, "status_code", None), error)


def log_response_failure(logger: Any, event: str, response: Any) -> None:
    if response.status_code < 400:
        return
    try:
        url = safe_url(str(response.request.url))
    except RuntimeError:
        url = "-"
    logger.warning("%s url=%s %s", event, url, response_diagnostics(response))
