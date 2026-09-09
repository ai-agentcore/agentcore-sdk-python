"""Shared URL rendering for diagnostic logs."""

from urllib.parse import urlsplit, urlunsplit


def safe_url(value: str) -> str:
    """Keep scheme, host, port and path; omit userinfo, query and fragment."""
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
