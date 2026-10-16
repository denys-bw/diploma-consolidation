from __future__ import annotations

import re

EMAIL = re.compile(r"^[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}$")
URL = re.compile(
    r"^(?:(?:https?|ftp)://)?(?:www\d?\.)?[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}(?::\d+)?(?:[/?#]\S*)?$",
    re.IGNORECASE,
)
SCHEME = re.compile(r"^(?:https?|ftp)://", re.IGNORECASE)
WWW = re.compile(r"^www\d?\.", re.IGNORECASE)
HOST_LABEL = re.compile(r"^[^\W_][\w-]+$")


def looks_like_email(text: str) -> bool:
    return EMAIL.match(_strip_mailto(text.strip())) is not None


def looks_like_url(text: str) -> bool:
    text = text.strip()
    if " " in text or "@" in text or URL.match(text) is None:
        return False
    if SCHEME.match(text) or WWW.match(text) or "/" in text:
        return True
    return _bare_host(text)


def normalize_email(text: str) -> str | None:
    candidate = _strip_mailto(text.strip()).lower()
    return candidate if EMAIL.match(candidate) else None


def normalize_url(text: str) -> str | None:
    candidate = text.strip()
    if not URL.match(candidate):
        return None
    candidate = WWW.sub("", SCHEME.sub("", candidate))
    host, _, rest = candidate.partition("/")
    rest = rest.rstrip("/")
    return host.lower() + (f"/{rest}" if rest else "")


def _strip_mailto(text: str) -> str:
    return text[7:] if text.lower().startswith("mailto:") else text


def _bare_host(text: str) -> bool:
    host = text.split(":", 1)[0]
    labels = host.split(".")
    return host == host.lower() and len(labels) >= 2 and all(HOST_LABEL.match(label) for label in labels)
