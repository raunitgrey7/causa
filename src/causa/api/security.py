"""API key authentication, active only when ``APP_API_KEY`` is set.

Health, metrics, the OpenAPI docs and the static dashboard stay open so probes, scrapers and a first look
work without secrets; every route that computes or stores something requires the ``X-API-Key`` header.
Keys are compared in constant time.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

OPEN_PATHS: tuple[str, ...] = ("/health", "/metrics", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico")
OPEN_PREFIXES: tuple[str, ...] = ("/static/",)


def is_open(path: str) -> bool:
    """Whether ``path`` is served without a key."""
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: reject requests without the configured key."""
    expected = request.app.state.settings.api_key
    if not expected or is_open(request.url.path):
        return
    provided = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
