"""Bearer-token auth for /v1/* routes. Never reuses CLAUDE_CODE_OAUTH_TOKEN."""
from __future__ import annotations

import hmac

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.errors import AuthError


async def require_api_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.proxy_api_key:
        # Misconfiguration: refuse rather than silently allow unauthenticated access.
        raise AuthError("PROXY_API_KEY is not configured on the server")

    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError("Missing Authorization: Bearer <PROXY_API_KEY> header")

    provided = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(provided, settings.proxy_api_key):
        raise AuthError("Invalid API key")
