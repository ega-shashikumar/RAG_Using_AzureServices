"""
app/middleware/auth.py — API key authentication via X-API-Key header.
Auth is disabled when API_KEY setting is empty (local dev).
"""
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    settings = get_settings()

    # Auth disabled — useful for local development
    if not settings.api_key:
        return

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header",
        )