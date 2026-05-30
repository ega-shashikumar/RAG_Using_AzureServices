"""
app/middleware/rate_limit.py — Per-IP rate limiting using slowapi.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings


def _get_limit() -> str:
    rpm = get_settings().rate_limit_per_minute
    return f"{rpm}/minute"


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_get_limit()],
)