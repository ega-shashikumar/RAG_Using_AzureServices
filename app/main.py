"""
app/main.py — FastAPI application factory.
Wires together all routers, middleware, lifespan, and observability.
"""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import chat, health, history, ingest
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.middleware.rate_limit import limiter
from app.services.search_service import SearchService


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at startup and shutdown."""
    setup_logging()
    settings = get_settings()
    logger.info(
        "Azure RAG API starting",
        environment=settings.environment,
    )

    # Auto-create search index on startup
    try:
        svc = SearchService()
        await svc.ensure_index()
        logger.info("search index verified")
    except Exception as exc:
        logger.error(
            "failed to verify search index",
            error=str(exc),
        )

    yield

    logger.info("Azure RAG API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Azure RAG API",
        description=(
            "Production RAG pipeline using Azure OpenAI, "
            "Azure AI Search, Blob Storage, and Cosmos DB."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Rate limiting ─────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ── Request tracing middleware ────────────────────────────────────
    @app.middleware("http")
    async def request_trace(request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        logger.info(
            "http request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=elapsed_ms,
        )
        response.headers["X-Trace-Id"] = trace_id
        return response

    # ── Global error handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled exception",
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred."
            },
        )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(chat.router)
    app.include_router(history.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

#2026-03-01