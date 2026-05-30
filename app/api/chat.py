"""
app/api/chat.py — RAG chat endpoint.
POST /chat
  - stream=true  → Server-Sent Events (text/event-stream)
  - stream=false → JSON ChatResponse
"""
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.azure_clients import get_openai_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SSEEvent,
    SSEEventType,
    SourceChunk,
)
from app.services.generation_service import GenerationService
from app.services.history_service import HistoryService
from app.services.search_service import SearchService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def _to_source_chunks(results: list[dict]) -> list[SourceChunk]:
    return [
        SourceChunk(
            document_id=r["document_id"],
            filename=r["filename"],
            chunk_index=r["chunk_index"],
            score=round(r["score"], 4),
            text_preview=r["content"][:200],
        )
        for r in results
    ]


async def _get_query_embedding(query: str) -> list[float]:
    """Embed the user query for vector search."""
    settings = get_settings()
    client = get_openai_client()
    resp = await client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=[query],
    )
    return resp.data[0].embedding


async def _sse_generator(
    body: ChatRequest,
    retrieved: list[dict],
    source_chunks: list[SourceChunk],
) -> AsyncIterator[str]:
    """Yield SSE-formatted events: delta → sources → done."""
    gen_service = GenerationService()
    history_service = HistoryService()

    history = await history_service.load(body.session_id)
    full_answer: list[str] = []

    # Stream text deltas
    try:
        async for delta in gen_service.generate_streaming(
            query=body.query,
            retrieved_chunks=retrieved,
            history=history,
        ):
            full_answer.append(delta)
            event = SSEEvent(type=SSEEventType.DELTA, data=delta)
            yield f"data: {event.model_dump_json()}\n\n"

    except Exception as exc:
        error_event = SSEEvent(
            type=SSEEventType.ERROR,
            data=str(exc),
        )
        yield f"data: {error_event.model_dump_json()}\n\n"
        return

    # Send sources
    sources_event = SSEEvent(
        type=SSEEventType.SOURCES,
        data=[s.model_dump() for s in source_chunks],
    )
    yield f"data: {sources_event.model_dump_json()}\n\n"

    # Send done
    done_event = SSEEvent(type=SSEEventType.DONE, data=None)
    yield f"data: {done_event.model_dump_json()}\n\n"

    # Persist to history — non-fatal
    await history_service.append(
        session_id=body.session_id,
        user_message=body.query,
        assistant_message="".join(full_answer),
    )


@router.post(
    "",
    dependencies=[Depends(verify_api_key)],
    summary="Chat with your documents using RAG",
)
@limiter.limit("60/minute")
async def chat(request: Request, body: ChatRequest):
    t0 = time.monotonic()

    logger.info(
        "chat request",
        session_id=body.session_id,
        query_preview=body.query[:80],
        stream=body.stream,
    )

    # 1. Embed query
    try:
        query_vector = await _get_query_embedding(body.query)
    except Exception as exc:
        logger.error("embedding failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed query. Check Azure OpenAI configuration.",
        )

    # 2. Retrieve relevant chunks
    search_service = SearchService()
    try:
        retrieved = await search_service.search(
            query=body.query,
            query_vector=query_vector,
            top_k=body.top_k,
            filters=body.filters,
        )
    except Exception as exc:
        logger.error("search failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Search service error.",
        )

    if not retrieved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found for your query.",
        )

    source_chunks = _to_source_chunks(retrieved)

    # 3a. Streaming response
    if body.stream:
        return StreamingResponse(
            _sse_generator(body, retrieved, source_chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # 3b. Non-streaming response
    gen_service = GenerationService()
    history_service = HistoryService()
    history = await history_service.load(body.session_id)

    try:
        answer, prompt_tokens, completion_tokens = await gen_service.generate(
            query=body.query,
            retrieved_chunks=retrieved,
            history=history,
        )
    except Exception as exc:
        logger.error("generation failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Generation service error.",
        )

    # Save history
    await history_service.append(
        session_id=body.session_id,
        user_message=body.query,
        assistant_message=answer,
    )

    return ChatResponse(
        session_id=body.session_id,
        answer=answer,
        sources=source_chunks,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )