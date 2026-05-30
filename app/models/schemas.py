"""
app/models/schemas.py — All Pydantic v2 request/response schemas.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestMetadata(BaseModel):
    source: str = "upload"
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    blob_url: str
    chunks_indexed: int
    message: str = "Document ingested successfully"


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: dict[str, str] | None = None
    stream: bool = True

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


# ── SSE Events ────────────────────────────────────────────────────────────────

class SSEEventType(str, Enum):
    DELTA = "delta"
    SOURCES = "sources"
    DONE = "done"
    ERROR = "error"


class SSEEvent(BaseModel):
    type: SSEEventType
    data: Any


# ── History ───────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class HistoryMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    created_at: datetime
    updated_at: datetime


# ── Health ────────────────────────────────────────────────────────────────────

class ServiceStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthResponse(BaseModel):
    status: ServiceStatus
    services: dict[str, ServiceStatus]
    version: str = "1.0.0"