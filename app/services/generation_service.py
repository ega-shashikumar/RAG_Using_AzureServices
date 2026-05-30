"""
app/services/generation_service.py — Streaming and non-streaming RAG
answer generation using Azure OpenAI.
Injects retrieved context + conversation history into the prompt.
"""
from __future__ import annotations

import time
from typing import AsyncIterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.azure_clients import get_openai_client
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['filename']} "
            f"(chunk {chunk['chunk_index']})\n"
            f"{chunk['content'].strip()}"
        )
    return "=== Context ===\n\n" + "\n\n".join(parts) + "\n\n=== End ==="


def _build_messages(
    system_prompt: str,
    context: str,
    history: list[dict],
    query: str,
) -> list[dict]:
    """Build the full message list for the chat completion API."""
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n{context}",
        }
    ]

    # Include last 10 history messages to bound context window
    for msg in history[-10:]:
        messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    messages.append({"role": "user", "content": query})
    return messages


class GenerationService:

    def __init__(self) -> None:
        self._settings = get_settings()

    async def generate_streaming(
        self,
        query: str,
        retrieved_chunks: list[dict],
        history: list[dict],
    ) -> AsyncIterator[str]:
        """
        Stream answer tokens as they arrive from Azure OpenAI.
        Yields text deltas one at a time.
        """
        context = _build_context(retrieved_chunks)
        messages = _build_messages(
            self._settings.system_prompt,
            context,
            history,
            query,
        )

        client = get_openai_client()
        logger.info(
            "generation started (streaming)",
            model=self._settings.azure_openai_chat_deployment,
            context_chunks=len(retrieved_chunks),
        )

        stream = await client.chat.completions.create(
            model=self._settings.azure_openai_chat_deployment,
            messages=messages,
            max_tokens=self._settings.max_tokens_response,
            temperature=self._settings.temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        history: list[dict],
    ) -> tuple[str, int, int]:
        """
        Non-streaming generation.
        Returns (answer, prompt_tokens, completion_tokens).
        """
        context = _build_context(retrieved_chunks)
        messages = _build_messages(
            self._settings.system_prompt,
            context,
            history,
            query,
        )

        client = get_openai_client()
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=self._settings.azure_openai_chat_deployment,
            messages=messages,
            max_tokens=self._settings.max_tokens_response,
            temperature=self._settings.temperature,
            stream=False,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        usage = response.usage
        answer = response.choices[0].message.content or ""

        logger.info(
            "generation complete",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=round(elapsed_ms),
        )

        return answer, usage.prompt_tokens, usage.completion_tokens