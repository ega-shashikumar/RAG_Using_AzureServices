"""
app/services/ingestion_service.py — Full document ingestion pipeline.
Steps:
  1. Upload raw file to Azure Blob Storage
  2. Parse and chunk text
  3. Generate embeddings via Azure OpenAI
  4. Index chunks in Azure AI Search
"""
from __future__ import annotations

import json

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.azure_clients import get_openai_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import IngestMetadata, IngestResponse
from app.services.blob_service import BlobService
from app.services.search_service import SearchService
from app.utils.chunker import TokenAwareChunker, parse_document

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 16


class IngestionService:

    def __init__(self) -> None:
        self._settings = get_settings()
        self._blob = BlobService()
        self._search = SearchService()
        self._chunker = TokenAwareChunker()

    # ── Embeddings ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        client = get_openai_client()
        resp = await client.embeddings.create(
            model=self._settings.azure_openai_embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def _embed_all(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed all texts in batches."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i: i + EMBED_BATCH_SIZE]
            vecs = await self._embed_batch(batch)
            embeddings.extend(vecs)
            logger.info(
                "embedded batch",
                batch=i // EMBED_BATCH_SIZE + 1,
                size=len(batch),
            )
        return embeddings

    # ── Search document builder ───────────────────────────────────────────────

    def _build_search_docs(
        self,
        document_id: str,
        filename: str,
        chunks,
        embeddings: list[list[float]],
        metadata: IngestMetadata,
    ) -> list[dict]:
        docs = []
        for chunk, vector in zip(chunks, embeddings):
            docs.append(
                {
                    "id": f"{document_id}-{chunk.chunk_index}",
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.text,
                    "content_vector": vector,
                    "source": metadata.source,
                    "tags": json.dumps(metadata.tags),
                }
            )
        return docs

    # ── Public entry point ────────────────────────────────────────────────────

    async def ingest(
        self,
        filename: str,
        data: bytes,
        metadata: IngestMetadata,
    ) -> IngestResponse:
        # 1. Compute deterministic document ID
        document_id = BlobService.compute_document_id(data, filename)
        logger.info(
            "ingestion started",
            document_id=document_id,
            filename=filename,
            size_bytes=len(data),
        )

        # 2. Upload to Blob Storage
        await self._blob.ensure_container()
        blob_url = await self._blob.upload_document(
            filename, data, document_id
        )

        # 3. Parse text from document
        text = parse_document(filename, data)
        if not text.strip():
            raise ValueError(
                "No text could be extracted from this document."
            )

        # 4. Split into chunks
        chunks = self._chunker.split(
            text,
            metadata=metadata.model_dump(),
        )
        if not chunks:
            raise ValueError("Document produced no chunks after splitting.")

        # 5. Generate embeddings
        embeddings = await self._embed_all([c.text for c in chunks])

        # 6. Build and index search documents
        search_docs = self._build_search_docs(
            document_id, filename, chunks, embeddings, metadata
        )
        await self._search.index_documents(search_docs)

        logger.info(
            "ingestion complete",
            document_id=document_id,
            chunks_indexed=len(chunks),
        )

        return IngestResponse(
            document_id=document_id,
            filename=filename,
            blob_url=blob_url,
            chunks_indexed=len(chunks),
        )