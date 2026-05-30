"""
app/services/search_service.py — Azure AI Search index management and retrieval.
Handles index creation, document indexing, and hybrid search with semantic reranking.
"""
from __future__ import annotations

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.azure_clients import get_search_client, get_search_index_client
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SearchService:

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Index Management ──────────────────────────────────────────────────────

    async def ensure_index(self) -> None:
        """Create or update the search index schema — idempotent."""
        dims = self._settings.azure_openai_embedding_dimensions
        idx_client = get_search_index_client()

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="filename",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="tags",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="hnsw-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algo",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-algo",
                )
            ],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[
                            SemanticField(field_name="content")
                        ]
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=self._settings.azure_search_index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        await idx_client.create_or_update_index(index)
        logger.info(
            "search index ready",
            index=self._settings.azure_search_index_name,
        )

    # ── Indexing ──────────────────────────────────────────────────────────────

    async def index_documents(
        self,
        documents: list[dict],
    ) -> None:
        """Upload document chunks to the search index in batches."""
        search_client = get_search_client()
        batch_size = 100

        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            results = await search_client.upload_documents(batch)
            failed = [r for r in results if not r.succeeded]
            if failed:
                logger.error(
                    "some chunks failed to index",
                    failed_count=len(failed),
                )
            else:
                logger.info(
                    "indexed batch",
                    batch=i // batch_size + 1,
                    count=len(batch),
                )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Hybrid search: BM25 keyword + vector ANN.
        Applies semantic reranking if enabled.
        Returns list of result dicts.
        """
        top_k = top_k or self._settings.top_k_retrieval
        search_client = get_search_client()

        # Build OData filter
        filter_str: str | None = None
        if filters:
            parts = [f"{k} eq '{v}'" for k, v in filters.items()]
            filter_str = " and ".join(parts)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields="content_vector",
        )

        search_kwargs: dict = dict(
            search_text=query,
            vector_queries=[vector_query],
            select=[
                "id",
                "document_id",
                "filename",
                "chunk_index",
                "content",
                "source",
                "tags",
            ],
            top=top_k,
            filter=filter_str,
        )

        if self._settings.semantic_ranker_enabled:
            search_kwargs.update(
                query_type="semantic",
                semantic_configuration_name="default",
                query_caption="extractive",
                query_answer="extractive",
            )

        logger.info(
            "search started",
            query_preview=query[:80],
            top_k=top_k,
            semantic=self._settings.semantic_ranker_enabled,
        )

        results = []
        async for r in await search_client.search(**search_kwargs):
            results.append(
                {
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "filename": r["filename"],
                    "chunk_index": r["chunk_index"],
                    "content": r["content"],
                    "source": r.get("source", ""),
                    "tags": r.get("tags", ""),
                    "score": (
                        r.get("@search.reranker_score")
                        or r.get("@search.score", 0.0)
                    ),
                }
            )

        logger.info("search complete", results=len(results))
        return results