"""
app/core/azure_clients.py — Singleton Azure SDK clients.
All clients are instantiated once and reused across requests.
"""
from functools import lru_cache

from azure.cosmos.aio import CosmosClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncAzureOpenAI:
    settings = get_settings()
    logger.info("initialising Azure OpenAI client")
    return AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    settings = get_settings()
    from azure.core.credentials import AzureKeyCredential
    logger.info("initialising Azure AI Search client")
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    settings = get_settings()
    from azure.core.credentials import AzureKeyCredential
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key),
    )


@lru_cache(maxsize=1)
def get_blob_client() -> BlobServiceClient:
    settings = get_settings()
    logger.info("initialising Azure Blob Storage client")
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )


@lru_cache(maxsize=1)
def get_cosmos_client() -> CosmosClient:
    settings = get_settings()
    logger.info("initialising Azure Cosmos DB client")
    return CosmosClient(
        url=settings.azure_cosmos_endpoint,
        credential=settings.azure_cosmos_key,
    )