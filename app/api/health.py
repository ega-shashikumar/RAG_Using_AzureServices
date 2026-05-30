"""
app/api/health.py — Health check endpoint.
GET /health → checks all 4 Azure services in parallel.
"""
import asyncio

from fastapi import APIRouter

from app.core.azure_clients import get_blob_client, get_cosmos_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


async def _check_openai() -> ServiceStatus:
    try:
        from app.core.azure_clients import get_openai_client
        client = get_openai_client()
        await client.embeddings.create(
            model=get_settings().azure_openai_embedding_deployment,
            input=["ping"],
        )
        return ServiceStatus.OK
    except Exception as e:
        logger.warning("openai health check failed", error=str(e))
        return ServiceStatus.DOWN


async def _check_search() -> ServiceStatus:
    try:
        from app.core.azure_clients import get_search_client
        client = get_search_client()
        await client.get_document_count()
        return ServiceStatus.OK
    except Exception as e:
        logger.warning("search health check failed", error=str(e))
        return ServiceStatus.DOWN


async def _check_blob() -> ServiceStatus:
    try:
        settings = get_settings()
        client = get_blob_client()
        container = client.get_container_client(
            settings.azure_storage_container_name
        )
        await container.get_container_properties()
        return ServiceStatus.OK
    except Exception as e:
        logger.warning("blob health check failed", error=str(e))
        return ServiceStatus.DEGRADED


async def _check_cosmos() -> ServiceStatus:
    try:
        settings = get_settings()
        client = get_cosmos_client()
        db = client.get_database_client(
            settings.azure_cosmos_database
        )
        await db.read()
        return ServiceStatus.OK
    except Exception as e:
        logger.warning("cosmos health check failed", error=str(e))
        return ServiceStatus.DOWN


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check health of all Azure services",
)
async def health_check() -> HealthResponse:
    results = await asyncio.gather(
        _check_openai(),
        _check_search(),
        _check_blob(),
        _check_cosmos(),
    )

    services = {
        "azure_openai": results[0],
        "azure_search": results[1],
        "azure_blob": results[2],
        "azure_cosmos": results[3],
    }

    statuses = list(services.values())
    if all(s == ServiceStatus.OK for s in statuses):
        overall = ServiceStatus.OK
    elif any(s == ServiceStatus.OK for s in statuses):
        overall = ServiceStatus.DEGRADED
    else:
        overall = ServiceStatus.DOWN

    return HealthResponse(status=overall, services=services)