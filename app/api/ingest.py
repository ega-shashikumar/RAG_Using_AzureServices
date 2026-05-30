"""
app/api/ingest.py — Document ingestion endpoint.
POST /ingest (multipart/form-data: file + optional metadata JSON)
"""
import json

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.core.logging import get_logger
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.schemas import IngestMetadata, IngestResponse
from app.services.ingestion_service import IngestionService
from app.utils.chunker import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)

MAX_FILE_MB = 50


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
    summary="Ingest a document into the RAG pipeline",
)
@limiter.limit("30/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
) -> IngestResponse:

    # Validate file extension
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{ext}' not supported. "
                f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Read and validate file size
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_MB} MB"
            ),
        )

    # Parse metadata
    try:
        meta_dict = json.loads(metadata)
        ingest_meta = IngestMetadata(**meta_dict)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid metadata JSON: {e}",
        )

    logger.info(
        "ingest request received",
        filename=file.filename,
        size_mb=round(size_mb, 2),
    )

    # Run ingestion pipeline
    service = IngestionService()
    try:
        result = await service.ingest(
            filename=file.filename or "document",
            data=data,
            metadata=ingest_meta,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error("ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingestion failed. Check server logs for details.",
        )

    return result