"""
app/services/blob_service.py — Upload/download documents from Azure Blob Storage.
"""
from __future__ import annotations

import hashlib
import mimetypes

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob.aio import BlobServiceClient

from app.core.azure_clients import get_blob_client
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BlobService:
    def __init__(self) -> None:
        self._client: BlobServiceClient = get_blob_client()
        self._container = get_settings().azure_storage_container_name

    async def ensure_container(self) -> None:
        """Create container if it does not exist — idempotent."""
        try:
            container = self._client.get_container_client(self._container)
            await container.create_container()
            logger.info("blob container created", container=self._container)
        except ResourceExistsError:
            pass

    async def upload_document(
        self,
        filename: str,
        data: bytes,
        document_id: str,
    ) -> str:
        from azure.storage.blob import ContentSettings

        blob_name = f"{document_id}/{filename}"
        content_type, _ = mimetypes.guess_type(filename)
        content_type = content_type or "application/octet-stream"

        container_client = self._client.get_container_client(self._container)
        blob_client = container_client.get_blob_client(blob_name)

        await blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

        logger.info(
            "document uploaded to blob",
            blob_name=blob_name,
            size_bytes=len(data),
        )
        return blob_client.url

    async def download_document(
        self,
        document_id: str,
        filename: str,
    ) -> bytes:
        """Download raw document bytes from Blob Storage."""
        blob_name = f"{document_id}/{filename}"
        container_client = self._client.get_container_client(self._container)
        blob_client = container_client.get_blob_client(blob_name)
        stream = await blob_client.download_blob()
        return await stream.readall()

    @staticmethod
    def compute_document_id(data: bytes, filename: str) -> str:
        """
        Deterministic document ID based on file content + name.
        Re-uploading the same file always produces the same ID.
        """
        digest = hashlib.sha256(data + filename.encode()).hexdigest()[:16]
        return f"doc-{digest}"