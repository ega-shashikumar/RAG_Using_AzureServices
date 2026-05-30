"""
app/services/history_service.py — Conversation history backed by Azure Cosmos DB.
"""
from __future__ import annotations

from datetime import datetime, timezone

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from app.core.azure_clients import get_cosmos_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ConversationHistory, HistoryMessage, MessageRole

logger = get_logger(__name__)

MAX_MESSAGES = 40  # keep last 20 turns (user + assistant)


class HistoryService:

    def __init__(self) -> None:
        self._settings = get_settings()

    async def _get_container(self):
        client = get_cosmos_client()
        db = client.get_database_client(
            self._settings.azure_cosmos_database
        )
        return db.get_container_client(
            self._settings.azure_cosmos_container
        )

    async def load(self, session_id: str) -> list[dict]:
        """
        Load last MAX_MESSAGES messages for a session.
        Returns raw dicts: [{"role": "user"|"assistant", "content": "..."}]
        """
        try:
            container = await self._get_container()
            item = await container.read_item(
                item=session_id,
                partition_key=session_id,
            )
            messages = item.get("messages", [])[-MAX_MESSAGES:]
            return [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]
        except CosmosResourceNotFoundError:
            return []
        except Exception as exc:
            logger.error(
                "failed to load history",
                session_id=session_id,
                error=str(exc),
            )
            return []

    async def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Append one user + one assistant turn.
        Non-fatal — errors are logged but never break the chat response.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            container = await self._get_container()
            try:
                item = await container.read_item(
                    item=session_id,
                    partition_key=session_id,
                )
                messages = item.get("messages", [])
            except CosmosResourceNotFoundError:
                messages = []
                item = {
                    "id": session_id,
                    "session_id": session_id,
                    "created_at": now,
                }

            messages.extend([
                {
                    "role": MessageRole.USER,
                    "content": user_message,
                    "timestamp": now,
                },
                {
                    "role": MessageRole.ASSISTANT,
                    "content": assistant_message,
                    "timestamp": now,
                },
            ])

            item["messages"] = messages[-MAX_MESSAGES:]
            item["updated_at"] = now
            await container.upsert_item(item)

            logger.info(
                "history saved",
                session_id=session_id,
                total_messages=len(item["messages"]),
            )
        except Exception as exc:
            logger.error(
                "failed to save history",
                session_id=session_id,
                error=str(exc),
            )

    async def get_conversation(
        self,
        session_id: str,
    ) -> ConversationHistory | None:
        """Fetch full conversation history for a session."""
        try:
            container = await self._get_container()
            item = await container.read_item(
                item=session_id,
                partition_key=session_id,
            )
            return ConversationHistory(
                session_id=session_id,
                messages=[
                    HistoryMessage(**m)
                    for m in item.get("messages", [])
                ],
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(
                    item.get("updated_at", item["created_at"])
                ),
            )
        except CosmosResourceNotFoundError:
            return None

    async def delete(self, session_id: str) -> bool:
        """Delete all history for a session. Returns True if deleted."""
        try:
            container = await self._get_container()
            await container.delete_item(
                item=session_id,
                partition_key=session_id,
            )
            logger.info("history deleted", session_id=session_id)
            return True
        except CosmosResourceNotFoundError:
            return False