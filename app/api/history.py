"""
app/api/history.py — Conversation history endpoints.
GET    /history/{session_id} → ConversationHistory
DELETE /history/{session_id} → 204 No Content
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.middleware.auth import verify_api_key
from app.models.schemas import ConversationHistory
from app.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])


@router.get(
    "/{session_id}",
    response_model=ConversationHistory,
    dependencies=[Depends(verify_api_key)],
    summary="Get conversation history for a session",
)
async def get_history(session_id: str) -> ConversationHistory:
    svc = HistoryService()
    history = await svc.get_conversation(session_id)
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No history found for session '{session_id}'",
        )
    return history


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_api_key)],
    summary="Delete conversation history for a session",
)
async def delete_history(session_id: str) -> None:
    svc = HistoryService()
    deleted = await svc.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No history found for session '{session_id}'",
        )