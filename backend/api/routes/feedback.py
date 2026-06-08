"""
Feedback collection endpoint.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from backend.api.middleware.auth import verify_api_key
from backend.core.schemas import FeedbackRequest, FeedbackResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    _api_key: str = Depends(verify_api_key),
) -> FeedbackResponse:
    """
    Collect user feedback (thumbs up/down, comments) on a response.

    Feedback is logged and can be used for fine-tuning, RLHF, or
    identifying hallucinations in production.
    """
    await logger.ainfo(
        "feedback_received",
        message_id=feedback.message_id,
        conversation_id=feedback.conversation_id,
        rating=feedback.rating.value,
        is_hallucination=feedback.is_hallucination,
        has_comment=feedback.comment is not None,
    )

    # In production, persist to database — for now, structured log is enough
    # TODO: Persist to PostgreSQL feedback table for analysis
    return FeedbackResponse(status="received")
