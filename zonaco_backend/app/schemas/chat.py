"""Schemas for chat interactions, satisfaction, more questions, rating, and escalation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for submitting a question to the RAG pipeline."""
    session_id: Optional[str] = Field(default=None, description="Optional active session UUID")
    category: Optional[str] = Field(
        default=None,
        description="Optional category to scope the retrieval search"
    )
    question: str = Field(
        ...,
        min_length=2,
        description="User question text"
    )


class AskResponse(BaseModel):
    """Response returned from RAG generation or threshold fallback."""
    session_id: str = Field(..., description="Active session UUID")
    answer: str = Field(..., description="Generated answer or fallback explanation")
    answer_source: str = Field(
        default="faq",
        description=(
            "Where the answer came from: 'faq' (bank FAQ knowledge base), "
            "'user_document' (the session's uploaded document), or "
            "'general_llm' (model's own knowledge, used when neither source matched)"
        )
    )
    detected_category: Optional[str] = Field(
        default=None,
        description="Category of the best-matching FAQ chunk"
    )
    matched_faq_intent: Optional[str] = Field(
        default=None,
        description="Matched intent tag from retrieved knowledge base"
    )
    links: List[str] = Field(
        default=[],
        description="List of official URLs retrieved from FAQ details"
    )
    confidence_score: float = Field(
        ...,
        description="Cosine similarity confidence score between 0.0 and 1.0"
    )
    should_offer_escalation: bool = Field(
        ...,
        description="Flag indicating if low confidence / escalation to live agent is recommended"
    )
    current_state: str = Field(
        ...,
        description="Current conversation state (e.g. ANSWERED)"
    )
    prompt_for_satisfaction: str = Field(
        default="Was this answer helpful?",
        description="Follow-up prompt asking for user satisfaction"
    )
    next_valid_states: List[str] = Field(
        ...,
        description="Valid subsequent states"
    )


class SatisfactionRequest(BaseModel):
    """Payload indicating whether the user was satisfied with the answer."""
    session_id: str = Field(..., description="Active session UUID")
    satisfied: bool = Field(..., description="True if answer was helpful, False otherwise")


class SatisfactionResponse(BaseModel):
    """Response returned after processing user satisfaction feedback."""
    session_id: str = Field(..., description="Active session UUID")
    current_state: str = Field(..., description="New conversation state")
    message: str = Field(..., description="Bot reply message")
    next_valid_states: List[str] = Field(..., description="Permitted subsequent states")


class MoreQuestionsRequest(BaseModel):
    """Payload indicating whether user has additional questions."""
    session_id: str = Field(..., description="Active session UUID")
    more: bool = Field(..., description="True to ask more questions, False to conclude or escalate")


class MoreQuestionsResponse(BaseModel):
    """Response returned after processing more-questions choice."""
    session_id: str = Field(..., description="Active session UUID")
    current_state: str = Field(..., description="New conversation state")
    message: str = Field(..., description="Next prompt or routing message")
    next_valid_states: List[str] = Field(..., description="Permitted subsequent states")


class EscalateRequest(BaseModel):
    """Payload to trigger live agent handoff."""
    session_id: str = Field(..., description="Active session UUID")


class EscalateResponse(BaseModel):
    """Response confirming human agent queue placement."""
    session_id: str = Field(..., description="Active session UUID")
    current_state: str = Field(..., description="New conversation state (LIVE_AGENT)")
    message: str = Field(..., description="Confirmation and wait time guidance")
    agent_queue_ticket: str = Field(..., description="Queued support ticket number")
    next_valid_states: List[str] = Field(..., description="Permitted subsequent states")


class RateRequest(BaseModel):
    """Payload for submitting conversation rating and feedback."""
    session_id: str = Field(..., description="Active session UUID")
    rating: int = Field(..., ge=1, le=5, description="Satisfaction score from 1 to 5")
    feedback_text: Optional[str] = Field(
        default=None,
        description="Optional qualitative user feedback"
    )


class RateResponse(BaseModel):
    """Final thank-you response upon conversation closure."""
    session_id: str = Field(..., description="Active session UUID")
    current_state: str = Field(..., description="Terminal conversation state (END)")
    message: str = Field(..., description="Farewell / thank-you message")
    next_valid_states: List[str] = Field(default=[], description="Empty list (terminal state)")