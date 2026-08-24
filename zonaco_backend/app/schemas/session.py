"""Schemas for session creation and history inspection."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import MenuOption


class StartSessionRequest(BaseModel):
    """Optional payload for initializing a new chat session."""
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional context metadata")


class StartSessionResponse(BaseModel):
    """Response returned upon successful session creation."""
    session_id: str = Field(..., description="Unique UUID for this conversation session")
    current_state: str = Field(..., description="Current session state machine status")
    message: str = Field(..., description="Initial greeting / welcome message")
    menu_options: List[MenuOption] = Field(..., description="Available top-level choices")
    next_valid_states: List[str] = Field(..., description="Permitted subsequent conversation states")


class MessageHistoryItem(BaseModel):
    """Represents a single message entry in conversation history."""
    role: str = Field(..., description="Sender role: 'user', 'bot', or 'system'")
    content: str = Field(..., description="Message text")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    intent: Optional[str] = Field(default=None, description="Matched FAQ intent if applicable")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score if applicable")
    links: Optional[List[str]] = Field(default=None, description="Relevant external URLs")


class SessionHistoryResponse(BaseModel):
    """Complete transcript and metadata for a conversation session."""
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    current_state: str = Field(..., description="Current state machine position")
    is_satisfied: Optional[bool] = Field(default=None, description="User satisfaction feedback status")
    rating: Optional[int] = Field(default=None, description="Numerical rating 1-5")
    feedback: Optional[str] = Field(default=None, description="User textual feedback")
    messages: List[MessageHistoryItem] = Field(..., description="Chronological message sequence")
