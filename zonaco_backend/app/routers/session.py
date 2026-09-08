"""Router for session lifecycle management, conversation history, and health probes."""

from fastapi import APIRouter, Depends, status
from app.config import Settings, get_settings
from app.dependencies import get_document_vector_store_dep, get_session_store_dep
from app.schemas.common import HealthResponse, MenuOption
from app.schemas.session import (
    MessageHistoryItem,
    SessionHistoryResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from app.services.document_store import DocumentVectorStoreService
from app.services.session_store import SessionStore
from app.state_machine import SessionState, StateMachine

router = APIRouter(tags=["Session Management"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and liveness probe",
    description="Returns service health status, configured LLM model, and vector store readiness."
)
async def health_check(
    settings: Settings = Depends(get_settings),
    doc_store: DocumentVectorStoreService = Depends(get_document_vector_store_dep)
) -> HealthResponse:
    """Liveness probe endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=settings.OPENROUTER_MODEL,
        vector_store_initialized=True
    )


@router.post(
    "/session/start",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize conversation session",
    description="Creates a new conversation session, initializes state machine, and presents the main menu."
)
async def start_session(
    payload: StartSessionRequest = StartSessionRequest(),
    session_store: SessionStore = Depends(get_session_store_dep)
) -> StartSessionResponse:
    """Create session and deliver welcome message with main menu options."""
    session = session_store.create_session(
        user_id=payload.user_id,
        metadata=payload.metadata
    )

    welcome_message = "Welcome to Zanaco Customer Support! How can we assist you today?"
    menu_options = [
        MenuOption(id="browse_faqs", label="Browse FAQ Categories"),
        MenuOption(id="privacy_policy", label="Privacy Policy"),
        MenuOption(id="talk_to_agent", label="Talk to an Agent"),
    ]

    # Log initial greeting to session history
    session_store.add_message(
        session_id=session.session_id,
        role="bot",
        content=welcome_message
    )

    return StartSessionResponse(
        session_id=session.session_id,
        current_state=session.current_state.value,
        message=welcome_message,
        menu_options=menu_options,
        next_valid_states=StateMachine.get_valid_next_states(session.current_state)
    )


@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Retrieve session history",
    description="Returns the full conversation transcript and metadata for an active or completed session."
)
async def get_session_history(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store_dep)
) -> SessionHistoryResponse:
    """Fetch complete conversation transcript for a session."""
    session = session_store.get_session(session_id)

    history_items = [
        MessageHistoryItem(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            intent=msg.intent,
            confidence_score=msg.confidence_score,
            links=msg.links
        )
        for msg in session.messages
    ]

    return SessionHistoryResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        current_state=session.current_state.value,
        is_satisfied=session.is_satisfied,
        rating=session.rating,
        feedback=session.feedback,
        messages=history_items
    )
