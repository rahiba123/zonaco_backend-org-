"""Router for chat Q&A, satisfaction checks, more questions, rating, and agent escalation."""

import uuid
from fastapi import APIRouter, Depends, Request
from app.config import Settings, get_settings
from app.dependencies import (
    get_document_vector_store_dep,
    get_limiter,
    get_rag_service_dep,
    get_session_store_dep,
)
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    EscalateRequest,
    EscalateResponse,
    MoreQuestionsRequest,
    MoreQuestionsResponse,
    RateRequest,
    RateResponse,
    SatisfactionRequest,
    SatisfactionResponse,
)
from app.services.rag import RAGService
from app.services.document_store import DocumentVectorStoreService
from app.services.session_store import SessionStore
from app.state_machine import SessionState, StateMachine
from app.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["Chat & Conversational Engine"])
limiter = get_limiter()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Submit question to RAG pipeline",
    description="Runs vector similarity search and LLM generation with category scoping and threshold checks."
)
@limiter.limit("30/minute")
async def ask_question(
    request: Request,
    payload: AskRequest,
    session_store: SessionStore = Depends(get_session_store_dep),
    rag_service: RAGService = Depends(get_rag_service_dep)
) -> AskResponse:
    """Process user question through RAG pipeline and advance state machine."""
    session = None
    session_id = payload.session_id

    if session_id:
        try:
            session = session_store.get_session(session_id)
        except Exception:
            logger.info(f"Unrecognized session_id: {session_id}. Auto-creating a new session...")
            session = session_store.create_session()
            session_id = session.session_id
    else:
        logger.info("No session_id provided. Auto-creating a new session...")
        session = session_store.create_session()
        session_id = session.session_id

    # Determine effective category from request only (avoid silent default to session category)
    effective_category = payload.category
    if effective_category:
        effective_category_clean = effective_category.strip().lower()
        if effective_category_clean in ("string", "null", ""):
            effective_category = None

    if effective_category:
        session_store.set_category(session_id, effective_category)

    # Record user message in history
    session_store.add_message(
        session_id=session_id,
        role="user",
        content=payload.question
    )

    # Execute RAG pipeline
    rag_result = await rag_service.answer_question(
        session_id=session_id,
        question=payload.question,
        category=effective_category
    )

    # Transition state machine to ANSWERED
    session_store.transition_state(
        session_id=session_id,
        target_state=SessionState.ANSWERED,
        action_name="ask_question"
    )

    # Record bot response in history
    session_store.add_message(
        session_id=session_id,
        role="bot",
        content=rag_result.answer,
        intent=rag_result.matched_faq_intent,
        confidence_score=rag_result.confidence_score,
        links=rag_result.links
    )

    # Extract detected category from top match if available
    detected_category = None
    if rag_result.retrieved_sources:
        detected_category = getattr(rag_result.retrieved_sources[0], "category", None)

    return AskResponse(
        session_id=session_id,
        answer=rag_result.answer,
        answer_source=rag_result.answer_source,
        detected_category=detected_category,
        matched_faq_intent=rag_result.matched_faq_intent,
        links=rag_result.links,
        confidence_score=rag_result.confidence_score,
        should_offer_escalation=rag_result.should_offer_escalation,
        current_state=SessionState.ANSWERED.value,
        prompt_for_satisfaction="Was this answer helpful?",
        next_valid_states=StateMachine.get_valid_next_states(SessionState.ANSWERED)
    )