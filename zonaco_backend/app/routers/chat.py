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
        detected_category = rag_result.retrieved_sources[0].category

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


@router.post(
    "/satisfaction",
    response_model=SatisfactionResponse,
    summary="Record answer helpfulness",
    description="Registers user satisfaction ('Was this helpful?') and moves to MORE_QUESTIONS_CHECK."
)
async def submit_satisfaction(
    payload: SatisfactionRequest,
    session_store: SessionStore = Depends(get_session_store_dep)
) -> SatisfactionResponse:
    """Process user satisfaction evaluation."""
    session = session_store.get_session(payload.session_id)
    session_store.set_satisfaction(payload.session_id, payload.satisfied)

    session_store.transition_state(
        session_id=session.session_id,
        target_state=SessionState.MORE_QUESTIONS_CHECK,
        action_name="submit_satisfaction"
    )

    if payload.satisfied:
        reply_message = "Glad to hear that! Do you have any more questions?"
    else:
        reply_message = "I apologize that the information was not fully helpful. Do you have any more questions?"

    session_store.add_message(
        session_id=session.session_id,
        role="bot",
        content=reply_message
    )

    return SatisfactionResponse(
        session_id=session.session_id,
        current_state=SessionState.MORE_QUESTIONS_CHECK.value,
        message=reply_message,
        next_valid_states=StateMachine.get_valid_next_states(SessionState.MORE_QUESTIONS_CHECK)
    )


@router.post(
    "/more-questions",
    response_model=MoreQuestionsResponse,
    summary="Handle more questions decision",
    description="Routes session back to category browsing if 'more=True', or to rating/agent if 'more=False'."
)
async def submit_more_questions(
    payload: MoreQuestionsRequest,
    session_store: SessionStore = Depends(get_session_store_dep)
) -> MoreQuestionsResponse:
    """Handle branch after more questions prompt."""
    session = session_store.get_session(payload.session_id)

    if payload.more:
        # Branch: YES -> back to FAQ Categories
        session_store.transition_state(
            session_id=session.session_id,
            target_state=SessionState.CATEGORY_SELECT,
            action_name="more_questions_yes"
        )
        msg = "Please select an FAQ category or ask your next question:"
        session_store.add_message(session_id=session.session_id, role="bot", content=msg)
        return MoreQuestionsResponse(
            session_id=session.session_id,
            current_state=SessionState.CATEGORY_SELECT.value,
            message=msg,
            next_valid_states=StateMachine.get_valid_next_states(SessionState.CATEGORY_SELECT)
        )
    else:
        # Branch: NO -> If satisfied: RATING; If not satisfied: LIVE_AGENT
        if session.is_satisfied is False:
            # Escalation path
            session_store.transition_state(
                session_id=session.session_id,
                target_state=SessionState.LIVE_AGENT,
                action_name="more_questions_no_unsatisfied"
            )
            ticket = f"ZNCO-AGENT-{uuid.uuid4().hex[:6].upper()}"
            session_store.set_agent_ticket(session.session_id, ticket)
            msg = (
                f"Connecting you with a Zanaco customer support specialist. "
                f"Your queue reference is {ticket}. An agent will assist you shortly."
            )
            session_store.add_message(session_id=session.session_id, role="bot", content=msg)
            return MoreQuestionsResponse(
                session_id=session.session_id,
                current_state=SessionState.LIVE_AGENT.value,
                message=msg,
                next_valid_states=StateMachine.get_valid_next_states(SessionState.LIVE_AGENT)
            )
        else:
            # Rating path
            session_store.transition_state(
                session_id=session.session_id,
                target_state=SessionState.RATING,
                action_name="more_questions_no_satisfied"
            )
            msg = "Thank you for chatting with Zanaco! Please rate your conversation today from 1 (poor) to 5 (excellent)."
            session_store.add_message(session_id=session.session_id, role="bot", content=msg)
            return MoreQuestionsResponse(
                session_id=session.session_id,
                current_state=SessionState.RATING.value,
                message=msg,
                next_valid_states=StateMachine.get_valid_next_states(SessionState.RATING)
            )


@router.post(
    "/escalate",
    response_model=EscalateResponse,
    summary="Escalate to human support agent",
    description="Transfers the session into the human support queue and generates an agent ticket."
)
async def escalate_to_agent(
    payload: EscalateRequest,
    session_store: SessionStore = Depends(get_session_store_dep)
) -> EscalateResponse:
    """Directly escalate conversation to a live banking support agent."""
    session = session_store.get_session(payload.session_id)

    session_store.transition_state(
        session_id=session.session_id,
        target_state=SessionState.LIVE_AGENT,
        action_name="escalate_to_agent"
    )

    ticket = f"ZNCO-ESC-{uuid.uuid4().hex[:6].upper()}"
    session_store.set_agent_ticket(session.session_id, ticket)

    msg = (
        f"Connecting you with a Zanaco live support agent. "
        f"Your queue ticket is #{ticket}. A representative will join this chat momentarily."
    )
    session_store.add_message(session_id=session.session_id, role="bot", content=msg)

    return EscalateResponse(
        session_id=session.session_id,
        current_state=SessionState.LIVE_AGENT.value,
        message=msg,
        agent_queue_ticket=ticket,
        next_valid_states=StateMachine.get_valid_next_states(SessionState.LIVE_AGENT)
    )


@router.post(
    "/rate",
    response_model=RateResponse,
    summary="Submit rating and feedback",
    description="Stores 1-5 star rating and feedback comment, closing the conversation (state END)."
)
async def rate_conversation(
    payload: RateRequest,
    session_store: SessionStore = Depends(get_session_store_dep),
    doc_store: DocumentVectorStoreService = Depends(get_document_vector_store_dep),
) -> RateResponse:
    """Collect user rating and feedback, terminating the session."""
    session = session_store.get_session(payload.session_id)

    session_store.set_rating_and_feedback(
        session_id=session.session_id,
        rating=payload.rating,
        feedback=payload.feedback_text
    )

    session_store.transition_state(
        session_id=session.session_id,
        target_state=SessionState.END,
        action_name="rate_conversation"
    )

    # Session has definitively ended: clean up any uploaded document immediately
    # rather than waiting for the background TTL sweep.
    try:
        deleted = doc_store.delete_session_documents(session.session_id)
        if deleted:
            logger.info(f"Cleaned up {deleted} document chunk(s) for ended session {session.session_id}.")
    except Exception as exc:
        # Cleanup failure shouldn't block the user from seeing their thank-you message;
        # the background sweep will catch it later via SESSION_DOC_TTL_SECONDS.
        logger.warning(f"Document cleanup failed for session {session.session_id}: {exc}")

    thank_you_message = (
        f"Thank you for your {payload.rating}-star rating! "
        f"We appreciate your feedback and look forward to serving you again. Have a great day with Zanaco!"
    )
    session_store.add_message(
        session_id=session.session_id,
        role="bot",
        content=thank_you_message
    )

    return RateResponse(
        session_id=session.session_id,
        current_state=SessionState.END.value,
        message=thank_you_message,
        next_valid_states=[]
    )