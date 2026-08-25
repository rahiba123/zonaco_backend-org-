"""Thread-safe in-memory session management service with history tracking."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
import uuid
from app.state_machine import SessionState, StateMachine
from app.utils.exceptions import SessionNotFoundException
from app.utils.logger import logger


@dataclass
class ChatMessage:
    """A single logged message in the conversation transcript."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    intent: Optional[str] = None
    confidence_score: Optional[float] = None
    links: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "confidence_score": self.confidence_score,
            "links": self.links,
        }


@dataclass
class SessionData:
    """Full data model for an active or completed user session."""
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_state: SessionState = SessionState.WELCOME
    selected_category: Optional[str] = None
    is_satisfied: Optional[bool] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    agent_ticket: Optional[str] = None
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """Thread-safe repository for conversation sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """Initialize and store a brand-new user session."""
        session_id = str(uuid.uuid4())
        meta = metadata or {}
        if user_id:
            meta["user_id"] = user_id

        session = SessionData(
            session_id=session_id,
            current_state=SessionState.MAIN_MENU,
            metadata=meta
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info(f"Initialized new session {session_id}")
        return session

    def get_session(self, session_id: str) -> SessionData:
        """Retrieve existing session or raise SessionNotFoundException."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session lookup failed for ID: {session_id}")
                raise SessionNotFoundException(session_id)
            return session

    def list_stale_session_ids(self, ttl_seconds: int) -> List[str]:
        """Return session IDs whose last activity is older than ttl_seconds.

        Used by the periodic cleanup sweep to find abandoned sessions (e.g. the
        user closed the tab without reaching the END state) so their uploaded
        document vectors don't accumulate indefinitely in ChromaDB.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
        with self._lock:
            return [
                sid for sid, session in self._sessions.items()
                if session.updated_at.timestamp() < cutoff
            ]

    def transition_state(
        self,
        session_id: str,
        target_state: SessionState,
        action_name: str
    ) -> SessionData:
        """Validate state machine rules and transition session to target_state."""
        session = self.get_session(session_id)
        with self._lock:
            new_state = StateMachine.validate_and_transition(
                current_state=session.current_state,
                target_state=target_state,
                action_name=action_name
            )
            session.current_state = new_state
            session.updated_at = datetime.now(timezone.utc)
            logger.info(f"Session {session_id} transitioned: {session.current_state} -> {new_state}")
            return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        confidence_score: Optional[float] = None,
        links: Optional[List[str]] = None
    ) -> ChatMessage:
        """Append a message to the session's conversation history."""
        session = self.get_session(session_id)
        msg = ChatMessage(
            role=role,
            content=content,
            intent=intent,
            confidence_score=confidence_score,
            links=links
        )
        with self._lock:
            session.messages.append(msg)
            session.updated_at = datetime.now(timezone.utc)
        return msg

    def set_category(self, session_id: str, category: str) -> None:
        """Set the active FAQ category context for the session."""
        session = self.get_session(session_id)
        with self._lock:
            session.selected_category = category
            session.updated_at = datetime.now(timezone.utc)

    def set_satisfaction(self, session_id: str, satisfied: bool) -> None:
        """Record satisfaction feedback on the session."""
        session = self.get_session(session_id)
        with self._lock:
            session.is_satisfied = satisfied
            session.updated_at = datetime.now(timezone.utc)

    def set_rating_and_feedback(
        self,
        session_id: str,
        rating: int,
        feedback: Optional[str] = None
    ) -> None:
        """Record final 1-5 rating and optional feedback."""
        session = self.get_session(session_id)
        with self._lock:
            session.rating = rating
            session.feedback = feedback
            session.updated_at = datetime.now(timezone.utc)

    def set_agent_ticket(self, session_id: str, ticket_number: str) -> None:
        """Record human escalation queue ticket number."""
        session = self.get_session(session_id)
        with self._lock:
            session.agent_ticket = ticket_number
            session.updated_at = datetime.now(timezone.utc)


# Singleton session store instance
session_store = SessionStore()


def get_session_store() -> SessionStore:
    """Dependency injection helper to obtain SessionStore instance."""
    return session_store