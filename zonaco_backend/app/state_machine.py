"""Session state machine and transition rules for Zanaco FAQ Chatbot."""

from enum import Enum
from typing import Dict, List, Optional, Set
from app.utils.exceptions import InvalidStateTransitionException


class SessionState(str, Enum):
    """Explicit states in the Zanaco chatbot conversation lifecycle."""
    WELCOME = "WELCOME"
    MAIN_MENU = "MAIN_MENU"
    CATEGORY_SELECT = "CATEGORY_SELECT"
    QUESTION_SELECT = "QUESTION_SELECT"
    ANSWERED = "ANSWERED"
    SATISFACTION_CHECK = "SATISFACTION_CHECK"
    MORE_QUESTIONS_CHECK = "MORE_QUESTIONS_CHECK"
    LIVE_AGENT = "LIVE_AGENT"
    RATING = "RATING"
    FEEDBACK = "FEEDBACK"
    PRIVACY_POLICY = "PRIVACY_POLICY"
    END = "END"


class StateMachine:
    """Manages valid state transitions and guarantees conversation protocol adherence."""

    # Explicit mapping of allowed forward transitions for each state
    ALLOWED_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
        SessionState.WELCOME: {
            SessionState.MAIN_MENU,
            SessionState.CATEGORY_SELECT,
            SessionState.PRIVACY_POLICY,
            SessionState.LIVE_AGENT,
            SessionState.ANSWERED,
        },
        SessionState.MAIN_MENU: {
            SessionState.CATEGORY_SELECT,
            SessionState.QUESTION_SELECT,
            SessionState.PRIVACY_POLICY,
            SessionState.LIVE_AGENT,
            SessionState.ANSWERED,
        },
        SessionState.CATEGORY_SELECT: {
            SessionState.QUESTION_SELECT,
            SessionState.ANSWERED,
            SessionState.MAIN_MENU,
            SessionState.PRIVACY_POLICY,
            SessionState.LIVE_AGENT,
        },
        SessionState.QUESTION_SELECT: {
            SessionState.ANSWERED,
            SessionState.CATEGORY_SELECT,
            SessionState.MAIN_MENU,
            SessionState.LIVE_AGENT,
        },
        SessionState.ANSWERED: {
            SessionState.SATISFACTION_CHECK,
            SessionState.MORE_QUESTIONS_CHECK,
            SessionState.CATEGORY_SELECT,
            SessionState.LIVE_AGENT,
            SessionState.ANSWERED,
        },
        SessionState.SATISFACTION_CHECK: {
            SessionState.MORE_QUESTIONS_CHECK,
            SessionState.LIVE_AGENT,
            SessionState.CATEGORY_SELECT,
            SessionState.ANSWERED,
        },
        SessionState.MORE_QUESTIONS_CHECK: {
            SessionState.CATEGORY_SELECT,
            SessionState.QUESTION_SELECT,
            SessionState.RATING,
            SessionState.FEEDBACK,
            SessionState.LIVE_AGENT,
            SessionState.MAIN_MENU,
            SessionState.ANSWERED,
        },
        SessionState.LIVE_AGENT: {
            SessionState.END,
            SessionState.MAIN_MENU,
        },
        SessionState.RATING: {
            SessionState.FEEDBACK,
            SessionState.END,
        },
        SessionState.FEEDBACK: {
            SessionState.END,
        },
        SessionState.PRIVACY_POLICY: {
            SessionState.MAIN_MENU,
            SessionState.CATEGORY_SELECT,
            SessionState.QUESTION_SELECT,
            SessionState.ANSWERED,
        },
        SessionState.END: set(),  # Terminal state
    }

    @classmethod
    def get_valid_next_states(cls, current_state: SessionState) -> List[str]:
        """Return list of valid next state names from the current state."""
        return [s.value for s in cls.ALLOWED_TRANSITIONS.get(current_state, set())]

    @classmethod
    def validate_and_transition(
        cls,
        current_state: SessionState,
        target_state: SessionState,
        action_name: str
    ) -> SessionState:
        """Validate transition from current_state to target_state or raise InvalidStateTransitionException."""
        allowed = cls.ALLOWED_TRANSITIONS.get(current_state, set())
        if target_state not in allowed:
            raise InvalidStateTransitionException(
                current_state=current_state.value,
                attempted_action=action_name,
                allowed_states=[s.value for s in allowed]
            )
        return target_state
