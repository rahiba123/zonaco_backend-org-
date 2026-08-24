"""Structured logging configuration for Zanaco Chatbot."""

import logging
import sys
from typing import Any, Dict, Optional
from app.config import get_settings


def setup_logger(name: str = "zonaco_chatbot") -> logging.Logger:
    """Configure and return a structured logger instance."""
    settings = get_settings()
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger()


def log_chat_interaction(
    session_id: str,
    question: str,
    category: Optional[str],
    confidence_score: float,
    matched_intent: Optional[str],
    should_offer_escalation: bool,
    latency_ms: float,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log structured details of every /chat/ask interaction for analytics and auditing."""
    status = "ESCALATION_OFFERED" if should_offer_escalation else "MATCHED"
    msg = (
        f"[CHAT_QUERY] Session={session_id} | Category={category} | Intent={matched_intent} "
        f"| Score={confidence_score:.4f} | Status={status} | Latency={latency_ms:.2f}ms | Question=\"{question}\""
    )
    if should_offer_escalation:
        logger.warning(msg)
    else:
        logger.info(msg)
