"""Custom application exceptions and FastAPI exception handlers."""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.utils.logger import logger


class ChatbotException(Exception):
    """Base exception for all Zanaco Chatbot errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class SessionNotFoundException(ChatbotException):
    """Raised when a requested session ID does not exist."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session with ID '{session_id}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id}
        )


class InvalidStateTransitionException(ChatbotException):
    """Raised when an operation is attempted in an invalid conversation state."""

    def __init__(self, current_state: str, attempted_action: str, allowed_states: list):
        super().__init__(
            message=(
                f"Cannot perform '{attempted_action}' in state '{current_state}'. "
                f"Valid next states/actions are: {allowed_states}."
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
            details={
                "current_state": current_state,
                "attempted_action": attempted_action,
                "allowed_states": allowed_states
            }
        )


class CategoryNotFoundException(ChatbotException):
    """Raised when an unknown FAQ category is specified."""

    def __init__(self, category: str):
        super().__init__(
            message=f"FAQ category '{category}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"category": category}
        )


class VectorStoreException(ChatbotException):
    """Raised when ChromaDB operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Vector store error: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class LLMServiceException(ChatbotException):
    """Raised when OpenRouter API communication fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"LLM service error: {message}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""

    @app.exception_handler(ChatbotException)
    async def chatbot_exception_handler(request: Request, exc: ChatbotException):
        logger.error(f"ChatbotException on {request.url.path}: {exc.message} (status {exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.message,
                "status_code": exc.status_code,
                "details": exc.details
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "message": "Request validation failed.",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "details": {"errors": exc.errors()}
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled internal server error on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "message": "An internal server error occurred. Please try again later.",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "details": {"error_type": type(exc).__name__}
            }
        )
