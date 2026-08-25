"""FastAPI dependency injection providers."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import Settings, get_settings
from app.services.document_parser import DocumentParser
from app.services.document_store import DocumentVectorStoreService, get_document_vector_store
from app.services.excel_parser import ExcelFAQParser
from app.services.rag import RAGService, get_rag_service
from app.services.session_store import SessionStore, get_session_store
from app.services.vector_store import VectorStoreService, get_vector_store

# Shared slowapi rate limiter instance
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


def get_limiter() -> Limiter:
    """Return configured slowapi rate limiter."""
    return limiter


def get_session_store_dep() -> SessionStore:
    """Provide session store dependency."""
    return get_session_store()


def get_vector_store_dep() -> VectorStoreService:
    """Provide vector store service dependency."""
    return get_vector_store()


def get_rag_service_dep() -> RAGService:
    """Provide RAG service dependency."""
    return get_rag_service()


def get_excel_parser_dep() -> ExcelFAQParser:
    """Provide Excel parser dependency."""
    return ExcelFAQParser()


def get_document_parser_dep() -> DocumentParser:
    """Provide user-uploaded document parser dependency. Stateless, safe to build fresh."""
    return DocumentParser()


def get_document_vector_store_dep() -> DocumentVectorStoreService:
    """Provide session-scoped document vector store dependency."""
    return get_document_vector_store()


def get_settings_dep() -> Settings:
    """Provide application settings dependency."""
    return get_settings()