"""Configuration module for Zanaco FAQ Chatbot Backend using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenRouter LLM Settings
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="API key for OpenRouter LLM access"
    )
    OPENROUTER_MODEL: str = Field(
        default="nvidia/nemotron-3.5-lightning:free",
        description="Free-tier OpenRouter instruct model name"
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API"
    )

    # Embeddings & Vector Database
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer embedding model name"
    )
    VECTOR_DB_PATH: str = Field(
        default="./data/chroma_db",
        description="Local directory for persistent ChromaDB storage"
    )
    COLLECTION_NAME: str = Field(
        default="zanaco_faq_collection",
        description="ChromaDB collection name for FAQ documents"
    )

    # RAG Scoring & Parameters
    SIMILARITY_THRESHOLD: float = Field(
        default=0.50,
        description="Minimum cosine similarity score required to generate answer"
    )
    TOP_K: int = Field(
        default=4,
        description="Number of top context chunks to retrieve"
    )

    # User-Uploaded Document RAG (session-scoped, separate from the fixed FAQ KB)
    USER_DOCS_COLLECTION_NAME: str = Field(
        default="user_uploaded_documents",
        description="ChromaDB collection name for session-scoped user-uploaded documents"
    )
    USER_DOC_SIMILARITY_THRESHOLD: float = Field(
        default=0.35,
        description=(
            "Minimum cosine similarity for an uploaded-document chunk to be considered "
            "relevant. Lower than the FAQ threshold on purpose: below this, we fall back "
            "to general LLM knowledge instead of refusing to answer."
        )
    )
    USER_DOC_TOP_K: int = Field(
        default=4,
        description="Number of top chunks to retrieve from a user's uploaded document"
    )
    GENERAL_LLM_FALLBACK_ENABLED: bool = Field(
        default=False,
        description=(
            "If True, when a question matches neither the uploaded document nor the FAQ "
            "knowledge base, answer using the LLM's general knowledge (clearly labeled as "
            "unofficial). If False (default, safer for banking), fall back to the existing "
            "live-agent escalation prompt instead of risking an ungrounded financial answer."
        )
    )
    SESSION_DOC_TTL_SECONDS: int = Field(
        default=86400,  # 24 hours
        description=(
            "How long an idle session's uploaded document stays indexed before the "
            "background sweep deletes it, for sessions that never reach /chat/rate."
        )
    )
    SESSION_DOC_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=3600,  # 1 hour
        description="How often the background sweep checks for and removes stale session documents."
    )

    # Server & Security
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, description="API port")
    ALLOWED_ORIGINS: Union[str, List[str]] = Field(
        default="*",
        description="Allowed CORS origins (comma-separated or list)"
    )
    RATE_LIMIT: str = Field(
        default="30/minute",
        description="Rate limit for chat endpoints"
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Knowledge Base Path
    FAQ_EXCEL_PATH: str = Field(
        default="Chat Bot Intents with Links (1).xlsx",
        description="Path to FAQ Excel workbook"
    )

    @property
    def cors_origins(self) -> List[str]:
        """Parse allowed origins into a list for FastAPI CORS middleware."""
        if isinstance(self.ALLOWED_ORIGINS, list):
            return self.ALLOWED_ORIGINS
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()