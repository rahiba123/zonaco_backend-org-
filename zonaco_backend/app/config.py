"""Configuration module for Chatbot Backend using Pydantic Settings."""

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

    # User-Uploaded Document RAG (session-scoped)
    USER_DOCS_COLLECTION_NAME: str = Field(
        default="user_uploaded_documents",
        description="ChromaDB collection name for session-scoped user-uploaded documents"
    )
    USER_DOC_SIMILARITY_THRESHOLD: float = Field(
        default=0.20,
        description="Minimum cosine similarity for an uploaded-document chunk to be considered relevant."
    )
    USER_DOC_TOP_K: int = Field(
        default=4,
        description="Number of top chunks to retrieve from a user's uploaded document"
    )
    SESSION_DOC_TTL_SECONDS: int = Field(
        default=86400,  # 24 hours
        description="How long an idle session's uploaded document stays indexed before the background sweep deletes it."
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