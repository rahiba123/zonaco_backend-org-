"""Common response and error schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MenuOption(BaseModel):
    """Represents an interactive menu item option."""
    id: str = Field(..., description="Unique identifier for the action/option")
    label: str = Field(..., description="User-facing label for the option")


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field(default="ok", description="Server operational status")
    version: str = Field(default="1.0.0", description="API version")
    model: str = Field(..., description="Configured LLM model")
    vector_store_initialized: bool = Field(..., description="ChromaDB status")


class ErrorResponse(BaseModel):
    """Standardized API error response format."""
    error: bool = Field(default=True, description="Indicates error presence")
    message: str = Field(..., description="Human-readable error description")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic details")
