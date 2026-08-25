"""Schemas for user-uploaded document ingestion and management.

Note: the upload endpoint itself takes the file via multipart/form-data
(FastAPI's `UploadFile`), not a JSON body, so there is no `UploadRequest`
schema here — only the response models and the small delete request.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class UploadDocumentResponse(BaseModel):
    """Response returned after a document has been parsed, chunked, and indexed."""
    session_id: str = Field(..., description="Active session UUID the document is scoped to")
    filename: str = Field(..., description="Original uploaded filename")
    chunks_indexed: int = Field(..., description="Number of text chunks embedded and stored")
    message: str = Field(
        default="Document uploaded and indexed. You can now ask questions about it.",
        description="Human-readable confirmation message"
    )


class SessionDocumentsResponse(BaseModel):
    """Response describing whether a session currently has an indexed document."""
    session_id: str = Field(..., description="Active session UUID")
    has_document: bool = Field(..., description="Whether this session has any indexed document chunks")


class DeleteSessionDocumentsRequest(BaseModel):
    """Payload to remove a session's uploaded document(s) from the vector store."""
    session_id: str = Field(..., description="Active session UUID whose documents should be deleted")


class DeleteSessionDocumentsResponse(BaseModel):
    """Response confirming how many chunks were removed."""
    session_id: str = Field(..., description="Active session UUID")
    chunks_deleted: int = Field(..., description="Number of document chunks removed from the vector store")
    message: str = Field(default="Document(s) removed from this session.", description="Confirmation message")