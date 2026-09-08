"""Router for uploading, indexing, and managing session-scoped user documents.

This is deliberately kept separate from `chat.py`'s state machine — uploading
a document is not itself a conversation-state transition; it's a prerequisite
step that a session can perform at any point before or during a chat.
"""

from fastapi import APIRouter, Depends, File, Request, UploadFile
from app.dependencies import (
    get_document_parser_dep,
    get_document_vector_store_dep,
    get_limiter,
    get_session_store_dep,
)
from app.schemas.documents import (
    DeleteSessionDocumentsResponse,
    SessionDocumentsResponse,
    UploadDocumentResponse,
)
from app.services.document_parser import DocumentParser
from app.services.document_store import DocumentVectorStoreService
from app.services.session_store import SessionStore
from app.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["Document Upload & Session RAG"])
limiter = get_limiter()


@router.post(
    "/upload",
    response_model=UploadDocumentResponse,
    summary="Upload a document for this session",
    description=(
        "Accepts a PDF, DOCX, TXT, XLSX, XLS, CSV, or Markdown file, extracts and chunks its text, embeds it, "
        "and indexes it in a session-scoped vector store. Once uploaded, questions sent "
        "to /chat/ask will be checked against this document before falling back to the "
        "FAQ knowledge base or general LLM knowledge."
    )
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
    session_store: SessionStore = Depends(get_session_store_dep),
    parser: DocumentParser = Depends(get_document_parser_dep),
    doc_store: DocumentVectorStoreService = Depends(get_document_vector_store_dep),
) -> UploadDocumentResponse:
    """Parse, chunk, embed, and index an uploaded document for the given session."""
    # Auto-create a session if none was supplied, mirroring /chat/ask's behavior.
    if session_id:
        try:
            session = session_store.get_session(session_id)
        except Exception:
            logger.info(f"Unrecognized session_id: {session_id}. Auto-creating a new session...")
            session = session_store.create_session()
            session_id = session.session_id
    else:
        session = session_store.create_session()
        session_id = session.session_id

    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"
    chunks = parser.parse(session_id=session_id, filename=filename, file_bytes=file_bytes)
    chunks_indexed = doc_store.add_chunks(chunks)

    session_store.add_message(
        session_id=session_id,
        role="user",
        content=f"[Uploaded document: {filename}]"
    )
    session_store.add_message(
        session_id=session_id,
        role="bot",
        content=f"I've indexed '{filename}' ({chunks_indexed} sections). You can now ask questions about it."
    )

    return UploadDocumentResponse(
        session_id=session_id,
        filename=filename,
        chunks_indexed=chunks_indexed,
    )


@router.get(
    "/{session_id}/status",
    response_model=SessionDocumentsResponse,
    summary="Check whether this session has an indexed document",
)
async def get_document_status(
    session_id: str,
    doc_store: DocumentVectorStoreService = Depends(get_document_vector_store_dep),
) -> SessionDocumentsResponse:
    """Report whether the given session currently has any indexed document chunks."""
    return SessionDocumentsResponse(
        session_id=session_id,
        has_document=doc_store.has_documents(session_id)
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteSessionDocumentsResponse,
    summary="Remove this session's uploaded document(s)",
    description="Deletes all indexed chunks for the session, e.g. to let the user upload a fresh document.",
)
async def delete_session_documents(
    session_id: str,
    doc_store: DocumentVectorStoreService = Depends(get_document_vector_store_dep),
) -> DeleteSessionDocumentsResponse:
    """Delete all document chunks belonging to a session."""
    deleted_count = doc_store.delete_session_documents(session_id)
    return DeleteSessionDocumentsResponse(
        session_id=session_id,
        chunks_deleted=deleted_count,
    )