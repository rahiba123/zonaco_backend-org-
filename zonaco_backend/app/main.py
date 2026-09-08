"""FastAPI main application entry point for Document Chatbot."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from app.config import get_settings
from app.dependencies import limiter
from app.routers import chat, documents, session
from app.services.document_store import get_document_vector_store
from app.services.session_store import get_session_store
from app.utils.exceptions import register_exception_handlers
from app.utils.logger import logger

settings = get_settings()


async def _stale_document_cleanup_loop():
    """Background sweep: periodically deletes uploaded document vectors for idle sessions."""
    session_store = get_session_store()
    doc_store = get_document_vector_store()

    while True:
        try:
            await asyncio.sleep(settings.SESSION_DOC_CLEANUP_INTERVAL_SECONDS)
            stale_ids = session_store.list_stale_session_ids(settings.SESSION_DOC_TTL_SECONDS)
            total_deleted = 0
            for session_id in stale_ids:
                if doc_store.has_documents(session_id):
                    total_deleted += doc_store.delete_session_documents(session_id)
            if total_deleted:
                logger.info(
                    f"Stale-document sweep: removed {total_deleted} chunk(s) across "
                    f"{len(stale_ids)} idle session(s)."
                )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"Stale-document cleanup sweep encountered an error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting Document Chatbot Backend...")
    logger.info(f"Configured OpenRouter model: {settings.OPENROUTER_MODEL}")
    logger.info(f"Vector database path: {settings.VECTOR_DB_PATH}")

    cleanup_task = asyncio.create_task(_stale_document_cleanup_loop())
    logger.info(
        f"Started background document cleanup sweep "
        f"(every {settings.SESSION_DOC_CLEANUP_INTERVAL_SECONDS}s, TTL {settings.SESSION_DOC_TTL_SECONDS}s)."
    )

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down Document Chatbot Backend...")


# FastAPI application instance
app = FastAPI(
    title="Document Chatbot API",
    description="Customer support chatbot backend with document upload and grounded RAG Q&A.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# SlowAPI Rate Limiter state registration
app.state.limiter = limiter


# Custom rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded on {request.url.path} from client {request.client.host if request.client else 'unknown'}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": True,
            "message": "Rate limit exceeded. Please wait a moment before sending more messages.",
            "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
            "details": {"retry_after": str(exc.detail)}
        }
    )


# Register domain and generic exception handlers
register_exception_handlers(app)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(session.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to API documentation."""
    return {
        "service": "Document Chatbot Backend",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }