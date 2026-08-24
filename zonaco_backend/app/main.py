"""FastAPI main application entry point for Zanaco FAQ Chatbot."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from app.config import get_settings
from app.dependencies import limiter
from app.routers import categories, chat, session
from app.services.vector_store import get_vector_store
from app.utils.exceptions import register_exception_handlers
from app.utils.logger import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting Zanaco FAQ Chatbot Backend...")
    logger.info(f"Configured OpenRouter model: {settings.OPENROUTER_MODEL}")
    logger.info(f"Vector database path: {settings.VECTOR_DB_PATH}")

    # Ensure vector store is initialized
    try:
        vs = get_vector_store()
        doc_count = vs.count()
        logger.info(f"ChromaDB ready with {doc_count} indexed FAQ documents.")
        if doc_count == 0:
            logger.warning(
                "ChromaDB has 0 documents! Please run 'python ingest.py' to index the FAQ knowledge base."
            )
    except Exception as exc:
        logger.error(f"Vector store initialization check encountered error: {exc}")

    yield

    logger.info("Shutting down Zanaco FAQ Chatbot Backend...")


# FastAPI application instance
app = FastAPI(
    title="Zanaco Banking FAQ Chatbot API",
    description=(
        "Production-ready customer support chatbot backend for Zambia National Commercial Bank (Zanaco). "
        "Implements a state machine session lifecycle, scoped ChromaDB vector retrieval, and OpenRouter RAG."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
app.include_router(categories.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to API documentation."""
    return {
        "service": "Zanaco Banking FAQ Chatbot Backend",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }
