# Zanaco Banking FAQ Chatbot Backend (FastAPI + RAG)

A production-ready customer support chatbot backend for **Zambia National Commercial Bank (Zanaco)**. It answers customer inquiries from a fixed FAQ knowledge base using a **Retrieval-Augmented Generation (RAG)** pipeline and escalates to a human support agent when necessary.

---

## Architecture Overview

```
                                  +-------------------------------------------------------+
                                  |              Excel Knowledge Base                     |
                                  |     (Chat_Bot_Intents_with_Links.xlsx)               |
                                  +---------------------------+---------------------------+
                                                              |
                                                              v  (ingest.py - offline/idempotent)
                                  +-------------------------------------------------------+
                                  |          ChromaDB (Persistent Vector Store)          |
                                  |   - Embeddings: sentence-transformers (Local CPU)    |
                                  |   - Filterable metadata: category, faq_intent, links |
                                  +---------------------------+---------------------------+
                                                              |
                                                              | Scoped Similarity Search
                                                              v
+------------------+   HTTP Requests    +---------------------------------------------------------+
|                  | =================> |                      FastAPI Backend                    |
|                  |                    |  - CORS & slowapi Rate Limiting                         |
|  Client / Web UI |                    |  - Explicit Session State Machine                       |
|                  | <================= |  - Global Exception Handling & Structured Logging       |
+------------------+   JSON Responses   +-----------------------------+---------------------------+
                                                                      |
                                                                      | Prompt + Context Chunks
                                                                      v
                                        +---------------------------------------------------------+
                                        |                   OpenRouter API                        |
                                        |  - Free-tier model (e.g. meta-llama/llama-3.3-70b-free) |
                                        |  - Strict system prompt: no hallucinations, cite context|
                                        +---------------------------------------------------------+
```

---

## Features

- **Strict Conversation State Machine**: Manages state transitions (`WELCOME` $\rightarrow$ `MAIN_MENU` $\rightarrow$ `CATEGORY_SELECT` $\rightarrow$ `QUESTION_SELECT` $\rightarrow$ `ANSWERED` $\rightarrow$ `SATISFACTION_CHECK` $\rightarrow$ `MORE_QUESTIONS_CHECK` $\rightarrow$ `RATING` / `FEEDBACK` or `LIVE_AGENT` $\rightarrow$ `END`).
- **Category-Scoped Vector Search**: ChromaDB vector queries are filtered by category metadata (e.g., `CARD SERVICES`, `INTERNET BANKING`) to prevent cross-topic confusion.
- **Local Embeddings (CPU-based & Free)**: Powered by `sentence-transformers/all-MiniLM-L6-v2` with zero per-embedding cost.
- **OpenRouter Free Tier LLM Integration**: Generates grounded banking responses using free-tier instruct models with strict guardrails against hallucinations.
- **Automatic Link Extraction**: Extracts URLs from FAQ entries via regex and returns them cleanly in metadata rather than relying on LLM recall.
- **Confidence Threshold Fallback**: Queries below `SIMILARITY_THRESHOLD` bypass the LLM and offer live agent escalation.
- **Idempotent Ingestion**: `ingest.py` generates deterministic SHA-256 document IDs to avoid duplicate embeddings upon re-indexing.
- **Rate Limiting & CORS**: Pre-configured with `slowapi` rate limiting and configurable CORS origins.

---

## Project Structure

```
zonaco_backend/
├── .env                          # Local environment variables
├── .env.example                  # Template configuration
├── .gitignore                    # Git ignore file
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Compose file (FastAPI + optional Redis)
├── README.md                     # Documentation
├── requirements.txt              # Production dependencies
├── ingest.py                     # Standalone knowledge base ingestion script
├── Chat Bot Intents with Links (1).xlsx # Excel FAQ source file
│
├── app/
│   ├── main.py                   # FastAPI initialization, CORS, middleware, routers
│   ├── config.py                 # Pydantic Settings
│   ├── state_machine.py          # Session state enum & transition rules
│   ├── dependencies.py           # Dependency injection providers
│   │
│   ├── routers/
│   │   ├── session.py            # /session/start, /session/{id}/history, /health
│   │   ├── categories.py         # /categories, /categories/{category}/questions, /privacy-policy
│   │   └── chat.py               # /chat/ask, /chat/satisfaction, /chat/more-questions, /chat/escalate, /chat/rate
│   │
│   ├── schemas/
│   │   ├── common.py             # Health & error models
│   │   ├── session.py            # Session & history models
│   │   ├── categories.py         # Category browsing models
│   │   └── chat.py               # Chat Q&A, rating, escalation models
│   │
│   ├── services/
│   │   ├── excel_parser.py       # Excel parser & regex link extraction
│   │   ├── session_store.py      # Thread-safe in-memory session manager
│   │   ├── vector_store.py       # ChromaDB persistent client wrapper
│   │   └── rag.py                # RAG retrieval & OpenRouter LLM service
│   │
│   └── utils/
│       ├── logger.py             # Structured logging & interaction tracker
│       └── exceptions.py         # Custom domain exceptions & global handler
│
├── data/
│   └── chroma_db/                # Local ChromaDB persistent storage
│
└── tests/
    └── test_api.py               # Automated endpoint & state machine tests
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- An OpenRouter API Key (free tier available at [openrouter.ai](https://openrouter.ai))

### 2. Clone & Virtual Environment
```bash
git clone <repo-url>
cd zonaco_backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your settings:
```ini
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/chroma_db
COLLECTION_NAME=zanaco_faq_collection
SIMILARITY_THRESHOLD=0.60
ALLOWED_ORIGINS=*
RATE_LIMIT=30/minute
FAQ_EXCEL_PATH=Chat Bot Intents with Links (1).xlsx
```

---

## Ingesting the FAQ Knowledge Base

To build or refresh the ChromaDB vector database from the Excel workbook:

```bash
python ingest.py
```

### Ingestion Options
- `--file <path>` : Specify a custom path to the FAQ workbook.
- `--reset` : Wipe the existing ChromaDB collection before re-indexing.

Example output:
```
======================================================================
 Zanaco FAQ Knowledge Base Ingestion Pipeline 
======================================================================
Target Excel file : Chat Bot Intents with Links (1).xlsx
Vector DB path    : ./data/chroma_db
Collection name   : zanaco_faq_collection
Embedding model   : all-MiniLM-L6-v2
Reset collection  : False
----------------------------------------------------------------------

Parsed 14 FAQ entries from sheet 'CARD SERVICES'
Parsed 8 FAQ entries from sheet 'EFTS'
Parsed 12 FAQ entries from sheet 'CREDIT'
...
Successfully parsed 85 FAQ documents across 8 categories.
Embedding and upserting 85 documents into ChromaDB...
Upserted 85/85 FAQ documents into ChromaDB.
======================================================================
 INGESTION SUMMARY
======================================================================
Status              : SUCCESS
Entries Processed   : 85
Entries Upserted    : 85
Total Vectors in DB : 85
Total Time Taken    : 4.12 seconds
======================================================================
```

---

## Running the API Server

### Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Running with Docker & Docker Compose

### 1. Build and Run via Docker Compose
```bash
docker-compose up --build -d
```

### 2. Run Ingestion Inside Docker
```bash
docker-compose exec chatbot-api python ingest.py
```

### 3. Check Logs
```bash
docker-compose logs -f chatbot-api
```

---

## API Endpoints Reference

| Method | Path | Description | Request Body |
|---|---|---|---|
| `GET` | `/health` | Liveness & readiness probe | None |
| `POST` | `/session/start` | Create session & return welcome menu | `{ "user_id"?: str, "metadata"?: dict }` |
| `GET` | `/categories` | List top-level FAQ categories | None |
| `GET` | `/categories/{category}/questions` | List browsable questions in a category | None |
| `POST` | `/chat/ask` | Submit question to RAG pipeline | `{ "session_id": str, "category"?: str, "question": str }` |
| `POST` | `/chat/satisfaction` | Record answer helpfulness | `{ "session_id": str, "satisfied": bool }` |
| `POST` | `/chat/more-questions` | Handle more questions decision | `{ "session_id": str, "more": bool }` |
| `POST` | `/chat/escalate` | Transfer to live human agent | `{ "session_id": str }` |
| `POST` | `/chat/rate` | Submit rating & feedback | `{ "session_id": str, "rating": int, "feedback_text"?: str }` |
| `GET` | `/privacy-policy` | Return Zanaco privacy policy | None |
| `GET` | `/session/{session_id}/history` | Return full conversation transcript | None |

---

## Session State Machine

```
      Start
        │
        ▼
   [WELCOME] ──► [MAIN_MENU]
                   │
         ┌─────────┼──────────────────┐
         │         │                  │
         ▼         ▼                  ▼
    [PRIVACY]  [LIVE_AGENT]    [CATEGORY_SELECT]
                           │          │
                           │          ▼
                           │   [QUESTION_SELECT]
                           │          │
                           │          ▼
                           │     [ANSWERED]
                           │          │
                           │          ▼
                           │  [SATISFACTION_CHECK]
                           │          │
                           │          ▼
                           │ [MORE_QUESTIONS_CHECK]
                           │    │           │
                     (more=YES) │           │ (more=NO)
                                │           │
                                ▼           ├───────────────┐
                       [CATEGORY_SELECT]    │ (if satisfied)│ (if unsatisfied)
                                            ▼               ▼
                                        [RATING]       [LIVE_AGENT]
                                            │               │
                                            ▼               ▼
                                         [END]            [END]
```

---

## Swapping to Redis Session Store (Multi-Instance Deployment)

The current implementation uses a thread-safe in-memory session repository (`app/services/session_store.py`). For horizontal scaling across multiple container instances:

1. Enable the Redis service in `docker-compose.yml`.
2. Install `redis` client: `pip install redis`.
3. Create `app/services/redis_session_store.py` implementing the same `SessionStore` interface (`create_session`, `get_session`, `transition_state`, `add_message`, `set_rating_and_feedback`) using Redis hashes or JSON strings with an expiration TTL (e.g. `redis.setex(session_id, 3600, json_data)`).
4. Update `app/dependencies.py` to inject `get_redis_session_store()` instead of `get_session_store()`.

---

## Running Tests

Execute the automated test suite with `pytest`:
```bash
pytest tests/ -v
```
