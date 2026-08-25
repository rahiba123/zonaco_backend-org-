"""ChromaDB-backed vector store for user-uploaded documents, scoped per session.

Design choice: rather than creating a new ChromaDB *collection* per session
(which doesn't scale — thousands of idle collections pile up), this uses ONE
shared collection (`USER_DOCS_COLLECTION_NAME`) and filters every read/write
by `session_id` metadata. This mirrors the existing category-filter pattern
in `vector_store.py`, just filtering on session instead of category.

Isolation guarantee: `query()` and `delete_session_documents()` always scope
by `session_id`, so one user's uploaded document is never visible to another
session's queries.
"""

from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.config import get_settings
from app.services.document_parser import DocumentChunk
from app.utils.exceptions import VectorStoreException
from app.utils.logger import logger


@dataclass
class DocumentSearchResult:
    """Standardized representation of a retrieved user-document chunk."""
    doc_id: str
    text: str
    source_filename: str
    chunk_index: int
    similarity_score: float
    distance: float


class DocumentVectorStoreService:
    """Persistent ChromaDB wrapper for session-scoped user-uploaded document chunks."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_fn = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB persistent client and embedding function.

        Reuses the same persistent client path as the FAQ vector store
        (VECTOR_DB_PATH) but a DIFFERENT collection name, so uploaded
        documents never mix with the bank's fixed FAQ knowledge base.
        """
        try:
            import os
            db_path = os.path.abspath(self.settings.VECTOR_DB_PATH)
            os.makedirs(db_path, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Same embedding model as the FAQ store, for consistency and so
            # both stores can share the model already loaded in memory.
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.EMBEDDING_MODEL_NAME
            )

            self.collection = self.client.get_or_create_collection(
                name=self.settings.USER_DOCS_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"User-document ChromaDB collection '{self.settings.USER_DOCS_COLLECTION_NAME}' ready. "
                f"Current chunk count: {self.collection.count()}"
            )
        except Exception as exc:
            logger.exception(f"Failed to initialize user-document vector store: {exc}")
            raise VectorStoreException(f"Failed to initialize user-document vector store: {exc}")

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        """Upsert a batch of document chunks. Idempotent via deterministic doc_id."""
        if not self.collection:
            self._initialize()
        if not chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.doc_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.to_metadata() for c in batch]

            try:
                self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_upserted += len(batch)
            except Exception as exc:
                logger.exception(f"Error upserting user-document chunks: {exc}")
                raise VectorStoreException(f"Batch upsert of user document failed: {exc}")

        logger.info(f"Upserted {total_upserted} chunk(s) for session {chunks[0].session_id}.")
        return total_upserted

    def query(
        self,
        session_id: str,
        query_text: str,
        n_results: Optional[int] = None
    ) -> List[DocumentSearchResult]:
        """Query only the calling session's uploaded document chunks."""
        if not self.collection:
            self._initialize()

        k = n_results or self.settings.USER_DOC_TOP_K
        where_filter = {"session_id": session_id}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as exc:
            logger.exception(f"User-document vector query failed: {exc}")
            raise VectorStoreException(f"User-document query failed: {exc}")

        search_results: List[DocumentSearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            search_results.append(
                DocumentSearchResult(
                    doc_id=doc_id,
                    text=doc_text,
                    source_filename=meta.get("source_filename", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    similarity_score=round(similarity, 4),
                    distance=round(float(dist), 4)
                )
            )

        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results

    def has_documents(self, session_id: str) -> bool:
        """Check whether a session has any uploaded document chunks indexed."""
        if not self.collection:
            self._initialize()
        try:
            existing = self.collection.get(where={"session_id": session_id}, limit=1)
            return bool(existing and existing.get("ids"))
        except Exception as exc:
            logger.warning(f"has_documents check failed for session {session_id}: {exc}")
            return False

    def delete_session_documents(self, session_id: str) -> int:
        """Delete all chunks belonging to a session (called on session end/cleanup)."""
        if not self.collection:
            self._initialize()
        try:
            existing = self.collection.get(where={"session_id": session_id})
            ids = existing.get("ids", []) if existing else []
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} document chunk(s) for session {session_id}.")
            return len(ids)
        except Exception as exc:
            logger.exception(f"Failed to delete documents for session {session_id}: {exc}")
            raise VectorStoreException(f"Failed to delete session documents: {exc}")


# Singleton instance, following the same pattern as vector_store.py
document_vector_store_service = DocumentVectorStoreService()


def get_document_vector_store() -> DocumentVectorStoreService:
    """Dependency injection provider for DocumentVectorStoreService."""
    return document_vector_store_service