"""ChromaDB persistent vector store wrapper with category metadata filtering."""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from app.config import get_settings
from app.services.excel_parser import FAQEntry
from app.utils.exceptions import VectorStoreException
from app.utils.logger import logger


@dataclass
class SearchResult:
    """Standardized representation of a vector retrieval match."""
    doc_id: str
    text: str
    category: str
    faq_intent: str
    question: str
    response: str
    topic_type: str
    details: str
    links: List[str]
    similarity_score: float
    distance: float


class VectorStoreService:
    """Persistent ChromaDB client wrapper for embedding, indexing, and category-filtered querying."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_fn = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize local ChromaDB persistent storage and SentenceTransformer embedding function."""
        try:
            db_path = os.path.abspath(self.settings.VECTOR_DB_PATH)
            os.makedirs(db_path, exist_ok=True)
            logger.info(f"Initializing ChromaDB persistent client at: {db_path}")

            # Instantiate persistent client
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Local CPU SentenceTransformers embedding function
            logger.info(f"Loading embedding model: {self.settings.EMBEDDING_MODEL_NAME}")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.EMBEDDING_MODEL_NAME
            )

            # Create or get collection with cosine space
            self.collection = self.client.get_or_create_collection(
                name=self.settings.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"ChromaDB collection '{self.settings.COLLECTION_NAME}' ready. "
                f"Current document count: {self.collection.count()}"
            )
        except Exception as exc:
            logger.exception(f"Failed to initialize ChromaDB vector store: {exc}")
            raise VectorStoreException(f"Failed to initialize vector store: {exc}")

    def add_entries(self, entries: List[FAQEntry], batch_size: int = 100) -> int:
        """Upsert a batch of FAQ entries into ChromaDB idempotently."""
        if not self.collection:
            self._initialize()

        if not entries:
            return 0

        total_upserted = 0
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            ids = [entry.doc_id for entry in batch]
            documents = [entry.formatted_text for entry in batch]
            metadatas = [entry.to_metadata() for entry in batch]

            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                total_upserted += len(batch)
                logger.info(f"Upserted {total_upserted}/{len(entries)} FAQ documents into ChromaDB.")
            except Exception as exc:
                logger.exception(f"Error during batch upsert into ChromaDB: {exc}")
                raise VectorStoreException(f"Batch upsert failed: {exc}")

        return total_upserted

    def query(
        self,
        query_text: str,
        category: Optional[str] = None,
        n_results: Optional[int] = None
    ) -> List[SearchResult]:
        """Query vector database for similar FAQ entries with optional category filtering."""
        if not self.collection:
            self._initialize()

        k = n_results or self.settings.TOP_K
        where_filter: Optional[Dict[str, Any]] = None
        if category and category.strip():
            where_filter = {"category": category.strip().upper()}

        try:
            logger.info(
                f"Querying vector store: \"{query_text}\" | Category Filter: {where_filter} | Top-K: {k}"
            )
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as exc:
            logger.exception(f"Vector search failed: {exc}")
            raise VectorStoreException(f"Vector query failed: {exc}")

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            logger.warning("Vector search returned 0 matches.")
            return search_results

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance d in [0, 2]; similarity = max(0, 1 - d)
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))

            raw_links = meta.get("links", "[]")
            try:
                links_list = json.loads(raw_links) if isinstance(raw_links, str) else []
            except Exception:
                links_list = []

            search_results.append(
                SearchResult(
                    doc_id=doc_id,
                    text=doc_text,
                    category=meta.get("category", ""),
                    faq_intent=meta.get("faq_intent", ""),
                    question=meta.get("question", ""),
                    response=meta.get("response", ""),
                    topic_type=meta.get("topic_type", ""),
                    details=meta.get("details", ""),
                    links=links_list,
                    similarity_score=round(similarity, 4),
                    distance=round(float(dist), 4)
                )
            )

        # Sort by highest similarity score first
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results

    def count(self) -> int:
        """Return total indexed document count in the collection."""
        if not self.collection:
            self._initialize()
        return self.collection.count()

    def is_healthy(self) -> bool:
        """Check if ChromaDB client and collection are active."""
        try:
            return self.collection is not None and self.collection.count() >= 0
        except Exception:
            return False


# Singleton vector store service instance
vector_store_service = VectorStoreService()


def get_vector_store() -> VectorStoreService:
    """Dependency injection provider for VectorStoreService."""
    return vector_store_service
