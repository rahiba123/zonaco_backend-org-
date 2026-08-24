"""Standalone idempotent ingestion script to parse Excel FAQ knowledge base into ChromaDB."""

import argparse
import os
import sys
import time
from typing import Dict

# Add project root to sys.path so app modules can be imported directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.services.excel_parser import ExcelFAQParser
from app.services.vector_store import VectorStoreService
from app.utils.logger import logger


def run_ingestion(excel_path: str = None, reset_collection: bool = False) -> None:
    """Parse Excel FAQ sheets and upsert into ChromaDB vector store."""
    settings = get_settings()
    target_file = excel_path or settings.FAQ_EXCEL_PATH

    print("=" * 70)
    print(" Zanaco FAQ Knowledge Base Ingestion Pipeline ")
    print("=" * 70)
    print(f"Target Excel file : {target_file}")
    print(f"Vector DB path    : {settings.VECTOR_DB_PATH}")
    print(f"Collection name   : {settings.COLLECTION_NAME}")
    print(f"Embedding model   : {settings.EMBEDDING_MODEL_NAME}")
    print(f"Reset collection  : {reset_collection}")
    print("-" * 70)

    start_time = time.perf_counter()

    # Step 1: Parse Excel file
    parser = ExcelFAQParser(file_path=target_file)
    try:
        entries = parser.parse_all()
    except Exception as exc:
        logger.error(f"Ingestion failed during Excel parsing: {exc}")
        print(f"\n[ERROR] Failed to parse Excel file: {exc}")
        sys.exit(1)

    if not entries:
        print("\n[WARNING] No valid FAQ entries found in workbook. Ingestion stopped.")
        return

    # Category breakdown stats
    category_counts: Dict[str, int] = {}
    total_links = 0
    for entry in entries:
        category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
        total_links += len(entry.links)

    print(f"\nSuccessfully parsed {len(entries)} FAQ documents across {len(category_counts)} categories:")
    for cat, count in category_counts.items():
        print(f"  • {cat:<25} : {count:>3} entries")
    print(f"  • Extracted reference URLs: {total_links} total links found in details")

    # Step 2: Initialize Vector Store
    print("\nInitializing ChromaDB persistent vector store...")
    vector_store = VectorStoreService()

    if reset_collection:
        print(f"Resetting existing collection '{settings.COLLECTION_NAME}'...")
        try:
            vector_store.client.delete_collection(settings.COLLECTION_NAME)
            # Recreate empty collection
            vector_store.collection = vector_store.client.create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=vector_store.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            print("Collection reset complete.")
        except Exception as exc:
            logger.warning(f"Could not delete collection (may not exist yet): {exc}")

    # Step 3: Batch upsert into ChromaDB
    print(f"Embedding and upserting {len(entries)} documents into ChromaDB...")
    upserted_count = vector_store.add_entries(entries, batch_size=50)

    elapsed = time.perf_counter() - start_time
    total_docs_in_db = vector_store.count()

    print("\n" + "=" * 70)
    print(" INGESTION SUMMARY")
    print("=" * 70)
    print(f"Status              : SUCCESS")
    print(f"Entries Processed   : {len(entries)}")
    print(f"Entries Upserted    : {upserted_count}")
    print(f"Total Vectors in DB : {total_docs_in_db}")
    print(f"Total Time Taken    : {elapsed:.2f} seconds")
    print("=" * 70)
    print("\nYou can now start the FastAPI server with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n")


def main():
    """CLI entry point for ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest Zanaco FAQ Excel workbook into ChromaDB vector database."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to FAQ Excel workbook (default: configured FAQ_EXCEL_PATH in .env)"
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        default=False,
        help="Delete existing ChromaDB collection before indexing (useful for clean rebuilds)"
    )

    args = parser.parse_args()
    run_ingestion(excel_path=args.file, reset_collection=args.reset)


if __name__ == "__main__":
    main()
