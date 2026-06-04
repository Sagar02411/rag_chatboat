"""
vector_store.py
---------------
ChromaDB wrapper — persistent local vector store.

IMPORTANT: We create a fresh PersistentClient on every public call.
Module-level singletons don't work reliably with Streamlit because the script
reruns on every user interaction and Python may garbage-collect the underlying
HTTP connection, causing "client has been closed" errors.

Creating PersistentClient is cheap (it just opens a local SQLite file).
"""

import logging
from sentence_transformers import SentenceTransformer
from typing import Optional

import chromadb

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)

# Only the embedder is cached — it's expensive (~seconds) to reload.
_embedder: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_collection():
    """Always return a fresh client + collection. Safe to call on every rerun."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}'…")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder


# Keep get_collection() as an alias so chunker.py import doesn't break
def get_collection():
    return _new_collection()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Semantic search over stored chunks."""
    col = _new_collection()
    count = col.count()

    if count == 0:
        logger.warning("ChromaDB collection is empty — no documents ingested yet.")
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode(question, convert_to_numpy=True).tolist()

    results = col.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "ts_label": meta.get("ts_label", ""),
            "distance": round(dist, 4),
        })

    logger.info(f"Retrieved {len(hits)} chunks for query: '{question[:60]}'")
    return hits


def list_sources() -> list[str]:
    """Return a sorted list of all unique source filenames in the DB."""
    col = _new_collection()
    if col.count() == 0:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return sorted({m.get("source", "") for m in all_meta if m.get("source")})


def delete_source(source_name: str) -> int:
    """Remove all chunks belonging to a specific source file."""
    col = _new_collection()
    result = col.get(where={"source": source_name}, include=["metadatas"])
    ids_to_delete = result.get("ids", [])
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_name}'")
    return len(ids_to_delete)


def chunk_count() -> int:
    """Return total number of chunks stored."""
    try:
        return _new_collection().count()
    except Exception:
        return 0
