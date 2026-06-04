"""
chunker.py
----------
Splits a transcript into overlapping chunks, embeds them with
sentence-transformers, and stores them in ChromaDB.
"""

import hashlib
import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)
from retrieval.vector_store import get_collection

logger = logging.getLogger(__name__)

# Module-level cache for the embedding model
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformer model."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}'…")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder


def _make_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Generate a stable, unique ID for a chunk."""
    raw = f"{source}::{chunk_index}::{text[:80]}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_transcript(
    full_text: str,
    source_name: str,
    segments: list[dict],
) -> int:
    """
    Chunk, embed, and store a transcript in ChromaDB.

    Parameters
    ----------
    full_text : str
        The complete transcript text.
    source_name : str
        The filename of the original media (used as metadata).
    segments : list[dict]
        Whisper segments with ``start``, ``end``, ``text`` keys.
        Used to attach approximate timestamps to each chunk.

    Returns
    -------
    int
        Number of chunks stored.
    """
    embedder = _get_embedder()

    # ── 1. Split into chunks ──────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[str] = splitter.split_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks for '{source_name}'")

    if not chunks:
        logger.warning(f"No chunks produced for '{source_name}' — skipping.")
        return 0

    # ── 2. Build timestamp metadata per chunk ────────────────────────────
    #  Simple approach: map chunk character offset → segment timestamp
    segment_starts = _build_char_offset_map(segments)

    ids, embeddings, metadatas, documents = [], [], [], []
    char_offset = 0

    for idx, chunk in enumerate(chunks):
        # Approximate start time from character position
        ts_start, ts_end = _approx_timestamps(char_offset, segment_starts)

        chunk_id = _make_chunk_id(source_name, idx, chunk)
        embedding = embedder.encode(chunk, convert_to_numpy=True).tolist()

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append(
            {
                "source":    source_name,
                "chunk_idx": idx,
                "ts_start":  round(ts_start, 1),
                "ts_end":    round(ts_end, 1),
                "ts_label":  _format_ts(ts_start) + " – " + _format_ts(ts_end),
            }
        )
        char_offset += len(chunk) - CHUNK_OVERLAP

    # ── 3. Upsert into ChromaDB (fresh client per call — no stale connections) ──
    col = get_collection()
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks from '{source_name}' in ChromaDB.")
    return len(chunks)


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_char_offset_map(segments: list[dict]) -> list[tuple[int, float, float]]:
    """
    Returns list of (cumulative_char_offset, seg_start_sec, seg_end_sec).
    """
    mapping: list[tuple[int, float, float]] = []
    offset = 0
    for seg in segments:
        mapping.append((offset, seg["start"], seg["end"]))
        offset += len(seg["text"]) + 1  # +1 for space between segments
    return mapping


def _approx_timestamps(
    char_offset: int,
    seg_map: list[tuple[int, float, float]],
) -> tuple[float, float]:
    """Return (start_sec, end_sec) for the given character offset."""
    if not seg_map:
        return 0.0, 0.0
    best_start, best_end = seg_map[0][1], seg_map[0][2]
    for (off, start, end) in seg_map:
        if off <= char_offset:
            best_start = start
            best_end = end
        else:
            break
    return best_start, best_end


def _format_ts(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
