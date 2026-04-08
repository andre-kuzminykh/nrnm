"""RAG layer: chunking, OpenAI embeddings, Qdrant upsert/query.

FR-P5 (ingest) and FR-P7 (query) — see SPEC.md.

All functions are graceful-noop when ``QDRANT_URL`` is empty — they return
empty results instead of raising, so the bot still boots and the ИИ-платформа
UI works for domain management / model selection even before Qdrant is wired.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

import config

logger = logging.getLogger(__name__)


# ── Qdrant client (lazy) ─────────────────────────────────────────

_qdrant_client = None


def _get_qdrant():
    """Return a cached QdrantClient or None if not configured."""
    global _qdrant_client
    if not config.QDRANT_URL:
        return None
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to init Qdrant client: %s", exc)
        return None
    return _qdrant_client


def is_configured() -> bool:
    return bool(config.QDRANT_URL)


# ── Chunking ─────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Simple char-based sliding window chunker. FR-P5."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


# ── Embeddings ───────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """FR-P5: get embeddings for a list of chunks via OpenAI.

    Batches inputs to stay under OpenAI's `max_tokens_per_request` (300k).
    For ~125 tokens per 500-char chunk, 1000 chunks ≈ 125k tokens — safe.
    Empty chunks are skipped silently.
    """
    if not texts:
        return []
    BATCH_SIZE = 500
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = await client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
            all_vectors.extend(d.embedding for d in resp.data)
        return all_vectors
    except Exception as exc:  # noqa: BLE001
        logger.error("embed_texts failed: %s", exc)
        return []


# ── Qdrant operations ────────────────────────────────────────────

def _ensure_collection(collection: str) -> bool:
    """Create the collection if missing. Returns True on success."""
    q = _get_qdrant()
    if q is None:
        return False
    try:
        from qdrant_client.http.models import Distance, VectorParams
        existing = [c.name for c in q.get_collections().collections]
        if collection not in existing:
            q.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("ensure_collection %s failed: %s", collection, exc)
        return False


async def ingest_document(
    collection: str,
    doc_id: str,
    filename: str,
    text: str,
    message_id: int | None = None,
) -> int:
    """FR-P5: chunk → embed → upsert document into Qdrant. Returns chunk count.

    `message_id` (FR-P18) is the originating tg message id, stored in the chunk
    payload so RAG answers can render clickable [N] source references.
    """
    q = _get_qdrant()
    if q is None:
        return 0
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = await embed_texts(chunks)
    if len(vectors) != len(chunks):
        return 0
    if not _ensure_collection(collection):
        return 0
    try:
        from qdrant_client.http.models import PointStruct
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_idx": i,
                    "text": chunk,
                    "message_id": message_id,
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        q.upsert(collection_name=collection, points=points)
        return len(points)
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest_document %s failed: %s", collection, exc)
        return 0


async def delete_document_vectors(collection: str, doc_id: str) -> int:
    """Delete all chunks of a document from Qdrant by `doc_id` payload filter."""
    q = _get_qdrant()
    if q is None:
        return 0
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector
        q.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )),
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("delete_document_vectors failed: %s", exc)
        return 0


async def query_rag(collection: str, question: str, top_k: int = 5) -> list[dict]:
    """FR-P7: embed question → top-k Qdrant search → list of chunk dicts."""
    q = _get_qdrant()
    if q is None:
        return []
    vectors = await embed_texts([question])
    if not vectors:
        return []
    try:
        hits = q.search(
            collection_name=collection,
            query_vector=vectors[0],
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": h.payload.get("text", ""),
                "filename": h.payload.get("filename", ""),
                "message_id": h.payload.get("message_id"),
                "score": float(h.score),
            }
            for h in hits
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("query_rag failed: %s", exc)
        return []
