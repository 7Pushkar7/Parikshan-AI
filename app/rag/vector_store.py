"""Long-term knowledge store for RAG (Retrieval-Augmented Generation), built on ChromaDB.

Every paper / trial / safety record a tool fetches is:
  1. split into chunks,
  2. turned into an embedding (a list of numbers that captures meaning),
  3. saved in ChromaDB on disk (data/chroma) with metadata (source id, url, session).
When the report writer runs, it embeds the question and pulls back the most similar chunks,
so the report is written ONLY from evidence that was actually fetched.
"""
from __future__ import annotations

import hashlib
import logging
import threading

import chromadb

from app.config import get_settings

log = logging.getLogger(__name__)
COLLECTION = "parikshan_evidence"

_client = None
_collection = None
_lock = threading.Lock()
_embedding_override = None  # tests inject a fake embedding function


def set_embedding_override(fn) -> None:
    global _embedding_override, _collection, _client
    _embedding_override = fn
    _collection = None
    _client = None


def _embedding_function():
    if _embedding_override is not None:
        return _embedding_override
    s = get_settings()
    if s.embeddings == "gemini":
        import os

        from chromadb.utils.embedding_functions import GoogleGenaiEmbeddingFunction
        os.environ.setdefault("GEMINI_API_KEY", s.gemini_api_key)  # reads key from env var
        return GoogleGenaiEmbeddingFunction(model_name=s.gemini_embedding_model)
    # Default: all-MiniLM-L6-v2 running locally via ONNX. Free, no API calls.
    # Downloads ~80 MB once on first use.
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def get_collection():
    global _client, _collection
    with _lock:
        if _collection is None:
            s = get_settings()
            _client = chromadb.PersistentClient(path=str(s.chroma_dir))
            _collection = _client.get_or_create_collection(
                COLLECTION, embedding_function=_embedding_function(),
                metadata={"hnsw:space": "cosine"})
        return _collection


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to cut at a sentence end
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def add_document(thread_id: str, source_id: str, source_type: str, title: str, url: str,
                 text: str) -> int:
    """Upsert one source (paper, trial, FAERS summary...) for a conversation thread."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    col = get_collection()
    ids = [hashlib.md5(f"{thread_id}|{source_id}|{i}".encode()).hexdigest() for i in range(len(chunks))]
    metas = [{"thread_id": thread_id, "source_id": source_id, "source_type": source_type,
              "title": title[:300], "url": url, "chunk": i} for i in range(len(chunks))]
    col.upsert(ids=ids, documents=chunks, metadatas=metas)
    return len(chunks)


def search(thread_id: str, query: str, k: int | None = None,
           source_ids: list[str] | None = None) -> list[dict]:
    col = get_collection()
    k = k or get_settings().rag_top_k
    where: dict = {"thread_id": thread_id}
    if source_ids:
        where = {"$and": [{"thread_id": thread_id}, {"source_id": {"$in": source_ids}}]}
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, n), where=where)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "score": round(1 - dist, 3), **meta})
    return hits


def get_source(thread_id: str, source_id: str) -> list[dict]:
    col = get_collection()
    res = col.get(where={"$and": [{"thread_id": thread_id}, {"source_id": source_id}]})
    rows = sorted(zip(res["documents"], res["metadatas"]), key=lambda r: r[1].get("chunk", 0))
    return [{"text": d, **m} for d, m in rows]


def list_sources(thread_id: str) -> dict[str, dict]:
    col = get_collection()
    res = col.get(where={"thread_id": thread_id}, include=["metadatas"])
    out: dict[str, dict] = {}
    for m in res["metadatas"]:
        out.setdefault(m["source_id"], {"source_id": m["source_id"], "source_type": m["source_type"],
                                        "title": m["title"], "url": m["url"]})
    return out


def delete_thread(thread_id: str) -> None:
    get_collection().delete(where={"thread_id": thread_id})
