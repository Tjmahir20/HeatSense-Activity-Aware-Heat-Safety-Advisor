"""
HeatSense retrieval-augmented generation (RAG) pipeline.

Implements hybrid lexical + dense retrieval over ``data/niosh_osha.txt`` using
patterns from Module 4 (chunking, OpenAI embeddings, ChromaDB, BM25, score
fusion). The public ``retrieve`` function is the only entry point used by
``pipeline.agent``.

v2 change: replaced sliding-window chunker with a block-aware splitter that
respects ``--- BLOCK: ... ---`` section markers in the knowledge base. Each
block is kept intact as a single retrievable unit so thresholds, work/rest
ratios, and hydration targets for a given work intensity are never split across
separate chunks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
_DATA_FILE = _ROOT / "data" / "niosh_osha.txt"
_DB_PATH = str(_ROOT / "data" / "chroma_db")
_COLLECTION_NAME = "heatsense_kb"
_EMBED_MODEL = "text-embedding-3-small"

# Block separator used in niosh_osha.txt. A line that starts with this prefix
# signals the beginning of a new self-contained semantic section.
_BLOCK_SEPARATOR = "--- BLOCK:"


def _chunk_text(text: str, chunk_size: int = 300, stride: int = 100) -> List[str]:
    """
    Split knowledge-base text into semantic blocks.

    If the text contains ``--- BLOCK: ...`` markers, each marked section is
    returned as one chunk (the separator line is kept as the first line of its
    chunk so the block title is embedded in the retrieval unit).

    Falls back to the original sliding-window approach when no markers are
    found, so the function is backward-compatible with older knowledge files.

    Args:
        text:       Full document body to segment.
        chunk_size: Used only in sliding-window fallback mode.
        stride:     Used only in sliding-window fallback mode.

    Returns:
        Non-empty stripped chunk strings in document order.
    """
    if not text:
        return []

    # Check whether the new block-structured format is present.
    if _BLOCK_SEPARATOR in text:
        return _split_by_blocks(text)

    # Legacy fallback: sliding window (kept for backward compatibility).
    chunks: List[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += stride
    return chunks


def _split_by_blocks(text: str) -> List[str]:
    """
    Split text into one chunk per ``--- BLOCK: ... ---`` section.

    The separator line is included as the opening line of each chunk so that
    the block title (e.g. "Heavy Work Thresholds and Guidance") is part of
    the embedding and BM25 token set, improving retrieval precision.

    Args:
        text: Full knowledge-base text containing block markers.

    Returns:
        List of block strings, each starting with its separator line.
    """
    lines = text.splitlines()
    blocks: List[str] = []
    current_lines: List[str] = []

    for line in lines:
        if line.strip().startswith(_BLOCK_SEPARATOR):
            # Flush the previous block (if any) before starting a new one.
            if current_lines:
                block_text = "\n".join(current_lines).strip()
                if block_text:
                    blocks.append(block_text)
            # Start the new block with the separator line as its title.
            current_lines = [line.strip()]
        else:
            current_lines.append(line)

    # Flush the final block after the loop ends.
    if current_lines:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(block_text)

    return [b for b in blocks if b]


def _get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Request dense embeddings from OpenAI for one or more text segments.

    Args:
        client: Authenticated OpenAI SDK client.
        texts:  Non-empty list of strings to embed.

    Returns:
        List of embedding vectors aligned with ``texts`` order.
    """
    if not texts:
        return []
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), 2048):
        batch = texts[start : start + 2048]
        resp = client.embeddings.create(model=_EMBED_MODEL, input=batch)
        ordered = sorted(resp.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
    return embeddings


def _get_or_build_collection(client: OpenAI) -> chromadb.Collection:
    """
    Open or create the persistent Chroma collection backing the knowledge base.

    On first run (empty collection), reads ``niosh_osha.txt``, splits it into
    semantic blocks, embeds all blocks, and upserts into Chroma.

    IMPORTANT: Delete ``data/chroma_db/`` whenever ``niosh_osha.txt`` is
    updated. The collection does not auto-detect file changes.

    Args:
        client: OpenAI client used only when the collection must be built.

    Returns:
        A Chroma ``Collection`` ready for ``get`` / ``query`` operations.
    """
    chroma = chromadb.PersistentClient(path=_DB_PATH)
    collection = chroma.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection

    text = _DATA_FILE.read_text(encoding="utf-8")
    chunks = _chunk_text(text)
    embeds = _get_embeddings(client, chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metas = [{"source": "niosh_osha.txt", "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, embeddings=embeds, metadatas=metas, ids=ids)
    return collection


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 using lowercase whitespace split."""
    return text.lower().split()


def _bm25_search(
    documents: List[str],
    query: str,
    k: int,
) -> List[Tuple[int, float, str]]:
    """Rank all documents with BM25 and return the top ``k`` hits."""
    index = BM25Okapi([_tokenize_for_bm25(d) for d in documents])
    scores = index.get_scores(_tokenize_for_bm25(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(int(idx), float(s), documents[idx]) for idx, s in ranked]


def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores into [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _hybrid_fusion(
    bm25_results: List[Tuple[int, float, str]],
    vec_results: List[Tuple[str, float, str]],
    alpha: float = 0.5,
) -> List[str]:
    """
    Fuse BM25 and dense retrieval lists into a single ranked text list.

    Uses the full block text as the deduplication key (not an 80-char prefix)
    because blocks are large enough that short prefixes may collide between
    adjacent sections that share similar opening lines.
    """
    bm25_norm = _normalize([r[1] for r in bm25_results])
    vec_norm = _normalize([r[1] for r in vec_results])

    merged: Dict[str, Dict] = {}

    for i, (_, __, text) in enumerate(bm25_results):
        # Use the full text as key; blocks are distinct enough.
        key = text
        merged[key] = {"text": text, "bm25": bm25_norm[i], "vec": None}

    for i, (_, __, text) in enumerate(vec_results):
        key = text
        if key in merged:
            merged[key]["vec"] = vec_norm[i]
        else:
            merged[key] = {"text": text, "bm25": None, "vec": vec_norm[i]}

    fused: List[Tuple[float, str]] = []
    for entry in merged.values():
        b, v = entry["bm25"], entry["vec"]
        if b is not None and v is not None:
            score = alpha * v + (1.0 - alpha) * b
        elif v is not None:
            score = v
        else:
            score = b or 0.0
        fused.append((score, entry["text"]))

    fused.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in fused]


_collection: chromadb.Collection | None = None
_openai_client: OpenAI | None = None


def _ensure_ready() -> None:
    """Lazily construct the OpenAI client and Chroma collection if needed."""
    global _collection, _openai_client
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment.")
        _openai_client = OpenAI(api_key=api_key)
        _collection = _get_or_build_collection(_openai_client)


def retrieve(query: str, k: int = 4) -> str:
    """
    Return the top ``k`` knowledge-base blocks for ``query`` as one string.

    With the block-structured knowledge base, ``k=4`` returns up to 4 complete
    semantic sections. For a typical shift-plan query (one work intensity type),
    the correct threshold block plus hydration, risk classification, and
    supervisor guidance blocks will rank in the top 4.

    Args:
        query: Natural-language search string (from the LLM tool call).
        k:     Number of blocks to keep after fusion.

    Returns:
        Newline-separated context string, or a short message when the KB is empty.
    """
    _ensure_ready()
    assert _collection is not None
    assert _openai_client is not None

    all_docs = _collection.get(include=["documents"])
    documents: List[str] = all_docs["documents"] or []
    if not documents:
        return "No knowledge base content found."

    bm25_results = _bm25_search(documents, query, k * 2)

    query_embedding = _get_embeddings(_openai_client, [query])[0]
    vec_raw = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k * 2, len(documents)),
        include=["documents", "distances"],
    )
    vec_ids = vec_raw["ids"][0] or []
    vec_docs = vec_raw["documents"][0] or []
    vec_dists = vec_raw["distances"][0] or []
    vec_results = [
        (vec_ids[i], 1.0 - vec_dists[i], vec_docs[i])
        for i in range(len(vec_ids))
    ]

    top_texts = _hybrid_fusion(bm25_results, vec_results)[:k]
    return "\n\n".join(top_texts)
