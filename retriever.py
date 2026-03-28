"""
retriever.py — hybrid semantic + keyword retrieval with three-tier context

Primary: OpenAI embedding cosine similarity (requires OPENAI_API_KEY)
Fallback: TF-IDF cosine similarity (no dependencies)

Tier 0: Signatures (~8 tokens each)     — always sent, all chunks
Tier 1: Docstrings (~25 tokens each)    — sent for relevant chunks
Tier 2: Full raw source (~130 tokens)   — pulled on demand via tool call

Intent classification:
  EDIT  → target identified, pulled to Tier 2 immediately
  READ  → model uses Tier 0+1 and requests Tier 2 if needed
"""

import os
import re
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from chunker import Chunk, _tokens, LIGHTWEIGHT_DOCSTRING_LANGUAGES


# ── Embedding helpers ──────────────────────────────────────────────────────

EMBED_DIM = 512
EMBED_MODEL = "text-embedding-3-small"

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _chunk_embed_text(chunk: Chunk) -> str:
    """Build the text representation used for embedding a chunk."""
    parts = [chunk.name, chunk.signature]
    if chunk.docstring:
        parts.append(chunk.docstring)
    if chunk.language not in LIGHTWEIGHT_DOCSTRING_LANGUAGES:
        parts.append(chunk.raw[:500])
    return '\n'.join(parts)


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """
    Embed a batch of texts using OpenAI text-embedding-3-small.
    Returns a list of float vectors (dimension=EMBED_DIM).
    Raises ImportError if openai package is missing, or returns [] on API failure.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return []
    client = OpenAI(api_key=key)
    all_embeddings = []

    for batch_start in range(0, len(texts), 96):
        batch = texts[batch_start:batch_start + 96]
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
                dimensions=EMBED_DIM,
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        except Exception as e:
            print(f"[ctxzip] Embedding batch failed: {e}")
            all_embeddings.extend([[] for _ in batch])

    return all_embeddings


def embed_query(text: str, api_key: str | None = None) -> list[float]:
    """Embed a single query string. Returns [] on failure."""
    result = embed_texts([text], api_key)
    return result[0] if result and result[0] else []


def _cosine_similarity_batch(query_vec: list[float], matrix: list[list[float]]) -> list[float]:
    """Compute cosine similarity between query and each row in matrix."""
    if HAS_NUMPY and matrix:
        q = np.array(query_vec, dtype=np.float32)
        m = np.array(matrix, dtype=np.float32)
        dots = m @ q
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        denom = np.clip(m_norms * q_norm, 1e-10, None)
        return (dots / denom).tolist()

    # Pure Python fallback
    scores = []
    q_norm = sum(x * x for x in query_vec) ** 0.5
    for row in matrix:
        dot = sum(a * b for a, b in zip(query_vec, row))
        r_norm = sum(x * x for x in row) ** 0.5
        denom = q_norm * r_norm
        scores.append(dot / denom if denom > 1e-10 else 0.0)
    return scores


# ── TF-IDF fallback ───────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Tokenize text for keyword matching. Keeps 2+ char words."""
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) >= 2]


class TFIDFScorer:
    """Lightweight TF-IDF scorer built from chunk text representations."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.N = len(chunks)
        self.df: Counter = Counter()
        self._doc_vecs: list[dict[str, float]] = []

        doc_token_sets = []
        for c in chunks:
            text = f"{c.name} {c.signature} {c.docstring or ''} {c.raw[:400]}"
            tokens = _tokenize(text)
            token_set = set(tokens)
            doc_token_sets.append((tokens, token_set))
            for t in token_set:
                self.df[t] += 1

        for tokens, token_set in doc_token_sets:
            tf = Counter(tokens)
            vec: dict[str, float] = {}
            for t in token_set:
                idf = math.log((self.N + 1) / (self.df[t] + 1)) + 1
                vec[t] = (1 + math.log(tf[t] + 1)) * idf
            self._doc_vecs.append(vec)

    def score(self, query: str) -> list[float]:
        """Score all chunks against a query. Returns list of scores."""
        tokens = _tokenize(query)
        tf = Counter(tokens)
        q_vec: dict[str, float] = {}
        for t in set(tokens):
            idf = math.log((self.N + 1) / (self.df.get(t, 0) + 1)) + 1
            q_vec[t] = (1 + math.log(tf[t] + 1)) * idf

        q_norm = sum(v ** 2 for v in q_vec.values()) ** 0.5
        scores = []
        for doc_vec in self._doc_vecs:
            all_keys = set(q_vec) | set(doc_vec)
            dot = sum(q_vec.get(t, 0) * doc_vec.get(t, 0) for t in all_keys)
            d_norm = sum(v ** 2 for v in doc_vec.values()) ** 0.5
            denom = q_norm * d_norm
            scores.append(dot / denom if denom > 1e-10 else 0.0)
        return scores


# ── Name-matching boost ───────────────────────────────────────────────────

def _name_boost(query: str, chunk: Chunk) -> float:
    """Strong boost when the query directly mentions a function/class name."""
    q_lower = query.lower()
    name_lower = (chunk.name or '').lower()
    if not name_lower:
        return 0.0
    if name_lower in q_lower:
        return 0.5
    q_words = set(_tokenize(query))
    if name_lower in q_words:
        return 0.4
    for qw in q_words:
        if len(qw) >= 4 and (qw in name_lower or name_lower in qw):
            return 0.2
    return 0.0


# ── Intent ────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    EDIT = "edit"
    READ = "read"

EDIT_VERBS = {
    'fix', 'edit', 'change', 'update', 'rewrite', 'refactor', 'modify', 'add', 'remove',
    'delete', 'rename', 'replace', 'debug', 'correct', 'adjust', 'implement', 'insert',
    'patch', 'improve', 'optimize', 'move', 'extract', 'split', 'merge', 'convert', 'migrate',
    'create', 'write', 'build', 'make',
}
READ_VERBS = {
    'explain', 'understand', 'describe', 'show', 'why', 'does', 'where',
    'when', 'trace', 'walk', 'tell', 'summarize', 'find', 'list',
}

def classify_intent(query: str) -> Intent:
    words = set(_tokenize(query))
    first = query.strip().split()[0].lower() if query.strip() else ''

    edit_score = len(words & EDIT_VERBS)
    read_score = len(words & READ_VERBS)

    # "how" alone is read, but "how do I fix X" is edit
    if 'how' in words and words & EDIT_VERBS:
        edit_score += 2
    elif 'how' in words or 'what' in words:
        read_score += 2

    if first in EDIT_VERBS:
        edit_score += 3
    elif first in READ_VERBS:
        read_score += 3

    return Intent.EDIT if edit_score > read_score else Intent.READ


# ── Tier representations ───────────────────────────────────────────────────

def tier0_repr(chunk: Chunk) -> str:
    return f"{chunk.id}  {chunk.signature}  [{chunk.file}:{chunk.start_line}]"

def tier1_repr(chunk: Chunk) -> str:
    doc = chunk.docstring or "(no description)"
    doc_lines = [l for l in doc.splitlines() if l.strip()][:3]
    doc_short = ' '.join(doc_lines)[:200]
    return f"{chunk.id}  {chunk.signature}\n  {doc_short}"

def tier2_repr(chunk: Chunk) -> str:
    return chunk.raw


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    intent: Intent
    query: str
    search_mode: str = "tfidf"

    tier0_chunks: list[Chunk] = field(default_factory=list)
    tier1_chunks: list[Chunk] = field(default_factory=list)
    tier2_chunks: list[Chunk] = field(default_factory=list)
    all_chunks: list[Chunk] = field(default_factory=list)

    def tier0_tokens(self) -> int:
        return sum(_tokens(tier0_repr(c)) for c in self.tier0_chunks)

    def tier1_tokens(self) -> int:
        return sum(_tokens(tier1_repr(c)) for c in self.tier1_chunks)

    def tier2_tokens(self) -> int:
        return sum(_tokens(tier2_repr(c)) for c in self.tier2_chunks)

    def total_payload_tokens(self) -> int:
        t1_ids = {c.id for c in self.tier1_chunks}
        t2_ids = {c.id for c in self.tier2_chunks}
        total = 0
        for c in self.all_chunks:
            if c.id in t2_ids:
                total += _tokens(tier2_repr(c))
            elif c.id in t1_ids:
                total += _tokens(tier1_repr(c))
            else:
                total += _tokens(tier0_repr(c))
        return total

    def full_codebase_tokens(self) -> int:
        return sum(c.raw_tokens for c in self.all_chunks)

    def savings_pct(self) -> float:
        full = self.full_codebase_tokens()
        return (1.0 - self.total_payload_tokens() / full) * 100 if full else 0.0


# ── Retriever ──────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, chunks: list[Chunk], top_k_tier1: int = 4):
        self.chunks = chunks
        self.top_k = top_k_tier1
        self._has_embeddings = any(c.embedding for c in chunks)
        self._tfidf: TFIDFScorer | None = None

    def _get_tfidf(self) -> TFIDFScorer:
        if self._tfidf is None:
            self._tfidf = TFIDFScorer(self.chunks)
        return self._tfidf

    def retrieve(self, query: str) -> RetrievalResult:
        intent = classify_intent(query)
        search_mode = "tfidf"

        # Primary: semantic search with embeddings
        if self._has_embeddings:
            q_vec = embed_query(query)
            if q_vec:
                search_mode = "semantic"
                chunk_vecs = [c.embedding for c in self.chunks]
                raw_scores = _cosine_similarity_batch(q_vec, chunk_vecs)
                # Blend with name-matching boost
                scores = [
                    raw + _name_boost(query, c)
                    for raw, c in zip(raw_scores, self.chunks)
                ]
            else:
                scores = self._tfidf_scores(query)
        else:
            scores = self._tfidf_scores(query)

        scored = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1], reverse=True
        )

        tier0 = list(self.chunks)

        # Identify edit target for EDIT intent
        tier2: list[Chunk] = []
        if intent == Intent.EDIT:
            q_lower = query.lower()
            name_match = next(
                (c for c in self.chunks if c.name and c.name.lower() in q_lower),
                None
            )
            target = name_match or (scored[0][0] if scored and scored[0][1] > 0 else None)
            if target:
                tier2 = [target]

        t2_ids = {c.id for c in tier2}
        tier1 = [
            c for c, score in scored
            if score > 0 and c.id not in t2_ids
        ][:self.top_k]

        return RetrievalResult(
            intent=intent,
            query=query,
            search_mode=search_mode,
            tier0_chunks=tier0,
            tier1_chunks=tier1,
            tier2_chunks=tier2,
            all_chunks=self.chunks,
        )

    def _tfidf_scores(self, query: str) -> list[float]:
        """TF-IDF cosine similarity + name boost."""
        tfidf = self._get_tfidf()
        raw = tfidf.score(query)
        return [s + _name_boost(query, c) for s, c in zip(raw, self.chunks)]

    def get_full_source(self, chunk_id: str) -> Optional[str]:
        chunk = next((c for c in self.chunks if c.id == chunk_id), None)
        if not chunk:
            chunk = next((c for c in self.chunks if c.id.startswith(chunk_id[:8])), None)
        return tier2_repr(chunk) if chunk else None

    def get_full_sources(self, chunk_ids: list[str]) -> dict[str, str]:
        return {
            cid: src
            for cid in chunk_ids
            if (src := self.get_full_source(cid))
        }
