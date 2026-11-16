"""BM25, exact dense cosine retrieval and reciprocal rank fusion."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import BenchmarkConfig
from .embeddings import OllamaEmbedder
from .models import Document, Query, RetrievalResult


_WORD = re.compile(r"[^\W_]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def _unique_documents(documents: Iterable[Document]) -> tuple[Document, ...]:
    by_id: dict[str, Document] = {}
    for document in documents:
        prior = by_id.setdefault(document.document_id, document)
        if prior != document:
            raise ValueError(f"Conflicting records for document ID {document.document_id!r}.")
    if not by_id:
        raise ValueError("At least one document is required.")
    return tuple(by_id[doc_id] for doc_id in sorted(by_id))


def _results(
    query: Query,
    method: str,
    candidates: Iterable[tuple[str, float]],
    top_k: int,
) -> list[RetrievalResult]:
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    ordered = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
    return [
        RetrievalResult(query.query_id, doc_id, method, float(score), rank)
        for rank, (doc_id, score) in enumerate(ordered[:top_k], start=1)
    ]


class BM25Retriever:
    method = "bm25"

    def __init__(self, documents: Iterable[Document]):
        self._documents = _unique_documents(documents)
        tokenized = [_tokens(f"{doc.title}\n{doc.text}") for doc in self._documents]
        self._postings: dict[str, list[int]] = defaultdict(list)
        for index, tokens in enumerate(tokenized):
            for term in dict.fromkeys(tokens):
                self._postings[term].append(index)
        if not self._postings:
            raise ValueError("Corpus contains no indexable tokens.")
        self._index = BM25Okapi(tokenized)

    def search(self, query: Query, top_k: int) -> list[RetrievalResult]:
        terms = list(dict.fromkeys(_tokens(query.question)))
        matched = sorted({index for term in terms for index in self._postings.get(term, ())})
        scores = self._index.get_batch_scores(terms, matched)
        candidates = (
            (self._documents[index].document_id, score)
            for index, score in zip(matched, scores, strict=True)
        )
        return _results(query, self.method, candidates, top_k)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str


def _document_chunks(document: Document, max_bytes: int) -> Iterator[Chunk]:
    data = f"{document.title}\n{document.text}".encode("utf-8")
    start = 0
    number = 0
    while start < len(data):
        end = min(start + max_bytes, len(data))
        while end < len(data) and data[end] & 0xC0 == 0x80:
            end -= 1
        piece = data[start:end].decode("utf-8")
        if end < len(data):
            boundary = max(piece.rfind(" "), piece.rfind("\n"), piece.rfind("\t"))
            if boundary >= len(piece) // 2:
                piece = piece[: boundary + 1]
                end = start + len(piece.encode("utf-8"))
        yield Chunk(f"{document.document_id}::{number:04d}", document.document_id, piece)
        start = end
        number += 1


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Cosine similarity requires finite vectors with nonzero norms.")
    return np.asarray(vectors / norms[:, np.newaxis], dtype=np.float32)


class DenseRetriever:
    method = "dense"

    def __init__(
        self,
        documents: Iterable[Document],
        embedder: OllamaEmbedder,
        *,
        max_chunk_bytes: int = 3072,
    ):
        if max_chunk_bytes < 4:
            raise ValueError("max_chunk_bytes must accommodate a UTF-8 character (at least 4).")
        documents = _unique_documents(documents)
        self._document_ids = tuple(document.document_id for document in documents)
        self._embedder = embedder
        self._offsets = []
        chunks = []
        for document in documents:
            self._offsets.append(len(chunks))
            chunks.extend(_document_chunks(document, max_chunk_bytes))
        self.chunks = tuple(chunks)
        self._matrix = _normalize(embedder.embed([chunk.text for chunk in chunks]))
        self._query_vectors: dict[str, np.ndarray] = {}

    def prepare_queries(self, queries: Iterable[Query]) -> None:
        texts = list(dict.fromkeys(
            query.question for query in queries if query.question not in self._query_vectors
        ))
        if texts:
            vectors = _normalize(self._embedder.embed(texts))
            self._query_vectors.update(zip(texts, vectors, strict=True))

    def search(self, query: Query, top_k: int) -> list[RetrievalResult]:
        self.prepare_queries((query,))
        vector = self._query_vectors[query.question]
        chunk_scores = np.einsum("ij,j->i", self._matrix, vector)
        scores = np.maximum.reduceat(chunk_scores, self._offsets)
        return _results(
            query,
            self.method,
            zip(self._document_ids, scores, strict=True),
            top_k,
        )


class HybridRetriever:
    method = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        *,
        fusion_depth: int = BenchmarkConfig.fusion_depth,
    ):
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative.")
        if fusion_depth < 1:
            raise ValueError("fusion_depth must be positive.")
        self._bm25 = bm25
        self._dense = dense
        self.rrf_k = rrf_k
        self.fusion_depth = fusion_depth

    def fuse(
        self,
        query: Query,
        bm25_results: Sequence[RetrievalResult],
        dense_results: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        fused: dict[str, float] = defaultdict(float)
        for ranking in (bm25_results, dense_results):
            best_rank: dict[str, int] = {}
            for result in ranking:
                if result.query_id != query.query_id or result.rank < 1:
                    raise ValueError("Each ranking must contain valid ranks for the selected query.")
                previous = best_rank.get(result.document_id, result.rank)
                best_rank[result.document_id] = min(previous, result.rank)
            for document_id, rank in best_rank.items():
                fused[document_id] += 1.0 / (self.rrf_k + rank)
        return _results(query, self.method, fused.items(), top_k)

    def search(self, query: Query, top_k: int) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        branch_depth = max(top_k, self.fusion_depth)
        return self.fuse(
            query,
            self._bm25.search(query, branch_depth),
            self._dense.search(query, branch_depth),
            top_k,
        )
