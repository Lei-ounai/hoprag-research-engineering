"""Read existing ranks and vectors without running a retrieval service."""

from collections import defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np

from hoprag.models import RetrievalResult
from hoprag.retrievers import Chunk, HybridRetriever, _document_chunks, _normalize


METHODS = ("bm25", "dense", "rrf")


def iter_rankings(path, queries, document_ids, config):
    """Reuse both branch rankings and extend RRF using its original input depth."""
    fuser = HybridRetriever(None, None, rrf_k=config["rrf_k"], fusion_depth=config["fusion_depth"])
    with path.open(encoding="utf-8") as stream:
        for query, line in zip(queries, stream, strict=True):
            record = json.loads(line)
            if record["query_id"] != query.query_id or record["question_type"] != query.question_type:
                raise ValueError(f"Ranking/query mismatch: {query.query_id}")
            if record["evidence_count"] != len(query.evidence_list):
                raise ValueError(f"Evidence count mismatch: {query.query_id}")

            ranks = {}
            for method in ("bm25", "dense", "hybrid"):
                rows = [RetrievalResult(**row) for row in record[method]]
                ids = set()
                for rank, row in enumerate(rows, start=1):
                    if (row.query_id != query.query_id or row.rank != rank or row.retrieval_method != method
                            or row.document_id not in document_ids or row.document_id in ids
                            or not np.isfinite(row.score)):
                        raise ValueError(f"Invalid {method} ranking for {query.query_id}")
                    ids.add(row.document_id)
                ranks[method] = rows

            depth = config["fusion_depth"]
            if len(ranks["dense"]) != min(depth, len(document_ids)) or len(ranks["bm25"]) > depth:
                raise ValueError(f"Unexpected branch depth for {query.query_id}")
            extended = fuser.fuse(query, ranks["bm25"], ranks["dense"], 2 * depth)
            if ranks.pop("hybrid") != extended[:config["top_k"]]:
                raise ValueError(f"Stored RRF prefix does not match replay: {query.query_id}")
            ranks["rrf"] = extended
            yield query, ranks


def load_chunks(path, documents, limit):
    by_document = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            chunk = Chunk(**json.loads(line))
            by_document[chunk.document_id].append(chunk)
    if set(by_document) != set(documents):
        raise ValueError("Stored chunks do not reference exactly the corpus document IDs")
    for doc_id, chunks in by_document.items():
        if chunks != list(_document_chunks(documents[doc_id], limit)):
            raise ValueError(f"Stored chunk mapping/content differs from baseline: {doc_id}")
    return dict(by_document)


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def load_cached_vectors(cache_path, model, dimensions, chunks, queries):
    """Validate every expected vector first. No writable DB or API client exists here."""
    source_hashes = {text_hash(chunk.text) for items in chunks.values() for chunk in items}
    query_hashes = {text_hash(query.question) for query in queries}
    vectors = {}
    invalid = set()
    read_error = None
    try:
        if not cache_path.is_file():
            raise FileNotFoundError(str(cache_path))
        sidecar = Path(f"{cache_path}-wal")
        if sidecar.exists() and sidecar.stat().st_size:
            raise OSError(f"Active SQLite WAL must be resolved before immutable audit: {sidecar}")
        with closing(sqlite3.connect(cache_path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            rows = db.execute("SELECT text_sha256, vector FROM embeddings WHERE model = ?", (model,))
            for key, blob in rows:
                if not isinstance(blob, bytes) or len(blob) != 4 * dimensions:
                    invalid.add(key)
                    continue
                vector = np.frombuffer(blob, dtype="<f4")
                if not np.isfinite(vector).all() or not np.any(vector):
                    invalid.add(key)
                    continue
                vectors[key] = vector
    except (OSError, sqlite3.Error) as exc:
        read_error = str(exc)
        vectors.clear()

    audit = {
        "model": model,
        "read_error": read_error,
        "dimensions": dimensions,
        "expected_corpus_vectors": len(source_hashes),
        "expected_query_vectors": len(query_hashes),
        "missing_corpus_vectors": len(source_hashes - vectors.keys()),
        "missing_query_vectors": len(query_hashes - vectors.keys()),
        "invalid_expected_vectors": len((source_hashes | query_hashes) & invalid),
        "cache_valid_vectors": len(vectors),
        "embedding_requests": 0,
    }
    return vectors, audit


class CachedDenseDiagnostics:
    """Replay existing cosine scores to inspect parent and evidence-bearing chunks."""

    def __init__(self, chunks, vectors):
        self.doc_ids = tuple(sorted(chunks))
        self.chunks = tuple(chunk for doc_id in self.doc_ids for chunk in chunks[doc_id])
        self.index = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        self.doc_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.offsets = np.cumsum([0, *(len(chunks[doc_id]) for doc_id in self.doc_ids)])
        self.matrix = _normalize(np.stack([vectors[text_hash(chunk.text)] for chunk in self.chunks]))
        self.vectors = vectors

    def scores(self, question, saved_ranking):
        query = _normalize(self.vectors[text_hash(question)][None, :])[0]
        chunk_scores = np.einsum("ij,j->i", self.matrix, query)
        parent_scores = np.maximum.reduceat(chunk_scores, self.offsets[:-1])
        # Doc IDs are already sorted, so stable descending argsort matches the baseline tie rule.
        order = np.argsort(-parent_scores, kind="stable")
        if any(
            row.document_id != self.doc_ids[index] or row.score != float(parent_scores[index])
            for row, index in zip(saved_ranking, order[:len(saved_ranking)], strict=True)
        ):
            raise ValueError("Read-only Dense replay does not match saved ranks and scores")
        ranks = np.empty(len(order), dtype=np.int32)
        ranks[order] = np.arange(1, len(order) + 1)
        return chunk_scores, ranks

    def inspect(self, scores, document_id, supporting_chunk_ids):
        chunk_scores, ranks = scores
        index = self.doc_index[document_id]
        start, end = self.offsets[index:index + 2]
        winner = int(start + np.argmax(chunk_scores[start:end]))
        support_indices = [self.index[chunk_id] for chunk_id in supporting_chunk_ids]
        best_support = max(support_indices, key=lambda i: chunk_scores[i]) if support_indices else None
        return {
            "document_rank_full": int(ranks[index]),
            "winning_chunk_id": self.chunks[winner].chunk_id,
            "winning_chunk_score": float(chunk_scores[winner]),
            "winning_chunk_contains_fact": winner in support_indices,
            "best_fact_chunk_id": self.chunks[best_support].chunk_id if best_support is not None else None,
            "best_fact_chunk_score": float(chunk_scores[best_support]) if best_support is not None else None,
        }


def compact_ranking(rows, limit=100):
    return [{"document_id": row.document_id, "rank": row.rank, "score": row.score} for row in rows[:limit]]
