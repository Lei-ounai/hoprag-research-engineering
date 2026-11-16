"""Fuse frozen P5-B ranks with expansion ranks and optional local Jina scores."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import isfinite
import os
from pathlib import Path
import sqlite3
from time import monotonic, perf_counter
from types import SimpleNamespace

from analysis.besa_v0.scoring import NeedScoreCache
from analysis.reranking.backend import FastEmbedBackend
from analysis.reranking.cache import digest
from analysis.reranking.core import FINAL_K, FUSION_K, make_pairs
from analysis.reranking_optimization.runner import identity
from hoprag.retrievers import _tokens


MAX_NEW_PAIRS = 8
LOCAL_CACHE_SOURCE = "local_cache"


@dataclass(frozen=True)
class DocumentScore:
    score: float
    chunk_id: str


@dataclass(frozen=True)
class ScoreBatch:
    parents: dict[str, DocumentScore] = field(default_factory=dict)
    actual_pairs: int = 0
    historical_hits: int = 0
    local_hits: int = 0
    duplicate_pairs: int = 0
    new_pairs: int = 0
    inference_seconds: float = 0.0
    wall_seconds: float = 0.0
    truncated_pairs: int = 0
    fallback_reason: str | None = None

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "parents": {key: DocumentScore(**row) for key, row in data["parents"].items()}})


@dataclass(frozen=True)
class RankingResult:
    ranking: list[str]
    document_ids: list[str]
    witnesses: dict[str, str]
    fallback_reason: str | None = None


def fuse(
    baseline,
    expanded,
    *,
    mode="rrf",
    expansion_weight=1.0,
    anchor=None,
    score_batch=None,
    baseline_scores=None,
    baseline_witnesses=None,
    expansion_witnesses=None,
    jina_strength=0.20,
):
    """Preserve P5-B's relative ranking and add unique expansion parents in a separate lane."""
    anchor = (5 if mode == "anchor" else 0) if anchor is None else anchor
    if mode not in {"baseline", "rrf", "anchor", "jina"}:
        raise ValueError(f"Unknown ranking mode: {mode}")
    if type(anchor) is not int or not 0 <= anchor <= FINAL_K:
        raise ValueError("Anchor must be within the Top 10 budget")
    if not isfinite(expansion_weight) or expansion_weight < 0:
        raise ValueError("Expansion weight must be finite and non-negative")
    if not isfinite(jina_strength) or not 0 <= jina_strength <= 1:
        raise ValueError("Jina strength must be within [0, 1]")

    baseline = list(baseline)
    if len(set(baseline)) != len(baseline):
        raise ValueError("Expected unique P5-B parents")
    novel = [value for value in dict.fromkeys(expanded) if value not in set(baseline)]
    witnesses = {doc: baseline_witnesses[doc] for doc in baseline if baseline_witnesses and doc in baseline_witnesses}
    if mode == "baseline" or not novel:
        return RankingResult(baseline, baseline[:FINAL_K], witnesses)
    modifiers = {}
    if mode == "jina" and jina_strength:
        if score_batch is None or score_batch.fallback_reason:
            reason = score_batch.fallback_reason if score_batch is not None else "jina_scores_unavailable"
            return RankingResult(baseline, baseline[:FINAL_K], witnesses, reason)
        if any(doc not in novel or not isfinite(row.score) for doc, row in score_batch.parents.items()):
            return RankingResult(baseline, baseline[:FINAL_K], witnesses, "invalid_jina_scores")
        reference = sorted(baseline_scores.values()) if baseline_scores is not None else []
        if score_batch.parents and (
            not reference or set(baseline_scores) != set(baseline) or not all(map(isfinite, reference))
        ):
            return RankingResult(baseline, baseline[:FINAL_K], witnesses, "invalid_baseline_jina_scores")
        for doc, scored in score_batch.parents.items():
            left = bisect_left(reference, scored.score)
            right = bisect_right(reference, scored.score)
            percentile = (left + right) / (2 * len(reference))
            modifiers[doc] = 1 + jina_strength * (2 * percentile - 1)
    if expansion_witnesses:
        witnesses.update({doc: expansion_witnesses[doc] for doc in novel if doc in expansion_witnesses})
    if mode == "jina" and jina_strength and score_batch is not None:
        witnesses.update({doc: row.chunk_id for doc, row in score_batch.parents.items()})

    protected = baseline[:anchor]
    values = [
        (doc, 1 / (FUSION_K + rank), 0, rank)
        for rank, doc in enumerate(baseline, 1) if rank > anchor
    ]
    values.extend(
        (doc, expansion_weight * modifiers.get(doc, 1.0) / (FUSION_K + rank), 1, rank)
        for rank, doc in enumerate(novel, 1)
    )
    ranking = [*protected, *(doc for doc, *_ in sorted(values, key=lambda row: (-row[1], row[2], row[3], row[0])))]
    return RankingResult(ranking, ranking[:FINAL_K], witnesses)


def local_jina_backend(model_cache, metadata, *, threads=4, batch_size=16):
    """Construct the pinned CPU model from the existing snapshot using local files only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    backend = FastEmbedBackend(
        metadata["model_id"], Path(model_cache), threads=threads, context_mode="unified",
        batch_size=batch_size, unified_limit=256,
    )
    if backend.metadata["provider"] != "CPUExecutionProvider" or identity(backend.metadata) != identity(metadata):
        raise ValueError("Existing Jina model differs from the frozen scoring configuration")
    return backend


class Scorer:
    """Score a capped set of new parent/Chunk pairs and persist only new cache entries."""

    def __init__(
        self, cache_path, historical_path, metadata, backend, *, max_parents=MAX_NEW_PAIRS,
        chunks_per_parent=1, batch_size=16,
    ):
        if any(type(value) is not int or value < 1 for value in (max_parents, chunks_per_parent, batch_size)):
            raise ValueError("Expected positive integer Jina limits")
        if max_parents * chunks_per_parent > MAX_NEW_PAIRS:
            raise ValueError("Schedule exceeds the hard Jina pair budget")
        history = (historical_path,) if isinstance(historical_path, (str, Path)) else (historical_path or ())
        self.history = {f"historical_{index}": Path(p) for index, p in enumerate(history)}
        self.cache_path = cache_path
        self.metadata = metadata
        self.backend = backend
        self.max_parents = max_parents
        self.chunks_per_parent = chunks_per_parent
        self.batch_size = batch_size
        self.cache = None
        self.cache_error = None

    def __enter__(self):
        try:
            self.cache = NeedScoreCache(self.cache_path, self.history, self.metadata)
        except (OSError, sqlite3.Error, ValueError) as error:
            self.cache_error = f"jina_cache_unavailable:{type(error).__name__}"
        return self

    def __exit__(self, *error):
        if self.cache is not None:
            self.cache.__exit__(*error)
            self.cache = None

    def _pairs(self, question, expansion_ids, corpus, witnesses, chunks):
        documents = []
        for doc in list(dict.fromkeys(expansion_ids))[:self.max_parents]:
            original = corpus[doc]
            blocks = original.chunks if chunks is None else chunks[doc]
            if not blocks or original.document_id != doc:
                raise ValueError(f"No original evidence Chunk for {doc}")
            documents.append(SimpleNamespace(document_id=doc, title=original.title, chunks=blocks))
        query_terms = set(_tokens(question))
        by_parent = {doc.document_id: [] for doc in documents}
        for pair in make_pairs(SimpleNamespace(candidates=documents)):
            by_parent[pair.document_id].append(pair)
        return [
            pair
            for doc, related in by_parent.items()
            for pair in sorted(
                related, key=lambda item: (
                    item.chunk_id != witnesses.get(doc),
                    -len(query_terms.intersection(_tokens(item.text))),
                    item.chunk_id,
                ),
            )[:self.chunks_per_parent]
        ]

    def score(
        self, query_id, question, expansion_ids, corpus, *,
        expansion_witnesses=None, chunks=None, deadline=None,
    ):
        started = perf_counter()
        actual = historical = local = duplicate = new = truncated = 0
        inference_seconds = 0.0
        parents = {}
        reason = None
        try:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError
            pairs = self._pairs(question, expansion_ids, corpus, expansion_witnesses or {}, chunks)
            if pairs and self.cache is None:
                return ScoreBatch(
                    actual_pairs=len(pairs), wall_seconds=perf_counter() - started,
                    fallback_reason=self.cache_error or "jina_cache_not_open",
                )
            actual = len(pairs)
            if pairs:
                query_hash = digest(question)
                hashed = [(digest(pair.text), pair) for pair in pairs]
                unique = dict(hashed)
                saved = self.cache.lookup_pairs(query_hash, unique)
                local = sum(source == LOCAL_CACHE_SOURCE for _, source in saved.values())
                historical = len(saved) - local
                duplicate = len(pairs) - len(unique)
                missing = [(key, pair) for key, pair in unique.items() if key not in saved]
                for start in range(0, len(missing), self.batch_size):
                    if deadline is not None and monotonic() >= deadline:
                        raise TimeoutError
                    if self.backend is None:
                        raise RuntimeError("Jina backend unavailable")
                    batch = missing[start:start + self.batch_size]
                    new += len(batch)
                    replies = list(self.backend.score(question, [pair.text for _, pair in batch]))
                    self.cache.put_batch(query_id, query_hash, batch, replies)
                    inference_seconds += sum(reply.inference_seconds for reply in replies)
                    for (key, _), reply in zip(batch, replies, strict=True):
                        saved[key] = (reply, "new_inference")

                for key, pair in hashed:
                    row = saved[key][0]
                    truncated += bool(row.truncated)
                    if not isfinite(row.score):
                        raise ValueError(f"Non-finite persisted Jina score for {pair.chunk_id}")
                    parent = pair.document_id
                    previous = parents.get(parent)
                    if previous is None or (-row.score, pair.chunk_id) < (-previous.score, previous.chunk_id):
                        parents[parent] = DocumentScore(row.score, pair.chunk_id)
        except TimeoutError:
            reason = "deadline_exhausted"
        except (sqlite3.Error, OSError) as error:
            reason = f"jina_cache_or_io_error:{type(error).__name__}"
        except Exception as error:
            reason = f"jina_scoring_error:{type(error).__name__}"
        return ScoreBatch(
            parents if reason is None else {}, actual, historical, local, duplicate, new, inference_seconds,
            perf_counter() - started, truncated, reason,
        )

