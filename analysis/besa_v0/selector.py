"""Gold-free selection of unique evidence parents from a frozen P5-B pool."""

from collections.abc import Mapping
from math import isfinite
import re
from unicodedata import normalize

import numpy as np


MODES = ("A", "B", "C", "D", "E")
_TOKEN = re.compile(r"[^\W_]+")
_NEW_SUPPORT = 0.05
_DUPLICATE_TEXT_OVERLAP = 0.80
_NEGATIONS = {"no", "not", "never", "without", "denied", "deny", "neither", "nor"}


def _basis(pool, order):
    candidates = {}
    for doc in pool.candidates:
        if doc.document_id in candidates:
            raise ValueError(f"Duplicate parent in candidate pool: {doc.document_id}")
        candidates[doc.document_id] = doc
    ordered = tuple(order)
    if len(ordered) != len(candidates) or set(ordered) != candidates.keys():
        raise ValueError("P5-B order must contain every candidate parent exactly once")
    ranks = {doc_id: index for index, doc_id in enumerate(ordered, 1)}
    # Reciprocal rank on a [0,1] scale, truncated only by the frozen candidate count.
    # Ranks remain steep near the top to protect the P5-B prior.
    count = len(ordered)
    prior = {
        doc_id: (1 / rank - 1 / count) / (1 - 1 / count) if count > 1 else 1.0
        for doc_id, rank in ranks.items()
    }
    return candidates, ordered, ranks, prior


def _baseline(mode, order, prior, limit, reason):
    return {
        "mode": mode,
        "document_ids": list(order[:limit]),
        "full_document_ids": list(order),
        "fallback_reason": reason,
        "lambda_bridge": 0,
        "choices": [
            {"document_id": doc_id, "position": index, "policy": "p5b",
             "prior": prior[doc_id], "marginal_coverage": 0.0, "redundancy": 0.0,
             "score": prior[doc_id], "need_gains": {}, "supports": {}, "duplicate_evidence": None}
            for index, doc_id in enumerate(order[:limit], 1)
        ],
    }


def _supports(scores, candidates):
    if not isinstance(scores, Mapping) or not 1 <= len(scores) <= 4:
        raise ValueError("Expected one to four scored evidence needs")
    by_document = {doc_id: {} for doc_id in candidates}
    chunk_parent = {
        chunk.chunk_id: doc_id for doc_id, candidate in candidates.items()
        for chunk in candidate.chunks if chunk.document_id == doc_id
    }
    for need_id, docs in scores.items():
        if not isinstance(need_id, str) or not need_id or not isinstance(docs, Mapping):
            raise ValueError("Invalid need score mapping")
        for doc_id, cell in docs.items():
            if doc_id not in by_document or not isinstance(cell, Mapping):
                raise ValueError("Need score refers to an unknown parent")
            value = cell.get("support")
            chunk_id = cell.get("chunk_id")
            if (not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1
                    or not isinstance(chunk_id, str) or chunk_parent.get(chunk_id) != doc_id):
                raise ValueError("Invalid normalized support or winning chunk parent")
            by_document[doc_id][need_id] = {"support": float(value), "chunk_id": chunk_id}
            raw = cell.get("raw_score")
            if isinstance(raw, (int, float)) and isfinite(raw):
                by_document[doc_id][need_id]["raw_score"] = float(raw)
    return by_document


def _unit(vector):
    result = np.asarray(vector, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    norm = np.linalg.norm(result)
    if not isfinite(norm) or norm <= 0:
        raise ValueError("Invalid vector norm")
    return result / norm


def _parent_cosines(order, vectors):
    if vectors is None:
        raise ValueError("Missing parent vectors")
    rows = []
    for doc_id in order:
        vector = vectors.get(doc_id)
        if vector is None:
            raise ValueError("Missing parent vector")
        rows.append(_unit(vector))
    matrix = np.stack(rows)
    return np.clip(matrix @ matrix.T, 0.0, 1.0)


def _text_signature(text):
    tokens = _TOKEN.findall(normalize("NFKC", text).casefold())
    grams = set(zip(tokens, tokens[1:], tokens[2:]))
    numbers = {token for token in tokens if any(char.isdecimal() for char in token)}
    negations = set(tokens) & _NEGATIONS
    return grams, numbers, negations


class _NeedDuplicates:
    """Penalize only textually close same-source winners for the same covered need."""

    def __init__(self, candidates, supports, vectors, threshold):
        self.supports = supports
        self.vectors = vectors or {}
        self.threshold = threshold
        self.chunks = {c.chunk_id: c.text for d in candidates.values() for c in d.chunks}
        self.sources = {
            doc_id: (
                " ".join(candidate.article_source.casefold().split())
                if isinstance(candidate.article_source, str) else ""
            ) for doc_id, candidate in candidates.items()
        }
        self.signatures = {}
        self.unit_vectors = {}
        self.possible = {}

    def _matching_scores(self, first, second):
        pair = tuple(sorted((first, second)))
        if pair in self.possible:
            return self.possible[pair]
        similarities = (0.0, 0.0)
        for chunk_id in pair:
            if chunk_id not in self.signatures:
                self.signatures[chunk_id] = _text_signature(self.chunks[chunk_id])
        left, right = (self.signatures[chunk_id][0] for chunk_id in pair)
        union = left | right
        overlap = len(left & right) / len(union) if union else 0.0
        if overlap >= _DUPLICATE_TEXT_OVERLAP:
            for chunk_id in pair:
                if chunk_id not in self.unit_vectors:
                    vector = self.vectors.get(chunk_id)
                    try:
                        self.unit_vectors[chunk_id] = _unit(vector) if vector is not None else None
                    except ValueError:
                        self.unit_vectors[chunk_id] = None
            left_vector, right_vector = (self.unit_vectors[chunk_id] for chunk_id in pair)
            if left_vector is not None and right_vector is not None and left_vector.shape == right_vector.shape:
                similarities = (float(np.clip(left_vector @ right_vector, 0, 1)), overlap)
        self.possible[pair] = similarities
        return similarities

    def penalty(self, doc_id, selected, need_gains):
        if any(gain > _NEW_SUPPORT for gain in need_gains.values()):
            return 0.0, None
        source = self.sources[doc_id]
        if not source:
            return 0.0, None
        best = 0.0
        detail = None
        for other in selected:
            if self.sources[other] != source:
                continue
            for need_id, current in self.supports[doc_id].items():
                previous = self.supports[other].get(need_id)
                if current["support"] <= 0 or previous is None or previous["support"] <= 0:
                    continue
                first, second = current["chunk_id"], previous["chunk_id"]
                cosine, overlap = self._matching_scores(first, second)
                if cosine < self.threshold or cosine <= best:
                    continue
                if self.signatures[first][1:] != self.signatures[second][1:]:
                    continue
                best = cosine
                detail = {
                    "need_id": need_id, "matched_document_id": other, "chunk_id": first,
                    "matched_chunk_id": second, "chunk_cosine": cosine, "trigram_jaccard": overlap,
                    "source": source,
                }
        return best, detail


def select(
    pool, baseline_order, *, mode="C", supports=None, parent_vectors=None, chunk_vectors=None,
    anchor_count=1, lambda_prior=0.45, lambda_need=0.55, lambda_repeat=0.15,
    mmr_lambda=0.75, duplicate_threshold=0.90, limit=10, fallback_reason=None,
):
    """Return a top-K plus full ranking. Gold labels are outside this input boundary."""
    if mode not in MODES or type(limit) is not int or limit < 0:
        raise ValueError("Unknown selector mode or invalid output limit")
    if (not all(isfinite(x) and x >= 0 for x in (lambda_prior, lambda_need, lambda_repeat))
            or not 0 <= mmr_lambda <= 1 or not 0 < duplicate_threshold <= 1
            or type(anchor_count) is not int or not 1 <= anchor_count <= 2):
        raise ValueError("Invalid selector parameter")
    candidates, order, ranks, prior = _basis(pool, baseline_order)
    if mode == "A" or fallback_reason or not order or not limit:
        return _baseline(mode, order, prior, limit, fallback_reason)

    if mode == "B":
        try:
            cosine = _parent_cosines(order, parent_vectors)
        except (TypeError, ValueError):
            return _baseline(mode, order, prior, limit, "invalid_or_missing_parent_vectors")
        by_document = {doc_id: {} for doc_id in order}
    else:
        try:
            by_document = _supports(supports, candidates)
        except ValueError:
            return _baseline(mode, order, prior, limit, "invalid_or_missing_need_support")
        if not any(cell["support"] > 0 for docs in by_document.values() for cell in docs.values()):
            return _baseline(mode, order, prior, limit, "no_positive_need_support")

    selected = []
    remaining = set(order)
    covered = {}
    choices = []
    duplicate = (
        _NeedDuplicates(candidates, by_document, chunk_vectors, duplicate_threshold)
        if mode in ("D", "E") else None
    )
    need_count = len(supports) if mode != "B" else 0
    for position in range(1, min(limit, len(order)) + 1):
        anchored = mode == "E" and position <= anchor_count
        requested = (order[position - 1],) if anchored else remaining
        best_key = None
        winner = None
        for doc_id in requested:
            cells = by_document[doc_id]
            gains = {
                need_id: max(0.0, cell["support"] - covered.get(need_id, 0.0))
                for need_id, cell in cells.items()
            }
            marginal = sum(gains.values()) / need_count if need_count else 0.0
            relevance = prior[doc_id]
            redundancy = 0.0
            detail = None
            if mode == "B":
                redundancy = max((float(cosine[ranks[doc_id] - 1, ranks[s] - 1]) for s in selected), default=0.0)
                value = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy
            else:
                if duplicate is not None:
                    redundancy, detail = duplicate.penalty(doc_id, selected, gains)
                value = lambda_prior * relevance + lambda_need * marginal - lambda_repeat * redundancy
            key = (-value, ranks[doc_id], doc_id)
            if best_key is None or key < best_key:
                best_key = key
                winner = {
                    "document_id": doc_id, "position": position, "policy": "anchor" if anchored else mode,
                    "prior": relevance, "marginal_coverage": marginal, "redundancy": redundancy,
                    "score": value, "need_gains": gains, "supports": cells, "duplicate_evidence": detail,
                }
        if winner is None:
            return _baseline(mode, order, prior, limit, "invalid_selector_output")
        doc_id = winner["document_id"]
        selected.append(doc_id)
        remaining.remove(doc_id)
        for need_id, cell in by_document[doc_id].items():
            covered[need_id] = max(cell["support"], covered.get(need_id, 0.0))
        choices.append(winner)

    return {
        "mode": mode,
        "document_ids": selected,
        "full_document_ids": [*selected, *(doc_id for doc_id in order if doc_id in remaining)],
        "fallback_reason": None,
        "lambda_bridge": 0,
        "choices": choices,
    }
