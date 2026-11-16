"""Gold-free candidate text construction and document-level rank aggregation."""

from dataclasses import dataclass
from math import isfinite


TEMPLATE = "title-newline-original-chunk-body/v1"
FINAL_K = 10
FUSION_K = 60


@dataclass(frozen=True)
class Pair:
    document_id: str
    chunk_id: str
    text: str


def make_pairs(pool):
    pairs = []
    for candidate in pool.candidates:
        title_prefix = candidate.title + "\n"
        for index, chunk in enumerate(candidate.chunks):
            if chunk.document_id != candidate.document_id:
                raise ValueError(f"Chunk {chunk.chunk_id} has the wrong parent")
            text = chunk.text.removeprefix(title_prefix) if index == 0 else chunk.text
            pairs.append(Pair(candidate.document_id, chunk.chunk_id, title_prefix + text))
    return pairs


def _rank(scores):
    if not all(isfinite(score) for score in scores.values()):
        raise ValueError("Non-finite ranking score")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [{"document_id": doc, "score": score, "rank": i} for i, (doc, score) in enumerate(ordered, 1)]


def aggregate(pool, pairs, pair_scores):
    parents = {}
    winners = {}
    for pair, score in zip(pairs, pair_scores, strict=True):
        if not isfinite(score):
            raise ValueError(f"Non-finite score for {pair.chunk_id}")
        previous = parents.get(pair.document_id)
        if previous is None or score > previous:
            parents[pair.document_id] = score
            winners[pair.document_id] = pair.chunk_id
    expected = {candidate.document_id for candidate in pool.candidates}
    if len(expected) != len(pool.candidates) or parents.keys() != expected:
        raise ValueError("Each candidate parent must be represented once")
    result = _rank(parents)
    for row in result:
        row["winning_chunk_id"] = winners[row["document_id"]]
    return result


def fuse(pool, ce_ranking, k=FUSION_K):
    original = {candidate.document_id: candidate.rrf_rank for candidate in pool.candidates}
    ce = {row["document_id"]: row["rank"] for row in ce_ranking}
    if ce.keys() != original.keys() or len(ce) != len(ce_ranking):
        raise ValueError("RRF/CE fusion requires the same unique candidate documents")
    return _rank({doc: 1 / (k + original[doc]) + 1 / (k + ce[doc]) for doc in original})
