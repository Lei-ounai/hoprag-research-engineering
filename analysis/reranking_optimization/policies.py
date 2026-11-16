"""Gold-free selections and weighted fusion over frozen Round 5 candidates."""

from math import isfinite

from analysis.reranking.core import FUSION_K, _rank
from analysis.reranking.pool import Candidate, QueryPool


POLICIES = ("P1", "P2", "P3", "P4", "P5", "P6")
DEV_SUPERSET = "DEV_SUPERSET"
POLICY_UNION = "POLICY_UNION"
WEIGHTS = {"A": 1, "B": 2, "C": 3, "D": 4}
LIMITS = {
    "P1": {"rrf": 20},
    "P2": {"rrf": 30},
    "P3": {"rrf": 50},
    "P4": {"bm25": 10, "dense": 10},
    "P5": {"bm25": 15, "dense": 15},
    "P6": {"bm25": 25, "dense": 25},
    DEV_SUPERSET: {"rrf": 50, "bm25": 50, "dense": 50},
    POLICY_UNION: {"rrf": 50, "bm25": 25, "dense": 25},
}


def _parents(pool: QueryPool) -> dict[str, Candidate]:
    unique = {}
    for candidate in pool.candidates:
        previous = unique.setdefault(candidate.document_id, candidate)
        if candidate != previous:
            raise ValueError(f"Conflicting candidate records for {candidate.document_id}")
    return unique


def _admitted_by(candidate: Candidate, policy: str) -> tuple[str, ...]:
    lanes = []
    for lane, cutoff in LIMITS[policy].items():
        rank = getattr(candidate, f"{lane}_rank")
        if rank is not None and rank <= cutoff:
            lanes.append(lane)
    return tuple(lanes)


def select(pool: QueryPool, policy: str) -> QueryPool:
    """Select distinct parent IDs while retaining their frozen ranks and all chunks."""
    if policy not in LIMITS:
        raise ValueError(f"Unknown candidate pool policy: {policy!r}")
    selected = tuple(
        candidate for candidate in _parents(pool).values()
        if _admitted_by(candidate, policy)
    )
    return QueryPool(pool.query_id, pool.question, selected)


def _rank_term(rank: int | None, weight: float, k: int) -> float:
    if rank is None:
        return 0.0
    if type(rank) is not int or rank < 1:
        raise ValueError(f"Invalid positive rank: {rank!r}")
    return weight / (k + rank)


def weighted_fuse(
    pool: QueryPool, ce_ranking, weight_rrf: float, weight_jina: float = 1, k: int = FUSION_K,
) -> list[dict]:
    """Fuse within-pool rankings while preserving the original RRF order."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not all(isfinite(weight) and weight >= 0 for weight in (weight_rrf, weight_jina)):
        raise ValueError("Fusion weights must be finite and non-negative")
    if weight_rrf == weight_jina == 0:
        raise ValueError("At least one fusion weight must be positive")
    candidates = _parents(pool)
    jina = {}
    for row in ce_ranking:
        doc_id = row["document_id"]
        if doc_id not in candidates or doc_id in jina:
            raise ValueError(f"Jina ranking references an unknown or duplicate parent: {doc_id}")
        jina[doc_id] = row["rank"]
    original_rrf = {
        doc_id: candidate.rrf_rank for doc_id, candidate in candidates.items()
        if candidate.rrf_rank is not None
    }
    if any(type(rank) is not int or rank < 1 for rank in original_rrf.values()):
        raise ValueError("Invalid original RRF rank")
    rrf_order = sorted(original_rrf, key=lambda doc_id: (original_rrf[doc_id], doc_id))
    rrf = {doc_id: rank for rank, doc_id in enumerate(rrf_order, 1)}
    return _rank({
        doc_id: _rank_term(rrf.get(doc_id), weight_rrf, k)
        + _rank_term(jina.get(doc_id), weight_jina, k)
        for doc_id in candidates
    })


def candidate_record(pool: QueryPool, policy: str | None = None) -> dict:
    """Reference title/body/chunk data in the immutable Round 5 document file."""
    return {
        "query_id": pool.query_id,
        "policy": policy,
        "candidates": [
            {
                "document_id": candidate.document_id,
                "parent_document_id": candidate.document_id,
                "title": candidate.title,
                "chunk_ids": [chunk.chunk_id for chunk in candidate.chunks],
                "source": list(candidate.source),
                "admitted_by": list(_admitted_by(candidate, policy)) if policy else None,
                "bm25_rank": candidate.bm25_rank,
                "dense_rank": candidate.dense_rank,
                "rrf_rank": candidate.rrf_rank,
                "bm25_score": candidate.bm25_score,
                "dense_score": candidate.dense_score,
                "rrf_score": candidate.rrf_score,
            }
            for candidate in pool.candidates
        ],
    }


def policy_metadata() -> dict:
    return {
        "pool_cutoffs": {name: dict(lanes) for name, lanes in LIMITS.items()},
        "content_source": "analysis/reranking/candidate_documents.jsonl",
        "parent_dedup_key": "document_id",
        "parent_score": "maximum Jina score across all original chunks",
        "fusion": {
            "formula": "w_rrf/(60+rrf_rank) + w_jina/(60+jina_rank)",
            "k": FUSION_K,
            "weights": {name: {"w_rrf": weight, "w_jina": 1} for name, weight in WEIGHTS.items()},
            "rrf_rank": "contiguous within the selected pool, ordered by original RRF position; original ranks remain in candidate metadata",
            "jina_rank": "within the same selected parent pool",
            "missing_rank": "zero contribution from the missing lane",
            "tie_break": "ascending document ID",
        },
    }
