"""Document-level retrieval metrics for the MultiHop-RAG baseline."""

from collections.abc import Iterable, Mapping, Sequence

from .models import Document, Query, RetrievalResult

OFFICIAL_METRICS = ("Hits@4", "Hits@10", "MAP@10", "MRR@10")
CUSTOM_METRICS = (
    "Evidence Recall@4",
    "Evidence Recall@10",
    "Complete Evidence Coverage@4",
    "Complete Evidence Coverage@10",
)


def _normalize(text: str) -> str:
    """Use the reference scorer's literal normalization."""
    return text.replace(" ", "").replace("\n", "")


def _official_scores(texts: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    # Reference: https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py
    # Its MAP contribution is newly found facts / current rank.
    found: set[str] = set()
    first_rank = 0
    precision_sum = 0.0
    for rank, text in enumerate(texts[:10], start=1):
        matches = {fact for fact in gold if fact in text}
        if not matches:
            continue
        if not first_rank:
            first_rank = rank
        precision_sum += len(matches - found) / rank
        found.update(matches)
    return {
        "Hits@4": float(0 < first_rank <= 4),
        "Hits@10": float(bool(first_rank)),
        "MAP@10": precision_sum / min(len(gold), 10),
        "MRR@10": 1 / first_rank if first_rank else 0.0,
    }


def _coverage_scores(
    document_ids: Sequence[str],
    texts: Sequence[str],
    required_evidence: set[tuple[str | None, str]],
) -> dict[str, float]:
    scores = {}
    for limit in (4, 10):
        ids = set(document_ids[:limit])
        evidence_found = sum(
            source_id in ids if source_id is not None else any(
                fact in text for text in texts[:limit]
            )
            for source_id, fact in required_evidence
        )
        scores[f"Evidence Recall@{limit}"] = evidence_found / len(required_evidence)
        scores[f"Complete Evidence Coverage@{limit}"] = float(
            evidence_found == len(required_evidence)
        )
    return scores


def _ordered_ids(
    query_id: str, rows: Sequence[RetrievalResult], documents: Mapping[str, str]
) -> list[str]:
    ids = []
    seen = set()
    for rank, row in enumerate(sorted(rows, key=lambda item: item.rank), start=1):
        if row.query_id != query_id or row.rank != rank:
            raise ValueError(f"{query_id}: ranking must have matching query IDs and contiguous ranks")
        if row.document_id not in documents:
            raise ValueError(f"{query_id}: unknown retrieved document {row.document_id!r}")
        if row.document_id in seen:
            raise ValueError(f"{query_id}: duplicate retrieved document {row.document_id!r}")
        seen.add(row.document_id)
        ids.append(row.document_id)
    return ids


def evaluate(
    queries: Iterable[Query],
    documents: Iterable[Document],
    rankings: Mapping[str, Sequence[RetrievalResult]],
) -> dict:
    """Score original article bodies and source document coverage separately.

    Official gold facts are normalized and matched against the article bodies.
    Custom coverage credits a fact via its resolved source document ID or, if
    unresolved, the same literal text match. Distinct facts in the same
    article count individually. Repeated source/fact pairs count once.
    """
    bodies = {}
    for document in documents:
        if document.document_id in bodies:
            raise ValueError(f"duplicate corpus document {document.document_id!r}")
        bodies[document.document_id] = _normalize(document.text)
    official_totals = dict.fromkeys(OFFICIAL_METRICS, 0.0)
    custom_totals = dict.fromkeys(CUSTOM_METRICS, 0.0)
    answerable = unanswerable = 0
    null_retrievals = {4: 0, 10: 0}
    null_with_results = 0
    seen_queries = set()
    for query in queries:
        if query.query_id in seen_queries or query.query_id not in rankings:
            raise ValueError(f"{query.query_id}: expected exactly one query and a ranking")
        seen_queries.add(query.query_id)
        ids = _ordered_ids(query.query_id, rankings[query.query_id], bodies)
        if query.question_type == "null_query":
            unanswerable += 1
            null_with_results += bool(ids)
            for limit in null_retrievals:
                null_retrievals[limit] += min(len(ids), limit)
            continue
        facts = [_normalize(evidence.evidence_text) for evidence in query.evidence_list]
        if not facts or not all(facts):
            raise ValueError(f"{query.query_id}: answerable question requires nonempty gold facts")
        texts = [bodies[document_id] for document_id in ids[:10]]
        evidence_keys = {
            (evidence.source_document_id, fact)
            for evidence, fact in zip(query.evidence_list, facts, strict=True)
        }
        answerable += 1
        for metric, value in _official_scores(texts, facts).items():
            official_totals[metric] += value
        for metric, value in _coverage_scores(ids, texts, evidence_keys).items():
            custom_totals[metric] += value

    return {
        "official": {
            metric: total / answerable if answerable else None
            for metric, total in official_totals.items()
        },
        "custom": {
            metric: total / answerable if answerable else None
            for metric, total in custom_totals.items()
        },
        "answerable_queries": answerable,
        "unanswerable": {
            "query_count": unanswerable,
            "queries_with_results": null_with_results,
            "queries_without_results": unanswerable - null_with_results,
            "average_returned_documents@4": null_retrievals[4] / unanswerable if unanswerable else None,
            "average_returned_documents@10": null_retrievals[10] / unanswerable if unanswerable else None,
        },
    }
