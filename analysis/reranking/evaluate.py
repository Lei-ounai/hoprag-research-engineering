"""Gold-side evaluation for frozen-pool Cross-Encoder experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median

from analysis.rankings import iter_rankings
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize, evaluate
from hoprag.models import RetrievalResult


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASELINE = ROOT / "results" / "baseline"
BASELINE_METHODS = {"B0": "bm25", "B1": "dense", "B2": "rrf"}
DEVELOPMENT_SPLITS = ("dev_5", "dev_20", "dev_100", "answerable_dev", "null_dev")
HELDOUT_SPLITS = ("answerable_heldout", "null_heldout")
SPLIT_NAMES = DEVELOPMENT_SPLITS + HELDOUT_SPLITS
EVIDENCE_FIELDS = (
    "split", "run_id", "cohort_sha256", "method", "query_id", "question_type",
    "evidence_id", "source_document_id", "in_frozen_pool", "rrf_rank", "rank",
    "rank_change", "rrf_worst_evidence_rank", "worst_evidence_rank",
    "lost_from_top10", "gained_into_top10", "worsened_by_at_least_10",
)
NULL_FIELDS = (
    "split", "run_id", "cohort_sha256", "method", "query_id", "candidate_count",
    "returned_count", "top1_document_id", "top10_document_ids", "top10_scores",
    "top1_score", "top10_score_min", "top10_score_median", "top10_score_mean",
    "top10_shared_with_rrf",
)


@dataclass
class Group:
    split: str
    run_id: str
    method: str
    query_ids: tuple[str, ...]
    rankings: dict[str, list[RetrievalResult]]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _average(values) -> float | None:
    return mean(values) if values else None


def _rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_splits(path, queries):
    records = json.loads(path.read_bytes())
    splits = {name: tuple(records[name]) for name in SPLIT_NAMES}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)) or not set(ids).issubset(queries):
            raise ValueError(f"Unknown or duplicate query IDs in {name}")
    for answerable, names in (
        (True, ("answerable_dev", "answerable_heldout")),
        (False, ("null_dev", "null_heldout")),
    ):
        left, right = (set(splits[name]) for name in names)
        expected = {key for key, query in queries.items() if query.is_answerable == answerable}
        if left & right or (left | right) != expected:
            raise ValueError(f"Split pair does not partition answerability group: {names}")
    for name in ("dev_5", "dev_20", "dev_100"):
        if len(splits[name]) != int(name.split("_")[1]) or not set(splits[name]).issubset(splits["answerable_dev"]):
            raise ValueError(f"Invalid development sample: {name}")
    return splits


def _baseline_rankings(dataset, pool_path):
    document_ids = {document.document_id for document in dataset.documents}
    config = json.loads((BASELINE / "run.json").read_bytes())["config"]
    baseline = {}
    pool_rows = {}
    for row in _rows(pool_path):
        query_id = row["query_id"]
        if query_id in pool_rows:
            raise ValueError(f"Duplicate frozen pool: {query_id}")
        pool_rows[query_id] = row
    for query, rankings in iter_rankings(BASELINE / "rankings_full.jsonl", dataset.queries, document_ids, config):
        frozen_row = pool_rows.pop(query.query_id, None)
        if frozen_row is None or frozen_row["question"] != query.question:
            raise ValueError(f"Frozen pool has no matching query: {query.query_id}")
        expected = [(row.document_id, row.rank, row.score) for row in rankings["rrf"]]
        actual = [
            (candidate["document_id"], candidate["rrf_rank"], candidate["rrf_score"])
            for candidate in frozen_row["candidates"]
        ]
        if actual != expected:
            raise ValueError(f"Frozen pool differs from original RRF union: {query.query_id}")
        for method in ("bm25", "dense"):
            original = {row.document_id: (row.rank, row.score) for row in rankings[method]}
            for candidate in frozen_row["candidates"]:
                actual_branch = (candidate[f"{method}_rank"], candidate[f"{method}_score"])
                if actual_branch != original.get(candidate["document_id"], (None, None)):
                    raise ValueError(f"Frozen {method} branch differs for {query.query_id}")
        baseline[query.query_id] = {code: rankings[name] for code, name in BASELINE_METHODS.items()}
    if pool_rows:
        raise ValueError("Frozen pool has query IDs absent from the benchmark")
    return baseline


def validate_reranking(record, frozen):
    """Require a ranking of the entire fixed union, with exactly one row per document."""
    query_id = record["query_id"]
    method = record["method"]
    frozen_ids = {row.document_id for row in frozen}
    rows = []
    for rank, value in enumerate(record["ranking"], start=1):
        if (
            value["rank"] != rank or value["document_id"] not in frozen_ids
            or not isinstance(value["score"], (float, int)) or not math.isfinite(value["score"])
        ):
            raise ValueError(f"Invalid rank, score or candidate: {query_id}/{method}")
        rows.append(RetrievalResult(query_id, value["document_id"], method, value["score"], rank))
    if len(rows) != len(frozen_ids) or len({row.document_id for row in rows}) != len(frozen_ids):
        raise ValueError(f"Ranking does not include each frozen candidate once: {query_id}/{method}")
    if record.get("candidate_count", len(frozen_ids)) != len(frozen_ids):
        raise ValueError(f"Recorded candidate count differs from frozen pool: {query_id}/{method}")
    if "top10" in record:
        expected = [row.document_id for row in rows[:10]]
        actual = [row["document_id"] for row in record["top10"]]
        if actual != expected:
            raise ValueError(f"Top 10 differs from full ranking prefix: {query_id}/{method}")
    return rows


def _read_groups(paths, splits, baseline):
    groups = defaultdict(dict)
    memberships = {name: set(ids) for name, ids in splits.items()}
    for path in paths:
        for record in _rows(path):
            query_id = record["query_id"]
            split = record["split"]
            method = record["method"]
            if method in BASELINE_METHODS:
                raise ValueError(f"Baseline method cannot be supplied by a reranker: {method}")
            if query_id not in memberships.get(split, ()):
                raise ValueError(f"Query {query_id} is outside declared split {split}")
            key = split, record.get("run_id", method), method
            if query_id in groups[key]:
                raise ValueError(f"Duplicate reranking for {key}: {query_id}")
            groups[key][query_id] = validate_reranking(record, baseline[query_id]["B2"])
    return [
        Group(split, run_id, method, tuple(qid for qid in splits[split] if qid in ranks), ranks)
        for (split, run_id, method), ranks in sorted(groups.items())
    ]


def _with_baselines(rerank_groups, splits, baseline, include_heldout):
    cohorts = {(split, splits[split]) for split in DEVELOPMENT_SPLITS}
    if include_heldout:
        cohorts.update((split, splits[split]) for split in HELDOUT_SPLITS)
    cohorts.update((group.split, group.query_ids) for group in rerank_groups)
    baselines = [
        Group(split, "baseline", method, ids, {qid: baseline[qid][method] for qid in ids})
        for split, ids in sorted(cohorts)
        for method in BASELINE_METHODS
    ]
    return baselines + rerank_groups


def evidence_ranks(evidence, ranking, bodies):
    rank_by_id = {row.document_id: row.rank for row in ranking}
    output = []
    for item in evidence:
        if item.source_document_id is not None:
            output.append(rank_by_id.get(item.source_document_id))
        else:
            fact = _normalize(item.evidence_text)
            output.append(next((row.rank for row in ranking if fact in bodies[row.document_id]), None))
    return output


def complete_at_10(ranks):
    return bool(ranks) and all(rank is not None and rank <= 10 for rank in ranks)


def worst_rank(ranks):
    return max(ranks) if ranks and all(rank is not None for rank in ranks) else None


def _metrics_and_details(group, all_queries, documents, bodies, baseline, expected_queries):
    queries = [all_queries[qid] for qid in group.query_ids]
    official = evaluate(queries, documents, group.rankings)
    cohort = sha256("\n".join(sorted(group.query_ids)).encode()).hexdigest()
    shared = {
        "split": group.split,
        "run_id": group.run_id,
        "cohort_sha256": cohort,
        "method": group.method,
    }
    metrics = {
        **shared,
        "queries": len(queries),
        "expected_split_queries": expected_queries,
        "complete_split": len(queries) == expected_queries,
        "answerable_queries": official["answerable_queries"],
        "null_queries": official["unanswerable"]["query_count"],
        "hits_at_10": official["official"]["Hits@10"],
        "map_at_10": official["official"]["MAP@10"],
        "mrr_at_10": official["official"]["MRR@10"],
        "evidence_recall_at_10": official["custom"]["Evidence Recall@10"],
        "complete_evidence_coverage_at_10": official["custom"]["Complete Evidence Coverage@10"],
    }
    evidence_details = []
    null_details = []
    candidate_sizes = []
    worst_values = []
    counts = defaultdict(int)
    for query in queries:
        qid = query.query_id
        current = group.rankings[qid]
        original = baseline[qid]["B2"]
        candidate_sizes.append(len(original))
        if not query.is_answerable:
            null_details.append(_null_row(shared, qid, current, original))
            continue
        prior_ranks = evidence_ranks(query.evidence_list, original, bodies)
        current_ranks = evidence_ranks(query.evidence_list, current, bodies)
        original_complete = complete_at_10(prior_ranks)
        current_complete = complete_at_10(current_ranks)
        pool_complete = worst_rank(prior_ranks) is not None
        prior_worst = worst_rank(prior_ranks)
        current_worst = worst_rank(current_ranks)
        if current_worst is not None:
            worst_values.append(current_worst)
        counts["pool_complete"] += pool_complete
        counts["rrf_top10_complete"] += original_complete
        counts["top10_complete"] += current_complete
        counts["pool_complete_and_current_complete"] += pool_complete and current_complete
        counts["recoverable_rrf_failures"] += pool_complete and not original_complete
        counts["rescued_rrf_failures"] += pool_complete and not original_complete and current_complete
        counts["rrf_regressions"] += original_complete and not current_complete
        for evidence, before, after in zip(query.evidence_list, prior_ranks, current_ranks, strict=True):
            lost = before is not None and before <= 10 and (after is None or after > 10)
            gained = after is not None and after <= 10 and (before is None or before > 10)
            delta = after - before if before is not None and after is not None else None
            severe = before is not None and (after is None or delta >= 10)
            counts["evidence_entries"] += 1
            counts["evidence_lost_from_top10"] += lost
            counts["evidence_gained_into_top10"] += gained
            counts["evidence_worsened_by_at_least_10"] += severe
            evidence_details.append({
                **shared,
                "query_id": qid,
                "question_type": query.question_type,
                "evidence_id": evidence.evidence_id,
                "source_document_id": evidence.source_document_id,
                "in_frozen_pool": before is not None,
                "rrf_rank": before,
                "rank": after,
                "rank_change": delta,
                "rrf_worst_evidence_rank": prior_worst,
                "worst_evidence_rank": current_worst,
                "lost_from_top10": lost,
                "gained_into_top10": gained,
                "worsened_by_at_least_10": severe,
            })
    # The release has unique annotated facts and resolved sources. This catches
    # disagreements between source-rank metrics and the historical evaluator.
    denominator = official["answerable_queries"]
    observed = _ratio(counts["top10_complete"], denominator)
    if observed != metrics["complete_evidence_coverage_at_10"]:
        raise ValueError(f"Historical and rank-based coverage disagree: {shared}")
    metrics.update({
        "candidate_count_mean": _average(candidate_sizes),
        "candidate_pool_evidence_coverage": _ratio(counts["pool_complete"], denominator),
        "candidate_pool_complete_queries": counts["pool_complete"],
        "rrf_top10_complete_queries": counts["rrf_top10_complete"],
        "top10_complete_queries": counts["top10_complete"],
        "reranking_recovery_rate": _ratio(counts["pool_complete_and_current_complete"], counts["pool_complete"]),
        "recoverable_rrf_failure_queries": counts["recoverable_rrf_failures"],
        "rescued_rrf_failure_queries": counts["rescued_rrf_failures"],
        "rrf_failure_rescue_rate": _ratio(counts["rescued_rrf_failures"], counts["recoverable_rrf_failures"]),
        "rrf_regression_queries": counts["rrf_regressions"],
        "ranking_regression_rate": _ratio(counts["rrf_regressions"], counts["rrf_top10_complete"]),
        "evidence_entries": counts["evidence_entries"],
        "evidence_lost_from_top10": counts["evidence_lost_from_top10"],
        "evidence_gained_into_top10": counts["evidence_gained_into_top10"],
        "evidence_worsened_by_at_least_10": counts["evidence_worsened_by_at_least_10"],
        "queries_with_all_evidence_ranked": len(worst_values),
        "worst_evidence_rank_median_when_all_ranked": median(worst_values) if worst_values else None,
    })
    return metrics, evidence_details, null_details


def _null_row(shared, query_id, rows, original):
    first = rows[:10]
    scores = [row.score for row in first]
    docs = [row.document_id for row in first]
    return {
        **shared,
        "query_id": query_id,
        "candidate_count": len(original),
        "returned_count": len(first),
        "top1_document_id": docs[0] if docs else None,
        "top10_document_ids": json.dumps(docs),
        "top10_scores": json.dumps(scores),
        "top1_score": scores[0] if scores else None,
        "top10_score_min": min(scores) if scores else None,
        "top10_score_median": median(scores) if scores else None,
        "top10_score_mean": _average(scores),
        "top10_shared_with_rrf": len(set(docs) & {row.document_id for row in original[:10]}),
    }


def _null_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("split", "run_id", "cohort_sha256", "method"))].append(row)
    summaries = []
    for (split, run_id, cohort, method), items in sorted(groups.items()):
        scores = [row["top1_score"] for row in items if row["top1_score"] is not None]
        top10 = [score for row in items for score in json.loads(row["top10_scores"])]
        summaries.append({
            "split": split, "run_id": run_id, "cohort_sha256": cohort, "method": method,
            "queries": len(items),
            "candidate_count_mean": mean(row["candidate_count"] for row in items),
            "returned_count_mean": mean(row["returned_count"] for row in items),
            "top10_shared_with_rrf_mean": mean(row["top10_shared_with_rrf"] for row in items),
            "top1_score_min": min(scores) if scores else None,
            "top1_score_median": median(scores) if scores else None,
            "top1_score_max": max(scores) if scores else None,
            "top10_scores_min": min(top10) if top10 else None,
            "top10_scores_median": median(top10) if top10 else None,
            "top10_scores_max": max(top10) if top10 else None,
        })
    return summaries


def _write_csv(path, rows, fields):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--heldout-baselines", action="store_true")
    args = parser.parse_args()
    if args.results and args.baseline_only:
        parser.error("--results and --baseline-only are mutually exclusive")
    if args.baseline_only:
        result_paths = ()
    elif args.results:
        result_paths = args.results
    else:
        default = OUTPUT / "reranking_results.jsonl"
        result_paths = (default,) if default.is_file() else ()
    if any(not path.resolve().is_relative_to(ROOT) for path in result_paths):
        parser.error("Results must be inside the official project directory")

    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    queries = {query.query_id: query for query in dataset.queries}
    splits = _load_splits(OUTPUT / "splits.json", queries)
    baseline = _baseline_rankings(dataset, OUTPUT / "candidate_pools.jsonl")
    reranked = _read_groups(result_paths, splits, baseline)
    groups = _with_baselines(reranked, splits, baseline, args.heldout_baselines)
    bodies = {doc.document_id: _normalize(doc.text) for doc in dataset.documents}
    metrics, evidence, nulls = [], [], []
    for group in groups:
        group_metrics, group_evidence, group_nulls = _metrics_and_details(
            group, queries, dataset.documents, bodies, baseline, len(splits[group.split]),
        )
        metrics.append(group_metrics)
        evidence.extend(group_evidence)
        nulls.extend(group_nulls)
    summary = _null_summary(nulls)
    _write_csv(OUTPUT / "reranking_metrics.csv", metrics, list(metrics[0]))
    _write_csv(OUTPUT / "evidence_rank_changes.csv", evidence, EVIDENCE_FIELDS)
    _write_csv(OUTPUT / "null_rankings.csv", nulls, NULL_FIELDS)
    _write_csv(OUTPUT / "null_retrieval_summary.csv", summary, list(summary[0]))
    print(
        f"{len(metrics)} evaluation groups, {len(evidence)} evidence comparisons, "
        f"{len(nulls)} no-answer rankings. Reranker groups: {len(reranked)}."
    )


if __name__ == "__main__":
    main()
