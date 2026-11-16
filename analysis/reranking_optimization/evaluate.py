"""Gold-side evaluation of Round 6 cached, frozen-pool rerankings."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

from analysis.reranking.core import _rank
from analysis.reranking.evaluate import complete_at_10, evidence_ranks, worst_rank
from analysis.reranking.pool import OUTPUT as ROUND5, iter_pools, load_splits
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize, evaluate
from hoprag.models import RetrievalResult

from .policies import POLICIES, WEIGHTS, select, weighted_fuse


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
DATASET = PROJECT / "data" / "raw" / "multihop_rag"
METHODS = ("bm25", "dense", "rrf")
Q0333 = "q-0333"
TOP_K = 10


def _read_json(path):
    return json.loads(path.read_bytes())


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _update_csv(path, rows, split):
    if not rows:
        return
    previous = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            previous = [row for row in csv.DictReader(stream) if row["split"] != split]
    fields = list(dict.fromkeys(key for row in [*rows, *previous] for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*previous, *rows])
    temporary.replace(path)


def _cohort_hash(query_ids):
    return sha256("\n".join(sorted(query_ids)).encode("utf-8")).hexdigest()


def _baseline_ranking(pool, method):
    rank_attr = f"{method}_rank"
    score_attr = f"{method}_score"
    ranked = sorted(
        (candidate for candidate in pool.candidates if getattr(candidate, rank_attr) is not None),
        key=lambda candidate: getattr(candidate, rank_attr),
    )
    return [
        RetrievalResult(
            pool.query_id, candidate.document_id, method,
            getattr(candidate, score_attr), getattr(candidate, rank_attr),
        )
        for candidate in ranked
    ]


def _from_rows(query_id, method, rows):
    return [RetrievalResult(query_id, row["document_id"], method, row["score"], row["rank"])
            for row in rows]


def _load_checkpoints(path, query_ids, *, allowed_subset=None):
    requested = set(query_ids)
    result = {}
    for row in _read_jsonl(path):
        query_id = row["query_id"]
        if query_id not in requested:
            continue
        if query_id in result:
            raise ValueError(f"Duplicate checkpoint for {query_id}: {path}")
        scores = row["scores"]
        if not isinstance(scores, dict) or not scores or not all(
            isinstance(value, (float, int)) and isfinite(value) for value in scores.values()
        ):
            raise ValueError(f"Invalid parent scores: {query_id}")
        result[query_id] = row
    if result.keys() != requested and (not allowed_subset or result.keys() != set(allowed_subset)):
        missing = requested.difference(result)
        raise ValueError(f"{len(missing)} missing checkpoint queries in {path}: {sorted(missing)[:5]}")
    return result


def _load_historical_dev100(query_ids):
    """Read exact archived parent maxima for a no-inference development sensitivity run."""
    requested = set(query_ids)
    result = {}
    for row in _read_jsonl(ROUND5 / "reranking_results.jsonl"):
        if (
            row["split"] == "dev_100" and row["method"] == "B4"
            and row["run_id"] == "jina-unified-256-3defdef1ae34"
            and row["query_id"] in requested
        ):
            query_id = row["query_id"]
            if query_id in result:
                raise ValueError(f"Duplicate historical Jina ranking: {query_id}")
            scores = {item["document_id"]: item["score"] for item in row["ranking"]}
            if len(scores) != len(row["ranking"]):
                raise ValueError(f"Duplicate historical Jina parent: {query_id}")
            result[query_id] = {
                "query_id": query_id, "scores": scores,
                "winning_chunks": {item["document_id"]: item.get("winning_chunk_id") for item in row["ranking"]},
            }
    if result.keys() != requested:
        raise ValueError("Archived Round 5 Jina parent scores do not cover the requested cohort")
    return result


def _selected_ranking(pool, checkpoint, weight, weight_jina=1, k=60, *, exact=False):
    scored = checkpoint["scores"]
    pool_ids = {candidate.document_id for candidate in pool.candidates}
    if not pool_ids <= scored.keys():
        raise ValueError(f"Incomplete scoring for {pool.query_id}: {sorted(pool_ids - scored.keys())}")
    if exact and pool_ids != scored.keys():
        raise ValueError(f"Checkpoint includes scores outside the frozen policy for {pool.query_id}")
    ce = _rank({doc_id: scored[doc_id] for doc_id in pool_ids})
    winning_chunks = checkpoint.get("winning_chunks", {})
    for row in ce:
        row["winning_chunk_id"] = winning_chunks.get(row["document_id"])
    fused = weighted_fuse(pool, ce, weight, weight_jina, k)
    if len(fused) != len(pool_ids) or len(fused) < TOP_K:
        raise ValueError(f"Cannot return ten unique parents for {pool.query_id}")
    return ce, fused


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def transition_metrics(original_complete, final_complete):
    """Compare paired coverage, with denominators over all original successes/failures."""
    pairs = tuple(zip(original_complete, final_complete, strict=True))
    before = sum(old for old, _ in pairs)
    after = sum(new for _, new in pairs)
    regressed = sum(old and not new for old, new in pairs)
    recovered = sum(not old and new for old, new in pairs)
    return {
        "original_complete": before,
        "final_complete": after,
        "regressed_queries": regressed,
        "recovered_queries": recovered,
        "ranking_regression_rate": _ratio(regressed, before),
        "recovery_rate": _ratio(recovered, len(pairs) - before),
        "net_gain": after - before,
    }


def _metrics(queries, documents, rankings):
    values = evaluate(queries, documents, rankings)
    return {
        "hits_at_10": values["official"]["Hits@10"],
        "map_at_10": values["official"]["MAP@10"],
        "mrr_at_10": values["official"]["MRR@10"],
        "evidence_recall_at_10": values["custom"]["Evidence Recall@10"],
        "complete_evidence_coverage_at_10": values["custom"]["Complete Evidence Coverage@10"],
    }


def _candidate_metrics(context, queries, chosen, bodies):
    parent_counts = []
    pair_counts = []
    recalls = []
    fact_distribution = Counter()
    rank_distribution = Counter()
    missed_queries = 0
    for query in queries:
        pool = chosen[query.query_id]
        parent_counts.append(len(pool.candidates))
        pair_counts.append(sum(len(candidate.chunks) for candidate in pool.candidates))
        ranking = [
            RetrievalResult(query.query_id, candidate.document_id, "candidate", candidate.rrf_score, rank)
            for rank, candidate in enumerate(pool.candidates, start=1)
        ]
        ranks = evidence_ranks(query.evidence_list, ranking, bodies)
        found = sum(rank is not None for rank in ranks)
        missed_queries += found < len(ranks)
        recalls.append(found / len(ranks))
        fact_distribution[f"{found}/{len(ranks)}"] += 1
        rank_distribution.update(
            "missing" if rank is None else "1-10" if rank <= 10 else "11-20" if rank <= 20
            else "21-30" if rank <= 30 else "31-50" for rank in ranks
        )
    return {
        **context,
        "candidate_evidence_recall": mean(recalls),
        "candidate_complete_evidence_coverage": (len(queries) - missed_queries) / len(queries),
        "candidate_complete_queries": len(queries) - missed_queries,
        "mean_candidate_documents": mean(parent_counts),
        "mean_candidate_pairs": mean(pair_counts),
        "total_candidate_pairs": sum(pair_counts),
        "facts_found_per_query_distribution": json.dumps(dict(sorted(fact_distribution.items()))),
        "fact_pool_rank_distribution": json.dumps(dict(sorted(rank_distribution.items()))),
    }


def _diagnostics(context, queries, original, final, ce, chosen, bodies):
    before_complete, after_complete = [], []
    per_query, per_fact = [], []
    for query in queries:
        query_id = query.query_id
        before, after = original[query_id], final[query_id]
        old_ranks = evidence_ranks(query.evidence_list, before, bodies)
        new_ranks = evidence_ranks(query.evidence_list, after, bodies)
        old_complete, new_complete = complete_at_10(old_ranks), complete_at_10(new_ranks)
        before_complete.append(old_complete)
        after_complete.append(new_complete)
        before_ids = [row.document_id for row in before[:TOP_K]]
        after_ids = {row.document_id for row in after[:TOP_K]}
        displaced = [doc_id for doc_id in before_ids if doc_id not in after_ids]
        evidence_docs = {item.source_document_id for item in query.evidence_list}
        ce_by_id = {row["document_id"]: row["rank"] for row in ce.get(query_id, ())}
        pool_ids = {candidate.document_id for candidate in chosen[query_id].candidates} if chosen else set()
        lost = 0
        for evidence, old, new in zip(query.evidence_list, old_ranks, new_ranks, strict=True):
            from_top10 = old is not None and old <= TOP_K and (new is None or new > TOP_K)
            lost += from_top10
            per_fact.append({
                **context,
                "query_id": query_id,
                "evidence_id": evidence.evidence_id,
                "source_document_id": evidence.source_document_id,
                "in_candidate_pool": evidence.source_document_id in pool_ids if chosen else None,
                "original_rrf_rank": old,
                "jina_rank": ce_by_id.get(evidence.source_document_id),
                "final_rank": new,
                "rank_change": new - old if old is not None and new is not None else None,
                "original_worst_evidence_rank": worst_rank(old_ranks),
                "final_worst_evidence_rank": worst_rank(new_ranks),
                "lost_from_top10": from_top10,
                "gained_into_top10": new is not None and new <= TOP_K and (old is None or old > TOP_K),
            })
        per_query.append({
            **context,
            "query_id": query_id,
            "question_type": query.question_type,
            "original_complete": old_complete,
            "final_complete": new_complete,
            "regressed": old_complete and not new_complete,
            "recovered": not old_complete and new_complete,
            "original_worst_evidence_rank": worst_rank(old_ranks),
            "final_worst_evidence_rank": worst_rank(new_ranks),
            "necessary_facts_lost_from_top10": lost,
            "original_rrf_top10_document_ids": json.dumps(before_ids),
            "displaced_rrf_top10_document_ids": json.dumps(displaced),
            "displaced_necessary_document_ids": json.dumps([doc for doc in displaced if doc in evidence_docs]),
            "final_top10_document_ids": json.dumps([row.document_id for row in after[:TOP_K]]),
        })
    summary = {
        **transition_metrics(before_complete, after_complete),
        "necessary_facts_lost_from_top10": sum(row["necessary_facts_lost_from_top10"] for row in per_query),
    }
    return summary, per_query, per_fact


def _cost(checkpoints, chosen, throughput):
    query_ids = tuple(chosen)
    pairs = {
        qid: sum(len(candidate.chunks) for candidate in pool.candidates) for qid, pool in chosen.items()
    }
    total_pairs = sum(pairs.values())
    subsets = {
        qid: {candidate.document_id for candidate in pool.candidates} for qid, pool in chosen.items()
    }
    exact_checkpoint = all(checkpoints[qid].get("pair_count") == count
                           and checkpoints[qid]["scores"].keys() == subsets[qid]
                           for qid, count in pairs.items())
    new_by_doc_available = all("new_pairs_by_document" in checkpoints[qid] for qid in query_ids)
    selected_new = None
    if exact_checkpoint and all("new_pairs" in checkpoints[qid] for qid in query_ids):
        selected_new = sum(checkpoints[qid]["new_pairs"] for qid in query_ids)
    elif new_by_doc_available:
        selected_new = sum(
            checkpoints[qid]["new_pairs_by_document"].get(doc_id, 0)
            for qid, ids in subsets.items() for doc_id in ids
        )
    measured = None
    if exact_checkpoint and all("inference_seconds" in checkpoints[qid] for qid in query_ids):
        measured = sum(checkpoints[qid]["inference_seconds"] for qid in query_ids)
    checkpoint_pairs = sum(checkpoints[qid].get("pair_count", 0) for qid in query_ids)
    checkpoint_new = sum(checkpoints[qid].get("new_pairs", 0) for qid in query_ids)
    checkpoint_inference = sum(checkpoints[qid].get("inference_seconds", 0) for qid in query_ids)
    return {
        "mean_pairs_per_query": total_pairs / len(query_ids),
        "total_candidate_pairs": total_pairs,
        "actual_new_candidate_pairs": selected_new,
        "candidate_cache_hit_rate": _ratio(total_pairs - selected_new, total_pairs) if selected_new is not None else None,
        "actual_candidate_inference_seconds": measured,
        "mean_actual_candidate_inference_seconds": _ratio(measured, len(query_ids)) if measured is not None else None,
        "mean_cold_candidate_inference_seconds_estimated": (
            total_pairs / len(query_ids) / throughput if throughput else None
        ),
        "checkpoint_scope_total_pairs": checkpoint_pairs or None,
        "checkpoint_scope_new_pairs": checkpoint_new if checkpoint_pairs else None,
        "checkpoint_scope_inference_seconds": checkpoint_inference if checkpoint_pairs else None,
        "checkpoint_scope_cache_hit_rate": (
            _ratio(checkpoint_pairs - checkpoint_new, checkpoint_pairs) if checkpoint_pairs else None
        ),
    }


def _selection_key(row):
    regression = row["ranking_regression_rate"]
    recovery = row["recovery_rate"]
    estimated = row["mean_cold_candidate_inference_seconds_estimated"]
    return (
        -row["complete_evidence_coverage_at_10"],
        regression if regression is not None else float("inf"),
        -recovery if recovery is not None else float("inf"),
        estimated if estimated is not None else row["mean_pairs_per_query"],
        row["mean_candidate_documents"],
        row["pool_id"],
        row["w_rrf"],
    )


def _q0333(output, facts, regressions, documents, split):
    current = [row for row in facts if row["query_id"] == Q0333]
    if not current:
        return
    previous = []
    with (ROUND5 / "evidence_rank_changes.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["query_id"], row["split"], row["method"]) == (Q0333, "dev_100", "B5_jina"):
                previous.append({
                    "source_document_id": row["source_document_id"],
                    "title": documents[row["source_document_id"]].title,
                    "rrf_rank": int(row["rrf_rank"]),
                    "round5_final_rank": int(row["rank"]),
                })
    _write_json(output / "q0333.json", {
        "split": split,
        "query_id": Q0333,
        "round5_reference": previous,
        "round6_evidence_ranks": current,
        "round6_query_diagnostics": [row for row in regressions if row["query_id"] == Q0333],
    })


def _context(split, query_ids, **fields):
    return {
        "split": split, "query_count": len(query_ids), "cohort_sha256": _cohort_hash(query_ids),
        **fields,
    }


def _cohort_inputs(split, query_ids):
    dataset = load_dataset(DATASET)
    ids = set(query_ids)
    query_by_id = {query.query_id: query for query in dataset.queries if query.query_id in ids}
    if len(ids) != len(query_ids) or query_by_id.keys() != ids:
        raise ValueError(f"Invalid manifest IDs: {split}")
    queries = [query_by_id[qid] for qid in query_ids]
    if any(query.is_answerable != (split != "null_all") for query in queries):
        raise ValueError(f"Unexpected answerability in manifest cohort {split}")
    pools = {pool.query_id: pool for pool in iter_pools(query_ids)}
    for query in queries:
        if query.question != pools[query.query_id].question:
            raise ValueError(f"Historical pool question mismatch: {query.query_id}")
    documents = {doc.document_id: doc for doc in dataset.documents}
    bodies = {doc_id: _normalize(doc.text) for doc_id, doc in documents.items()}
    return queries, pools, documents, bodies


def _candidate_configurations(output, split, query_ids, queries, pools, bodies, policies):
    candidates = {}
    metrics = []
    for policy in policies:
        chosen = {qid: select(pool, policy) for qid, pool in pools.items()}
        candidates[policy] = chosen
        metrics.append(_candidate_metrics(_context(split, query_ids, pool_id=policy), queries, chosen, bodies))
    _update_csv(output / "candidate_pool_metrics.csv", metrics, split)
    return candidates, {row["pool_id"]: row for row in metrics}


def _evaluate_config(context, queries, documents, bodies, original, chosen, checkpoints, throughput, candidate,
                     ranking_records):
    ce, final = {}, {}
    for query_id, pool in chosen.items():
        ce[query_id], rows = _selected_ranking(
            pool, checkpoints[query_id], context["w_rrf"], context["w_jina"], context["fusion_k"],
            exact=context["scoring_source"] == "frozen_checkpoint",
        )
        final[query_id] = _from_rows(query_id, context["method"], rows)
        key = query_id, context["pool_id"]
        if key not in ranking_records:
            ranking_records[key] = {
                "query_id": query_id, "pool_id": context["pool_id"], "scoring_source": context["scoring_source"],
                "rrf_pool_order": [
                    candidate.document_id for candidate in sorted(pool.candidates, key=lambda item: item.rrf_rank)
                ],
                "original_rrf_top10": [row.document_id for row in original[query_id][:TOP_K]],
                "jina_ranking": ce[query_id], "final_top10_by_rrf_weight": {},
            }
        ranking_records[key]["final_top10_by_rrf_weight"][context["w_rrf"]] = rows[:TOP_K]
    result = _metrics(queries, documents.values(), final)
    transition, per_query, per_fact = _diagnostics(context, queries, original, final, ce, chosen, bodies)
    if result["complete_evidence_coverage_at_10"] != transition["final_complete"] / len(queries):
        raise ValueError("Evidence rank diagnostics and historical coverage evaluator disagree")
    cost = _cost(checkpoints, chosen, throughput)
    return ({
        **context, **result, **transition, **cost,
        "candidate_complete_evidence_coverage": candidate["candidate_complete_evidence_coverage"],
        "candidate_evidence_recall": candidate["candidate_evidence_recall"],
        "mean_candidate_documents": candidate["mean_candidate_documents"],
    }, per_query, per_fact)


def _frozen_parameters(path):
    config = _read_json(path)
    if config.get("state") != "frozen":
        raise ValueError(f"Missing frozen experiment configuration: {path}")
    policy, weight_rrf, weight_jina = config["policy"], config["weight_rrf"], config["weight_jina"]
    k = config["rrf_k"]
    for field, name in (("split_sha256", "splits.json"), ("candidate_manifest_sha256", "candidate_pool_manifest.json")):
        if config[field] != sha256((ROUND5 / name).read_bytes()).hexdigest():
            raise ValueError(f"Frozen {field} differs from the preserved Round 5 artifact")
    if policy not in POLICIES or weight_rrf not in WEIGHTS.values() or weight_jina != 1 or k != 60:
        raise ValueError("Frozen parameters are outside the prespecified design")
    return policy, weight_rrf, weight_jina, k


def _development(output, split, query_ids, queries, pools, documents, bodies, checkpoints,
                 eligible, throughput, scoring_source):
    chosen, candidates = _candidate_configurations(output, split, query_ids, queries, pools, bodies, POLICIES)
    original = {qid: _baseline_ranking(pool, "rrf") for qid, pool in pools.items()}
    rows, regressions, facts = [], [], []
    ranking_records = {}
    for policy in POLICIES:
        for weight_id, weight_rrf in WEIGHTS.items():
            context = _context(
                split, query_ids, method=f"{policy}-{weight_id}", pool_id=policy, weight_id=weight_id,
                w_rrf=weight_rrf, w_jina=1, fusion_k=60, scoring_source=scoring_source,
            )
            row, per_query, per_fact = _evaluate_config(
                context, queries, documents, bodies, original, chosen[policy], checkpoints,
                throughput, candidates[policy], ranking_records,
            )
            row["budget_eligible"] = policy in eligible
            rows.append(row)
            regressions.extend(per_query)
            facts.extend(per_fact)
    _write_jsonl(output / f"development_rankings_{split}.jsonl", ranking_records.values())
    _update_csv(output / "development_comparison.csv", rows, split)
    _update_csv(output / "regression_analysis.csv", regressions, split)
    _update_csv(output / "evidence_rank_changes.csv", facts, split)
    _q0333(output, facts, regressions, documents, split)
    recommendation = min((row for row in rows if row["budget_eligible"]), key=_selection_key, default=None)
    _write_json(output / "development_selection.json", {
        **_context(split, query_ids),
        "scoring_source": scoring_source,
        "eligible_pools": list(eligible),
        "selection_order": [
            "maximize complete evidence coverage at 10",
            "minimize ranking regression rate",
            "maximize recovery rate",
            "minimize estimated inference time or pair count",
            "minimize candidate documents",
            "deterministic policy and weight tie break",
        ],
        "original_rrf": _metrics(queries, documents.values(), original),
        "recommendation": recommendation,
    })
    return recommendation


def _heldout(output, split, query_ids, queries, pools, documents, bodies, checkpoints, throughput, frozen):
    original = {qid: _baseline_ranking(pool, "rrf") for qid, pool in pools.items()}
    rows, regressions, facts = [], [], []
    for method in METHODS:
        context = _context(split, query_ids, method=method, scoring_source="original_baseline")
        rankings = {qid: _baseline_ranking(pool, method) for qid, pool in pools.items()}
        transition, per_query, per_fact = _diagnostics(context, queries, original, rankings, {}, None, bodies)
        rows.append({**context, **_metrics(queries, documents.values(), rankings), **transition})
        regressions.extend(per_query)
        facts.extend(per_fact)
    if checkpoints is not None:
        policy, weight_rrf, weight_jina, k = frozen
        chosen, candidates = _candidate_configurations(output, split, query_ids, queries, pools, bodies, (policy,))
        context = _context(
            split, query_ids, method="optimized_jina_rrf", pool_id=policy, w_rrf=weight_rrf,
            w_jina=weight_jina, fusion_k=k, scoring_source="frozen_checkpoint",
        )
        ranking_records = {}
        row, per_query, per_fact = _evaluate_config(
            context, queries, documents, bodies, original, chosen[policy], checkpoints, throughput,
            candidates[policy], ranking_records,
        )
        _write_jsonl(output / "heldout_rankings.jsonl", ranking_records.values())
        rows.append(row)
        regressions.extend(per_query)
        facts.extend(per_fact)
    _update_csv(output / "heldout_results.csv", rows, split)
    _update_csv(output / "regression_analysis.csv", regressions, split)
    _update_csv(output / "evidence_rank_changes.csv", facts, split)
    return rows


def _distribution(scores):
    if not scores:
        return None
    quartiles = quantiles(scores, n=4, method="inclusive") if len(scores) > 1 else [scores[0]] * 3
    return {
        "minimum": min(scores), "q1": quartiles[0], "median": median(scores), "q3": quartiles[2],
        "maximum": max(scores), "mean": mean(scores),
    }


def _no_answer(output, query_ids, pools, checkpoints, frozen, split_manifest):
    rows = []
    split_by_id = {qid: name for name in ("null_dev", "null_heldout") for qid in split_manifest[name]}
    for query_id in query_ids:
        pool = pools[query_id]
        original = _baseline_ranking(pool, "rrf")
        details = {candidate.document_id: candidate for candidate in pool.candidates}
        original_top = [row.document_id for row in original[:TOP_K]]
        output_row = {
            "query_id": query_id,
            "split": split_by_id[query_id],
            "question": pool.question,
            "original_rrf_top10": original_top,
            "original_rrf_top10_details": [
                {"document_id": row.document_id, "rank": row.rank, "score": row.score,
                 "title": details[row.document_id].title, "source": details[row.document_id].article_source}
                for row in original[:TOP_K]
            ],
            "scored": False,
        }
        if checkpoints is not None and query_id in checkpoints:
            policy, w_rrf, w_jina, k = frozen
            chosen = select(pool, policy)
            ce, fused = _selected_ranking(chosen, checkpoints[query_id], w_rrf, w_jina, k, exact=True)
            ce_by_id = {row["document_id"]: row for row in ce}
            final_ids = [row["document_id"] for row in fused[:TOP_K]]
            output_row.update({
                "scored": True,
                "pool_id": policy,
                "jina_top10": ce[:TOP_K],
                "jina_ranking": ce,
                "jina_best_score": ce[0]["score"],
                "jina_score_distribution": _distribution([row["score"] for row in ce]),
                "jina_original_rrf_top10_overlap": len(set(original_top) & {r["document_id"] for r in ce[:TOP_K]}),
                "optimized_original_rrf_top10_overlap": len(set(original_top) & set(final_ids)),
                "optimized_ranking": [
                    {
                        **row,
                        "title": details[row["document_id"]].title,
                        "source": details[row["document_id"]].article_source,
                        "rrf_rank": details[row["document_id"]].rrf_rank,
                        "jina_rank": ce_by_id[row["document_id"]]["rank"],
                        "jina_score": ce_by_id[row["document_id"]]["score"],
                    }
                    for row in fused
                ],
            })
            output_row["optimized_top10"] = output_row["optimized_ranking"][:TOP_K]
        rows.append(output_row)
    _write_jsonl(output / "no_answer_results.jsonl", rows)
    scored = [row for row in rows if row["scored"]]
    summary = {
        **_context("null_all", query_ids),
        "scored_queries": len(scored),
        "unscored_queries": len(rows) - len(scored),
        "jina_best_score_distribution": _distribution([row["jina_best_score"] for row in scored]),
        "mean_jina_top10_overlap_with_rrf": (
            mean(row["jina_original_rrf_top10_overlap"] for row in scored) if scored else None
        ),
        "mean_optimized_top10_overlap_with_rrf": (
            mean(row["optimized_original_rrf_top10_overlap"] for row in scored) if scored else None
        ),
        "interpretation": "Relevance scores and ranks are retrieval diagnostics. No abstention was evaluated.",
    }
    _write_json(output / "no_answer_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("candidate", "development", "heldout", "no-answer"), required=True)
    parser.add_argument("--cohort", choices=("answerable_dev", "dev_100"), default="answerable_dev")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--historical-dev100", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--eligible-pools", choices=POLICIES, nargs="*", default=list(POLICIES))
    parser.add_argument("--throughput", type=float)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--frozen-config", type=Path, default=OUTPUT / "frozen_reranking_config.json")
    args = parser.parse_args(argv)
    if args.throughput is not None and (not isfinite(args.throughput) or args.throughput <= 0):
        parser.error("--throughput must be a positive finite value")
    if not args.output.resolve().is_relative_to(OUTPUT):
        parser.error("Output must be inside analysis/reranking_optimization/")
    if any(path and not path.resolve().is_relative_to(OUTPUT) for path in (args.checkpoint, args.frozen_config)):
        parser.error("Checkpoint and frozen config must be inside analysis/reranking_optimization/")
    if args.historical_dev100 and (args.stage != "development" or args.cohort != "dev_100" or args.checkpoint):
        parser.error("--historical-dev100 applies only to dev_100 without an alternate checkpoint")
    if args.baseline_only and (args.stage not in ("heldout", "no-answer") or args.checkpoint):
        parser.error("--baseline-only applies to heldout/no-answer without a checkpoint")
    if args.stage != "candidate" and not (args.checkpoint or args.historical_dev100 or args.baseline_only):
        parser.error("Provide --checkpoint, --historical-dev100, or --baseline-only")
    if args.stage == "candidate" and args.checkpoint:
        parser.error("Candidate-only metrics require no checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    manifest = load_splits()
    split = {"heldout": "answerable_heldout", "no-answer": "null_all"}.get(args.stage, args.cohort)
    query_ids = tuple(manifest["null_dev"] + manifest["null_heldout"]) if split == "null_all" else tuple(manifest[split])
    if args.stage == "heldout" and len(query_ids) != 1804:
        raise ValueError("Formal heldout evaluation requires all 1,804 original heldout IDs")
    if args.stage == "no-answer" and len(query_ids) != 301:
        raise ValueError("No-answer evaluation requires the original 301 IDs")
    frozen = _frozen_parameters(args.frozen_config) if args.stage in ("heldout", "no-answer") and args.checkpoint else None
    if args.stage == "heldout" and args.checkpoint:
        lock = OUTPUT / "heldout_run_lock.json"
        if not lock.is_file() or _read_json(lock)["frozen_config_sha256"] != sha256(args.frozen_config.read_bytes()).hexdigest():
            raise ValueError("The requested frozen configuration differs from the heldout run lock")
    if args.checkpoint:
        subset = manifest["null_dev"] if args.stage == "no-answer" else None
        checkpoints = _load_checkpoints(args.checkpoint, query_ids, allowed_subset=subset)
    elif args.historical_dev100:
        checkpoints = _load_historical_dev100(query_ids)
    else:
        checkpoints = None
    queries, pools, documents, bodies = _cohort_inputs(split, query_ids)
    if args.stage == "candidate":
        _, outcome = _candidate_configurations(args.output, split, query_ids, queries, pools, bodies, POLICIES)
    elif args.stage == "development":
        outcome = _development(
            args.output, split, query_ids, queries, pools, documents, bodies, checkpoints,
            args.eligible_pools, args.throughput, "round5_archived_parent_scores" if args.historical_dev100
            else "round6_checkpoint",
        )
    elif args.stage == "heldout":
        outcome = _heldout(
            args.output, split, query_ids, queries, pools, documents, bodies, checkpoints, args.throughput, frozen,
        )
    else:
        outcome = _no_answer(args.output, query_ids, pools, checkpoints, frozen, manifest)
    print(json.dumps({
        "stage": args.stage, "cohort": split, "outcome": outcome, "evaluation_seconds": perf_counter() - started,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
