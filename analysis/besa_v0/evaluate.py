"""Offline, Gold-side evaluation of fixed BESA-v0 rankings on development only."""

from collections import Counter, defaultdict
from itertools import product
import csv
from statistics import mean
from time import perf_counter

import numpy as np

from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize, evaluate
from hoprag.models import RetrievalResult
from analysis.reranking_optimization.evaluate import transition_metrics

from .experiment import BUDGETS, ROOT, development_inputs, read, rows, write, write_rows
from .run import OUTPUT, _chunks
from .scoring import DenseVectors
from .experiment import cache_source
from .pipeline import MODES, assemble as _execute, candidate_inputs

# Declare the trade-off grid before loading any Validation labels.
THRESHOLDS = (-0.5, 0.0, 0.5)
NEED_WEIGHTS = (0.25, 0.55, 0.85)
REPEAT_WEIGHTS = (0.1, 0.25)


def csv_write(path, data):
    data = list(data)
    with path.open("w", encoding="utf-8", newline="") as stream:
        columns = list(dict.fromkeys(key for row in data for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def choose_variant(metric):
    """Same-subset coverage first, lower regression then lower inference budget."""
    return (
        -metric["complete_coverage_at10"], metric["regressed_queries"],
        metric["mean_pairs"], metric["budget"], metric["config_id"],
    )


def _stats(score):
    if score is None:
        return {"pair_count": 0, "cache_hits": 0, "new_pairs": 0, "inference_seconds": 0.0,
                "parents_scored": 0, "needs_scored": 0, "truncated": 0}
    values = score["pairs"]
    return {
        "pair_count": score["actual_pairs"], "cache_hits": score["cache_hits"],
        "new_pairs": score["new_pairs"], "inference_seconds": score["inference_seconds"],
        "parents_scored": score["scored_parents"], "needs_scored": score["scored_needs"],
        "truncated": score["truncated_pairs"],
        "scored_chunks": [pair["chunk_id"] for pair in values],
        "cache_sources": dict(Counter(pair["cache_source"] for pair in values)),
    }


def _known_chunks(item, document_id, score):
    if score is None:
        return [item["witnesses"][document_id]]
    winners = {
        cell["chunk_id"]
        for support in score["supports"].values()
        for candidate, cell in support.items()
        if candidate == document_id and cell["support"] > 0
    }
    return sorted(winners) if winners else [item["witnesses"][document_id]]


def _facts(item, query, selected, score):
    ranking = selected["ranking"]
    by_rank = {d: i for i, d in enumerate(ranking, 1)}
    old = {d: i for i, d in enumerate(item["baseline"], 1)}
    chosen = {d: _known_chunks(item, d, score) for d in ranking[:10]}
    scores_docs = {pair["document_id"] for pair in score["pairs"]} if score else set()
    scores_ids = {pair["chunk_id"] for pair in score["pairs"]} if score else set()
    facts = []
    for f in query.evidence_list:
        doc_id = f.source_document_id
        literal = _normalize(f.evidence_text)
        if not doc_id:
            raise ValueError(f"Unmapped Gold source in benchmark: {f.evidence_id}")
        parent = item["candidates"].get(doc_id)
        spanning = [] if parent is None else [c.chunk_id for c in parent.chunks if literal in _normalize(c.text)]
        chosen_chunks = chosen.get(doc_id, [])
        facts.append({
            "evidence_id": f.evidence_id, "source_document_id": doc_id, "evidence_text": f.evidence_text,
            "candidate_parent": parent is not None, "in_original_top10": old.get(doc_id, 1000) <= 10,
            "baseline_rank": old.get(doc_id), "final_rank": by_rank.get(doc_id),
            "in_top10": by_rank.get(doc_id, 1000) <= 10,
            "matching_original_chunk_ids": spanning,
            "parent_was_scored": doc_id in scores_docs,
            "fact_chunk_was_scored": bool(scores_ids.intersection(spanning)),
            "retained_chunk_ids": chosen_chunks,
            "literal_fact_in_retained_chunk": bool(set(chosen_chunks).intersection(spanning)),
            "literal_fact_in_baseline_winner": item["witnesses"].get(doc_id) in spanning,
        })
    return facts, chosen


def _classify(final, facts, raw, scored):
    if all(fact["in_top10"] for fact in facts):
        return "COMPLETE"
    if any(not fact["candidate_parent"] for fact in facts):
        return "P5_MISSING_SOURCE"
    if not raw["plan"]["accepted"]:
        return "PLANNER_NOT_ACCEPTED"
    if final["fallback_reason"]:
        return "RUNTIME_FALLBACK:" + final["fallback_reason"]
    missing = [f for f in facts if not f["in_top10"]]
    if scored is None:
        return "SELECTOR_NO_NEEDS"
    if any(f["matching_original_chunk_ids"] and not f["fact_chunk_was_scored"] for f in missing):
        return "BUDGET_CHUNK_NOT_SCORED"
    if any(not f["matching_original_chunk_ids"] for f in missing):
        return "CHUNK_BOUNDARY_OR_LITERAL_MISMATCH"
    covered = {
        (pair["document_id"], pair["chunk_id"]) for pair in scored["pairs"]
    }
    supporting = {
        (doc_id, cell["chunk_id"])
        for need in scored["supports"].values()
        for doc_id, cell in need.items() if cell["support"] > 0
    }
    if any(
        bool(covered.intersection((f["source_document_id"], c) for c in f["matching_original_chunk_ids"]))
        and not supporting.intersection((f["source_document_id"], c) for c in f["matching_original_chunk_ids"])
        for f in missing
    ):
        return "SCORER_DID_NOT_RETAIN_LITERAL_WITNESS"
    return "WITNESSED_SOURCE_NOT_SELECTED"



def _brief_records(query_ids, data, queries, params, *, detailed=False):
    for qid in query_ids:
        item = data[qid]
        query = queries[qid]
        selected, score, seconds = _execute(item, **params)
        ranking = selected["ranking"]
        # Existing release resolves every Gold source to a unique parent. The
        # selected metrics are independently checked with the shared evaluator.
        found = {d for d in ranking[:10]}
        before = set(item["baseline"][:10])
        evidence = query.evidence_list
        flags = [e.source_document_id in found for e in evidence]
        old = [e.source_document_id in before for e in evidence]
        budget = params.get("budget", 0)
        stats = _stats(score)
        ratio = stats["pair_count"] / len(item["pairs"]) if item["pairs"] else 0.0
        is_need = params["mode"] in MODES
        # Smaller prefix cost is reconstructed from its actual score times;
        # prefilter/cache overhead is estimated from the max-prefix profile.
        shared_overhead = (max(0.0, item["raw"]["scorer_wall_seconds"] -
                              sum(p.inference_seconds for p in item["pairs"] if p.cache_source == "new_inference"))
                           if ratio else 0.0)
        row = {
            "query_id": qid, "question_type": query.question_type, "fact_count": len(evidence),
            "planner_accepted": item["raw"]["plan"]["accepted"],
            "fallback_reason": selected["fallback_reason"], "baseline_complete": all(old),
            "complete": all(flags), "recall": sum(flags) / len(flags), "baseline_recall": sum(old) / len(old),
            "budget": budget, **stats, "planner_seconds": item["raw"]["planner_seconds"],
            "selector_wall_seconds": seconds,
            "estimated_pipeline_seconds": seconds + (item["raw"]["planner_seconds"] +
                stats["inference_seconds"] + shared_overhead if is_need else 0.0),
        }
        if detailed:
            facts, chosen = _facts(item, query, selected, score)
            row.update({
                "question": query.question, "plan": item["raw"]["plan"], "original_top10": item["baseline"][:10],
                "final_top10": ranking[:10], "full_ranking": ranking,
                "choices": selected["choices"], "supports": score["supports"] if score else {},
                "selected_chunk_ids": chosen, "facts": facts,
                "failure_diagnostic": _classify(selected, facts, item["raw"], score),
                "top10_titles": {doc: item["candidates"][doc].title for doc in {*item["baseline"][:10], *ranking[:10]}},
            })
        yield row


def summarize(rows_, **labels):
    rows_ = list(rows_)
    transitions = transition_metrics((r["baseline_complete"] for r in rows_), (r["complete"] for r in rows_))
    pairs = sum(r["pair_count"] for r in rows_)
    return {
        **labels, "queries": len(rows_), **transitions,
        "complete_coverage_at10": mean(r["complete"] for r in rows_),
        "evidence_recall_at10": mean(r["recall"] for r in rows_),
        "planner_acceptance_rate": mean(r["planner_accepted"] for r in rows_),
        "runtime_fallbacks": sum(r["fallback_reason"] is not None for r in rows_),
        "mean_pairs": pairs / len(rows_), "cache_hits": sum(r["cache_hits"] for r in rows_),
        "new_pairs": sum(r["new_pairs"] for r in rows_),
        "cache_hit_rate": sum(r["cache_hits"] for r in rows_) / pairs if pairs else None,
        "mean_inference_seconds": mean(r["inference_seconds"] for r in rows_),
        "estimated_mean_pipeline_seconds": mean(r["estimated_pipeline_seconds"] for r in rows_),
        "estimated_p95_pipeline_seconds": float(np.percentile([r["estimated_pipeline_seconds"] for r in rows_], 95)),
    }


def _configs():
    for mode, budget, threshold, weight in product(MODES, BUDGETS, THRESHOLDS, NEED_WEIGHTS):
        repeat_values = (0.0,) if mode == "C" else REPEAT_WEIGHTS
        for repeat in repeat_values:
            config_id = f"{mode}-B{budget}-t{threshold:+.1f}-n{weight:.2f}-r{repeat:.2f}"
            yield config_id, dict(mode=mode, budget=budget, threshold=threshold, need_weight=weight, repeat_weight=repeat)


def tune(query_ids, data, queries):
    summaries = []
    options = {}
    for config_id, params in _configs():
        summaries.append(summarize(
            _brief_records(query_ids, data, queries, params), config_id=config_id, **params,
        ))
        options[config_id] = params
    per_arm = {}
    for mode, budget in product(MODES, BUDGETS):
        best = min((r for r in summaries if r["mode"] == mode and r["budget"] == budget), key=choose_variant)
        per_arm[f"{mode}-B{budget}"] = {"config_id": best["config_id"], **options[best["config_id"]]}
    final = min((r for r in summaries if r["mode"] in ("C", "D", "E")), key=choose_variant)
    write(OUTPUT / "tune_freeze.json", {
        "scope": "271 Tune questions only", "selection_order": ["complete_coverage", "lower_regressions", "lower_pairs", "budget", "stable_config_id"],
        "selected_arm": f"{final['mode']}-B{final['budget']}", "frozen": per_arm,
        "tune_query_ids": query_ids,
    })
    csv_write(OUTPUT / "tune_grid.csv", summaries)
    return per_arm, f"{final['mode']}-B{final['budget']}"


def evaluate_frozen(query_ids, data, queries, documents, frozen):
    metrics = []
    subsets = []
    detailed = {}
    methods = {"A": {"mode": "A"}, "B": {"mode": "B"}, **{
        label: {k: v for k, v in params.items() if k != "config_id"} for label, params in frozen.items()
    }}
    for method, params in methods.items():
        result = list(_brief_records(query_ids, data, queries, params, detailed=True))
        detailed[method] = result
        for label, ids in (("tune", set(read(OUTPUT / "development_split.json")["tune"])),
                           ("validation", set(read(OUTPUT / "development_split.json")["validation"])),
                           ("full_development", set(query_ids))):
            list_subset = [r for r in result if r["query_id"] in ids]
            if not list_subset:
                continue
            row = summarize(list_subset, method=method, split=label, **params)
            chosen = [queries[r["query_id"]] for r in list_subset]
            rankings = {
                r["query_id"]: [RetrievalResult(r["query_id"], d, method, 0.0, i)
                                for i, d in enumerate(r["full_ranking"], 1)]
                for r in list_subset
            }
            comparison = evaluate(chosen, documents, rankings)
            custom = comparison["custom"]
            if (abs(row["complete_coverage_at10"] - custom["Complete Evidence Coverage@10"]) > 1e-12
                    or abs(row["evidence_recall_at10"] - custom["Evidence Recall@10"]) > 1e-12):
                raise ValueError(f"Shared evaluator disagrees on {method} {label}")
            facts = [f for r in list_subset for f in r["facts"]]
            row.update({
                "candidate_complete_queries": sum(all(f["candidate_parent"] for f in r["facts"]) for r in list_subset),
                "gold_facts": len(facts),
                "gold_facts_source_in_pool": sum(f["candidate_parent"] for f in facts),
                "gold_facts_source_top10": sum(f["in_top10"] for f in facts),
                "gold_facts_retained_chunk_literal": sum(f["literal_fact_in_retained_chunk"] for f in facts),
                "gold_facts_parent_not_scored_on_attempt": sum(
                    f["candidate_parent"] and not f["parent_was_scored"] for r in list_subset if r["pair_count"] for f in r["facts"]
                ),
                "gold_facts_match_chunk_not_scored_on_attempt": sum(
                    bool(f["matching_original_chunk_ids"]) and not f["fact_chunk_was_scored"]
                    for r in list_subset if r["pair_count"] for f in r["facts"]
                ),
            })
            metrics.append(row)
            for name, key in (("question_type", "question_type"), ("fact_count", "fact_count"), ("planner_accepted", "planner_accepted")):
                groups = defaultdict(list)
                for r in list_subset:
                    groups[r[key]].append(r)
                for value, group in sorted(groups.items()):
                    subsets.append(summarize(group, method=method, split=label, dimension=name, value=value))
    return metrics, subsets, detailed


def main():
    split = read(OUTPUT / "development_split.json")
    ordered_ids = sorted(split["tune"] + split["validation"])
    records = {r["query_id"]: r for r in rows(OUTPUT / "development_scores.jsonl")}
    if set(records) != set(ordered_ids) or len(records) != 451 or set(split["tune"]) & set(split["validation"]):
        raise ValueError("Incomplete or overlapping development inputs")
    data_source = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    # Retain annotations only for the development queries. Selection never sees these objects.
    docs = data_source.documents
    queries = {q.query_id: q for q in data_source.queries if q.query_id in records}
    vectors = DenseVectors.load(cache_source(), _chunks())
    inputs = []
    for pool, p5b, witnesses in development_inputs(ordered_ids):
        inputs.append((pool, p5b, witnesses, records[pool.query_id]))
    data = {pool.query_id: candidate_inputs(pool, baseline, witnesses, record, vectors)
            for pool, baseline, witnesses, record in inputs}
    frozen, selected = tune(split["tune"], data, {q: queries[q] for q in split["tune"]})
    metrics, subsets, detailed = evaluate_frozen(ordered_ids, data, queries, docs, frozen)
    csv_write(OUTPUT / "metrics.csv", metrics)
    csv_write(OUTPUT / "subgroups.csv", subsets)
    write_rows(OUTPUT / "comparisons.jsonl", (
        {"method": method, **r} for method, method_rows in detailed.items() for r in method_rows
    ))
    write_rows(OUTPUT / "selected_results.jsonl", detailed[selected])
    decisions = [{
        "method": method, "query_id": r["query_id"],
        "split": "tune" if r["query_id"] in split["tune"] else "validation",
        "transition": "recovered" if r["complete"] else "regressed",
    } for method, method_rows in detailed.items() for r in method_rows if r["baseline_complete"] != r["complete"]]
    csv_write(OUTPUT / "transitions.csv", decisions)
    write(OUTPUT / "evaluation_summary.json", {
        "selected_arm": selected, "development_queries": len(data), "grid_size": len(list(_configs())),
        "planner_fallbacks": dict(Counter(r["fallback_reason"] for r in records.values() if r["fallback_reason"])),
        "selected_failures": dict(Counter(r["failure_diagnostic"] for r in detailed[selected] if not r["complete"])),
        "processing_note": "A/B use the historical Jina winner chunk as witness. Need arms keep positive-support winners, falling back per document to the historical winning chunk. Literal matches are offline diagnostics.",
        "timing_note": "Directly measured maximum-prefix scoring; estimated prefix wall-time by adding recorded new-pair inference, shared scheduling/cache overhead and measured selector time. No re-inference for ablations.",
    })
    print(f"Frozen selected arm {selected}; evaluated {len(ordered_ids)} development queries", flush=True)


if __name__ == "__main__":
    main()

