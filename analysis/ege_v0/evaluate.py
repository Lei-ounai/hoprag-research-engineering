"""Tune on the frozen Tune split and evaluate saved retrieval traces offline."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
from statistics import mean
from time import perf_counter
import csv

import numpy as np

from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize, evaluate
from hoprag.models import RetrievalResult
from hoprag.retrievers import _tokens
from analysis.besa_v0.experiment import read, rows, write, write_rows
from analysis.reranking_optimization.evaluate import transition_metrics

from .expansion import LexicalIndex
from .ranking import ScoreBatch, fuse
from .run import DATA, LEXICAL, OUT, ROOT, SPLIT, JINA_PARENTS, corpus, ordered_novel, winner_run


WEIGHTS = (0.88, 0.92, 0.96, 1.0)
ANCHORS = (0, 5, 7, 8)
JINA_STRENGTHS = (0.10, 0.20, 0.40)
DEFAULT_WEIGHT = 0.92


def write_csv(path, records):
    records = list(records)
    keys = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=keys)
        output.writeheader()
        output.writerows(records)


def evidence(query):
    keys = {(e.source_document_id, _normalize(e.evidence_text)) for e in query.evidence_list}
    if not keys or any(doc is None for doc, _ in keys):
        raise ValueError(f"Need resolved development facts: {query.query_id}")
    return keys


def coverage(keys, ranking):
    ids = set(ranking)
    return sum(doc in ids for doc, _ in keys) / len(keys)


def make_choice(variant, hops, weight=DEFAULT_WEIGHT, anchor=0):
    return {"variant": variant, "hops": hops, "mode": "anchor" if anchor else "rrf",
            "expansion_weight": weight, "anchor": anchor}


def get_run(row, choice):
    variant = choice["variant"]
    if variant == "question_only":
        return row["question_only"]
    if variant == "baseline":
        return None
    if variant.startswith("fixed"):
        depth = variant.removeprefix("fixed")
        baseline = set(row["baseline"])
        novel = [doc for doc in row["fixed"][depth] if doc not in baseline]
        return {"new_candidates": novel, "queries": [], "candidate_document_ids": [*row["baseline"], *novel],
                "max_hop_reached": 0, "bm25_seconds": 0.0, "wall_seconds": row["fixed_seconds"],
                "fallback_reason": None}
    return winner_run(row, choice)


def rank_row(row, choice, run, batch=None):
    baseline = row["baseline"]
    scores = {r["document_id"]: r["score"] for r in row["jina"]}
    witnesses = {r["document_id"]: r["winning_chunk_id"] for r in row["jina"]}
    new_witnesses = {}
    if run is None:
        novel = []
    elif choice["variant"].startswith("fixed"):
        novel = run["new_candidates"]
    else:
        novel = ordered_novel(run)
        new_witnesses = {r["document_id"]: r["chunk_id"] for r in run["new_candidates"]}
    return fuse(baseline, novel, mode=choice["mode"], expansion_weight=choice["expansion_weight"],
                anchor=choice["anchor"], score_batch=batch, baseline_scores=scores,
                baseline_witnesses=witnesses, expansion_witnesses=new_witnesses,
                jina_strength=choice.get("jina_strength", 0.0))


def choice_key(result):
    return (-result["final_complete"], result["regressed_queries"], -result["recovery_queries"],
            -result["candidate_complete"], result["mean_candidates"],
            result.get("hops", 0), result.get("jina_strength", 0.0),
            result["anchor"], result["expansion_weight"], result["variant"])


def tune_choice(choices, data, gold, *, batches=None):
    summaries = []
    for choice in choices:
        complete = regressions = recoveries = cand_complete = candidates = 0
        for qid, row in data.items():
            keys = gold[qid]
            run = get_run(row, choice)
            ranked = rank_row(row, choice, run, batches.get(qid) if batches else None)
            original_complete = coverage(keys, row["baseline"][:10]) == 1
            final_complete = coverage(keys, ranked.document_ids) == 1
            pool = run["candidate_document_ids"] if run else row["baseline"]
            complete += final_complete
            regressions += original_complete and not final_complete
            recoveries += not original_complete and final_complete
            cand_complete += coverage(keys, pool) == 1
            candidates += len(pool)
        summaries.append({**choice, "final_complete": complete, "regressed_queries": regressions,
                          "recovery_queries": recoveries, "candidate_complete": cand_complete,
                          "mean_candidates": candidates / len(data)})
    best = min(summaries, key=choice_key)
    return {key: best[key] for key in choices[0]}, summaries


def seed_tune(data, gold):
    out = []
    for method, hops in product(sorted(LEXICAL), (1, 2)):
        runs = [(winner_run(row, {"variant": method, "hops": hops}), gold[qid]) for qid, row in data.items()]
        recalls = [coverage(keys, run["candidate_document_ids"]) for run, keys in runs]
        out.append({"variant": method, "hops": hops, "complete": recalls.count(1),
                    "recall": mean(recalls), "mean_new": mean(len(r["new_candidates"]) for r, _ in runs)})
    # The retained one-hop arm is the same first-hop prefix as E4, for attribution.
    best = min((r for r in out if r["hops"] == 2),
               key=lambda r: (-r["complete"], -r["recall"], r["mean_new"], r["variant"]))
    return best["variant"], out


def freeze(tune, tune_gold, batches=None):
    path = OUT / "experiment_config.json"
    if batches is not None:
        previous = read(path)
        lexical = previous["lexical_winner"]
        choices = [dict(lexical, mode="jina", jina_strength=strength) for strength in JINA_STRENGTHS]
        active = any(batch is not None and not batch.fallback_reason for batch in batches.values())
        if active:
            best, grid = tune_choice(choices, tune, tune_gold, batches=batches)
            previous["jina_winner"] = best
            write_csv(OUT / "tune_jina_grid.csv", grid)
        write(path, previous)
        return previous

    selected_seed, seed_grid = seed_tune(tune, tune_gold)
    fixed_choice, fixed = tune_choice(
        [make_choice(f"fixed{depth}", 0, weight, anchor)
         for depth, weight, anchor in product((30, 50, 100), WEIGHTS, ANCHORS)], tune, tune_gold)
    controls, control_grid = tune_choice(
        [make_choice("question_only", 1, weight, anchor) for weight, anchor in product(WEIGHTS, ANCHORS)],
        tune, tune_gold)
    by_mode = {}
    lexical_grid = []
    for label, anchors in (("rrf", (0,)), ("anchor", (5, 7, 8))):
        best, scored = tune_choice([make_choice(name, hops, weight, anchor)
                                   for name, hops, weight, anchor in product(LEXICAL, (1, 2), WEIGHTS, anchors)],
                                  tune, tune_gold)
        by_mode[label] = best
        lexical_grid.extend(scored)
    picked = min((r for r in lexical_grid), key=choice_key)
    choice = {key: picked[key] for key in by_mode["rrf"]}
    config = {
        "scope": "271 frozen Tune questions; Validation and null cohorts do not select parameters",
        "split": "analysis/besa_v0/run/development_split.json", "tune_queries": len(tune),
        "variants": {name: asdict(config) for name, config in LEXICAL.items()},
        "candidate_only_fixed_depths": [15, 30, 50, 100],
        "seed_variants": {"h1": selected_seed, "h2": selected_seed},
        "seed_choice": "max Tune candidate-complete then recall, fewer candidates, stable ID; E3 is exact E4 prefix",
        "fixed_winner": fixed_choice, "question_winner": controls, "fusions": by_mode,
        "lexical_winner": choice, "jina_winner": None, "jina_max_new_pairs_per_question": JINA_PARENTS,
        "jina_strength_grid": list(JINA_STRENGTHS), "fusion_weights": list(WEIGHTS), "anchor_sizes": list(ANCHORS),
        "fusion_ties": "more complete, fewer regressions, more recoveries, candidate complete, fewer candidates, "
                      "shallower hops, lower Jina strength, smaller anchor, smaller weight, stable variant",
        "lexical_seed_no_planner": True, "dense_new_query_embeddings": 0,
    }
    if path.exists():
        old = read(path)
        for key in config:
            if key != "jina_winner" and config[key] != old[key]:
                raise ValueError("Reproduced Tune selection differs from frozen lexical configuration")
        return old
    write_csv(OUT / "tune_candidate_grid.csv", seed_grid)
    write_csv(OUT / "tune_fusion_grid.csv", [*fixed, *control_grid, *lexical_grid])
    write(path, config)
    return config


def methods(config):
    return {
        "E0": dict(make_choice("baseline", 0), mode="baseline"),
        "E1": config["fixed_winner"],
        "E2": config["question_winner"],
        "E3": make_choice(config["seed_variants"]["h1"], 1),
        "E4": make_choice(config["seed_variants"]["h2"], 2),
        "E5-RRF": config["fusions"]["rrf"],
        "E5-Anchor": config["fusions"]["anchor"],
        "E5": config["lexical_winner"],
        **({"E6": config["jina_winner"]} if config.get("jina_winner") else {}),
    }


def evaluate_method(label, choice, row, gold, query, batch, prior_failure, index):
    start = perf_counter()
    run = get_run(row, choice)
    final = rank_row(row, choice, run, batch)
    baseline = row["baseline"]
    top, pool = final.document_ids, run["candidate_document_ids"] if run else baseline
    if len(top) != 10 or len(set(top)) != len(top) or not set(top) <= set(pool) or not set(baseline) <= set(pool):
        raise ValueError(f"Invalid merged ranking: {label}/{row['query_id']}")
    if final.fallback_reason and final.ranking != baseline:
        raise ValueError("Fallback does not preserve the complete baseline")
    witnesses = {}
    question_terms = set(_tokens(row["question"]))
    for doc in top:
        witness = final.witnesses.get(doc)
        if witness is None and choice["variant"].startswith("fixed"):
            witness = index.witness(doc, question_terms).chunk_id
        if witness is None or index.chunk_map[witness].document_id != doc:
            raise ValueError(f"No valid original Chunk for {label}/{row['query_id']}/{doc}")
        witnesses[doc] = witness
    ranking_seconds = perf_counter() - start
    keys = gold[query.query_id]
    source_ids = {doc for doc, _ in keys}
    base_recall, candidate_recall, recall = (coverage(keys, value) for value in (baseline[:10], pool, top))
    actual_batch = batch if label == "E6" and batch is not None else ScoreBatch()
    return {
        "method": label, "query_id": query.query_id, "question_type": query.question_type,
        "fact_count": len(keys), "prior_diagnostic": prior_failure.get(query.query_id, ""),
        "baseline_complete": base_recall == 1, "complete": recall == 1, "recall": recall,
        "candidate_recall": candidate_recall, "candidate_complete": candidate_recall == 1,
        "candidate_count": len(pool), "new_candidate_count": len(set(pool) - set(baseline)),
        "new_required_source_ids": sorted(source_ids & (set(pool) - set(baseline))),
        "new_required_top10_source_ids": sorted(source_ids & (set(top) - set(baseline[:10]))),
        "lost_required_top10_source_ids": sorted(source_ids & (set(baseline[:10]) - set(top))),
        "expansion_queries": len(run["queries"]) if run else 0,
        "hops": run["max_hop_reached"] if run else 0,
        "bm25_seconds": run["bm25_seconds"] if run else 0.0, "dense_seconds": 0.0,
        "pipeline_seconds": (run["wall_seconds"] if run else 0.0) + actual_batch.wall_seconds + ranking_seconds,
        "jina_inference_seconds": actual_batch.inference_seconds,
        "jina_new_pairs": actual_batch.new_pairs, "jina_actual_pairs": actual_batch.actual_pairs,
        "jina_history_hits": actual_batch.historical_hits, "jina_local_hits": actual_batch.local_hits,
        "jina_truncated_pairs": actual_batch.truncated_pairs,
        "embedding_requests": 0,
        "candidate_ids": pool, "baseline_top10": baseline[:10], "top10": top, "full_ranking": final.ranking,
        "winner_chunks": witnesses,
        "fallback_reason": final.fallback_reason or (run["fallback_reason"] if run else None),
        "ranking_fallback_reason": final.fallback_reason,
    }


def metrics(records, **labels):
    return {
        **labels, "questions": len(records),
        **transition_metrics((r["baseline_complete"] for r in records), (r["complete"] for r in records)),
        "complete_evidence_coverage_at10": mean(r["complete"] for r in records),
        "evidence_recall_at10": mean(r["recall"] for r in records),
        "candidate_complete_count": sum(r["candidate_complete"] for r in records),
        "candidate_complete_coverage": mean(r["candidate_complete"] for r in records),
        "candidate_evidence_recall": mean(r["candidate_recall"] for r in records),
        "mean_candidates": mean(r["candidate_count"] for r in records),
        "mean_new_candidates": mean(r["new_candidate_count"] for r in records),
        "new_required_sources": sum(len(r["new_required_source_ids"]) for r in records),
        "new_required_top10_sources": sum(len(r["new_required_top10_source_ids"]) for r in records),
        "lost_required_top10_sources": sum(len(r["lost_required_top10_source_ids"]) for r in records),
        "mean_hops": mean(r["hops"] for r in records),
        "mean_expansion_queries": mean(r["expansion_queries"] for r in records),
        "mean_pipeline_seconds": mean(r["pipeline_seconds"] for r in records),
        "p95_pipeline_seconds": float(np.percentile([r["pipeline_seconds"] for r in records], 95)),
        "mean_inference_seconds": mean(r["jina_inference_seconds"] for r in records),
        "p95_inference_seconds": float(np.percentile([r["jina_inference_seconds"] for r in records], 95)),
        "mean_bm25_seconds": mean(r["bm25_seconds"] for r in records),
        "mean_dense_seconds": 0.0,
        "new_jina_pairs": sum(r["jina_new_pairs"] for r in records),
        "jina_actual_pairs": sum(r["jina_actual_pairs"] for r in records),
        "jina_history_hits": sum(r["jina_history_hits"] for r in records),
        "jina_local_hits": sum(r["jina_local_hits"] for r in records),
        "jina_truncated_pairs": sum(r["jina_truncated_pairs"] for r in records),
        "embedding_requests": 0, "fallbacks": sum(r["fallback_reason"] is not None for r in records),
    }


def verify_shared(aggregate, records, annotations, all_docs, method):
    rank = {
        r["query_id"]: [RetrievalResult(r["query_id"], doc, method, 0.0, i)
                        for i, doc in enumerate(r["top10"], 1)]
        for r in records
    }
    ref = evaluate([annotations[r["query_id"]] for r in records], all_docs, rank)["custom"]
    if abs(ref["Complete Evidence Coverage@10"] - aggregate["complete_evidence_coverage_at10"]) > 1e-12:
        raise ValueError("Existing source evaluator differs in complete coverage")
    if abs(ref["Evidence Recall@10"] - aggregate["evidence_recall_at10"]) > 1e-12:
        raise ValueError("Existing source evaluator differs in evidence recall")


def per_term(label, trace_rows, gold, paths, docs):
    stats = defaultdict(Counter)
    for row, run in trace_rows:
        if run is None or not run["queries"]:
            continue
        baseline, req = set(row["baseline"]), {d for d, _ in gold[row["query_id"]]}
        for call in run["queries"]:
            term = call["term"]
            leader = term["seed_document_id"]
            resolved = {hit["document_id"] for hit in call["hits"] if hit["candidate_status"] == "admitted"}
            paths.append({"query_id": row["query_id"], "method": label, **call})
            key = (label, " ".join(term["term_tokens"]), str(leader or ""), call["hop"])
            values = stats[key]
            values["query_uses"] += 1
            values["df"] = term["term_document_frequency"]
            values["idf"] = term["idf"]
            values["term_kind"] = term["term_kind"]
            values["admitted"] += len(resolved)
            values["required_hits"] += len(resolved & (req - baseline))
            values["other_admitted"] += len(resolved - req)
            values["already_visited_hits"] += sum(hit["candidate_status"] == "already_visited" for hit in call["hits"])
    return [{"method": label, "term": term, "seed_document_id": parent, "hop": hop,
             "seed_title": docs[parent].title if parent else "", **counter}
            for (label, term, parent, hop), counter in sorted(stats.items())]


def load_batches(path):
    return {r["query_id"]: ScoreBatch.from_dict(r["batch"]) if r["batch"] else None for r in rows(path)}


def evaluate_all(data, annotations, gold, docs, batches, config, tune, valid, prior):
    phase = {"tune": tune, "validation": valid, "all": tune | valid}
    aggregates, detail, terms, paths = [], [], [], []
    target_groups = {"p5_gap_46": {qid for qid, cause in prior.items() if cause == "P5_MISSING_SOURCE"},
                     "planner_reject_54": {qid for qid, cause in prior.items() if cause == "PLANNER_NOT_ACCEPTED"},
                     "other_15": {qid for qid, cause in prior.items()
                                  if cause not in {"P5_MISSING_SOURCE", "PLANNER_NOT_ACCEPTED"}}}
    corpus_docs, chunks = corpus()
    index = LexicalIndex(corpus_docs.values(), chunks)
    for label, choose in methods(config).items():
        records = [
            evaluate_method(label, choose, data[qid], gold, annotations[qid], batches.get(qid), prior, index)
            for qid in sorted(data)
        ]
        detail.extend(records)
        for split, wanted in phase.items():
            subset = [r for r in records if r["query_id"] in wanted]
            main = metrics(subset, method=label, split=split, group_type="all", group="all")
            verify_shared(main, subset, annotations, docs.values(), label)
            aggregates.append(main)
            for key in ("question_type", "fact_count"):
                for value in sorted({r[key] for r in subset}):
                    aggregates.append(metrics([r for r in subset if r[key] == value],
                                              method=label, split=split, group_type=key, group=value))
            for name, ids in target_groups.items():
                group = [r for r in subset if r["query_id"] in ids]
                if group:
                    aggregates.append(metrics(group, method=label, split=split,
                                              group_type="historical_failure", group=name))
        if label in {"E2", "E3", "E4", "E5"}:
            terms.extend(per_term(label, ((r, get_run(r, choose)) for r in data.values()), gold, paths, docs))
    write_csv(OUT / "expansion_metrics.csv", aggregates)
    write_csv(OUT / "expansion_term_statistics.csv", terms)
    write_csv(OUT / "final_ranking_comparison.csv", (
        {**{k: v for k, v in r.items() if k not in {"candidate_ids", "full_ranking", "winner_chunks",
                                                 "baseline_top10", "top10", "new_required_source_ids",
                                                 "new_required_top10_source_ids", "lost_required_top10_source_ids"}},
         **{k: " ".join(r[k]) for k in ("baseline_top10", "top10", "new_required_source_ids",
                                         "new_required_top10_source_ids", "lost_required_top10_source_ids")},
         "transition": ("regressed" if r["baseline_complete"] and not r["complete"] else
                        "recovered" if r["complete"] and not r["baseline_complete"] else "unchanged")}
        for r in detail
    ))
    write_rows(OUT / "expansion_paths.jsonl", paths)
    by_query = defaultdict(dict)
    for r in detail:
        by_query[r["query_id"]][r["method"]] = r
    write_rows(OUT / "candidate_expansion_results.jsonl", (
        {
            "query_id": qid, "question": data[qid]["question"], "split": "tune" if qid in tune else "validation",
            "p5b_initial_candidates": data[qid]["baseline"],
            "fixed_union_candidates": data[qid]["fixed"], "lexical_controls": {
                label: get_run(data[qid], choice) for label, choice in methods(config).items()
                if label in {"E2", "E3", "E4", "E5"}
            },
            "offline_comparisons": by_query[qid],
        } for qid in sorted(data)
    ))
    return aggregates


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "final"))
    args = parser.parse_args()
    start = perf_counter()
    split = read(SPLIT)
    tune, valid = set(split["tune"]), set(split["validation"])
    data = {r["query_id"]: r for r in rows(DATA / "development_expansion.jsonl")}
    if data.keys() != tune | valid or len(data) != 451:
        raise ValueError("No complete frozen development snapshot")
    benchmarks = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    ann = {q.query_id: q for q in benchmarks.queries if q.query_id in data}
    if ann.keys() != data.keys() or any(not q.is_answerable for q in ann.values()):
        raise ValueError("Missing or non-answerable offline annotation")
    tune_rows = {qid: data[qid] for qid in sorted(tune)}
    tune_gold = {qid: evidence(ann[qid]) for qid in tune_rows}
    config = freeze(tune_rows, tune_gold)
    if args.stage == "freeze":
        print(config["lexical_winner"], flush=True)
        return
    batch_file = DATA / "development_jina.jsonl"
    batches = load_batches(batch_file) if batch_file.exists() else {}
    if batches and batches.keys() != data.keys():
        raise ValueError("Incomplete E6 snapshot")
    if batches:
        config = freeze(tune_rows, tune_gold, {qid: batches[qid] for qid in tune_rows})
    gold = {qid: evidence(q) for qid, q in ann.items()}
    prior = {r["query_id"]: r["classification"] for r in rows(OUT / "failure_reclassification.jsonl")}
    out = evaluate_all(data, ann, gold, {d.document_id: d for d in benchmarks.documents}, batches, config, tune, valid, prior)
    write(DATA / "evaluation_wall.json", {"wall_seconds": perf_counter() - start})
    base = next(r for r in out if r["method"] == "E0" and r["split"] == "all" and r["group_type"] == "all")
    if base["final_complete"] != 333 or base["candidate_complete_count"] != 405:
        raise ValueError("E0 failed to reproduce the historical scores")
    for r in out:
        if r["group_type"] == "all":
            print(r["method"], r["split"], r["candidate_complete_count"], r["final_complete"],
                  r["recovered_queries"], r["regressed_queries"], flush=True)


if __name__ == "__main__":
    main()

