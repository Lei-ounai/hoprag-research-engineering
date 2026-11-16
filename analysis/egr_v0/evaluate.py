"""Freeze EGR parameters on Tune and compare every arm on recorded local scores."""

from argparse import ArgumentParser
from collections import Counter
from itertools import product
from statistics import mean
from time import monotonic, perf_counter, time
import csv
import json

import numpy as np

from analysis.besa_v0.experiment import read, rows, write, write_rows
from analysis.reranking_optimization.evaluate import transition_metrics
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize, evaluate
from hoprag.models import RetrievalResult

from .run import CLOCK, OUT, ROOT, RUN, check_time, cohorts
from .runtime import competitions, control, fair_scores, gate_config
from .selector import choose
from dataclasses import asdict


QUALITY = (0.60, 0.75, 0.90)
MARGINS = (0.10, 0.35, 0.70)
NOVELTY = (0.80, 0.90)
CAPS = (1, 2, 3)
CONFIG = OUT / "frozen_config.json"


def joined(stage):
    a, b = RUN / f"{stage}_scores.jsonl", RUN / f"{stage}_witnesses.jsonl"
    for record, witness in zip(rows(a), rows(b), strict=True):
        if record["query_id"] != witness["query_id"]:
            raise ValueError("Witness records do not align with fair scores")
        yield record, witness


def facts(query):
    keys = {(x.source_document_id, _normalize(x.evidence_text)) for x in query.evidence_list}
    if not keys or any(doc is None for doc, _ in keys):
        raise ValueError("Expected resolved Gold facts for an answerable question")
    return keys


def recall(keys, ids):
    docs = set(ids)
    return sum(doc in docs for doc, _ in keys) / len(keys)


def arm_configs(frozen):
    for method in ("E3", "E4", "E5"):
        selected = frozen["thresholds"]["E3" if method == "E3" else "E4"]
        for cap in CAPS:
            yield method, cap, gate_config(cap, selected["quality"], selected["margin"], selected["novelty"])


def freeze(tune, annotations, deadline):
    configs = [{"arm": arm, "cap": cap, "quality": q, "margin": m, "novelty": n}
               for arm in ("E3", "E4") for cap, q, m, n in
               product(CAPS, QUALITY, (0.0,) if arm == "E3" else MARGINS, NOVELTY)]
    tune_counts = [Counter() for _ in configs]
    seen = set()
    for record, witness in joined("development"):
        qid = record["query_id"]
        if qid not in tune:
            continue
        check_time(deadline)
        seen.add(qid)
        gold = facts(annotations[qid])
        competition = competitions(record, witness["witnesses"])["fixed"]
        original = recall(gold, record["baseline"][:10]) == 1
        for c, stats in zip(configs, tune_counts, strict=True):
            cfg = gate_config(*(c[key] for key in ("cap", "quality", "margin", "novelty")))
            result = choose(competition, cfg, record_decisions=False)
            complete = recall(gold, result.document_ids) == 1
            stats["complete"] += complete
            stats["recovered"] += not original and complete
            stats["regressed"] += original and not complete
            stats["replacements"] += len(result.replacements)
    if seen != tune:
        raise ValueError("Tune freeze requires all 271 Tune IDs")
    grid = [{**c, **stats} for c, stats in zip(configs, tune_counts, strict=True)]

    def criterion(row):
        return (-row["complete"], row["regressed"], -row["recovered"], row["replacements"],
                row["cap"], -row["quality"], -row["margin"], row["novelty"])

    best = {arm: min((r for r in grid if r["arm"] == arm), key=criterion) for arm in ("E3", "E4")}
    frozen = {
        "tune_ids_sha_source": "analysis/besa_v0/run/development_split.json", "tune_count": len(tune),
        "thresholds": best, "selected_arm": f"E4-B{best['E4']['cap']}",
        "e5_thresholds": "E4 thresholds fixed for a matched one-hop candidate ablation",
        "tie_order": "highest complete, fewer regressions, more recoveries, fewer swaps, lower cap, higher quality and margin, lower overlap",
        "chunk_budget_per_parent": 3, "expansion": "archived R8 E5 s3-q1 first hop",
    }
    if CONFIG.exists() and read(CONFIG) != frozen:
        raise ValueError("Recomputed Tune freeze changed")
    write(CONFIG, frozen)
    csv_write(OUT / "tune_grid.csv", grid)
    return frozen


def csv_write(path, data):
    data = list(data)
    fields = list(dict.fromkeys(field for row in data for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def compact_json(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def ungated_changes(baseline, top, witnesses, scores):
    baseline_ids = set(baseline)
    new = iter(doc for doc in top if doc not in baseline_ids)
    for index, doc in enumerate(baseline, 1):
        if doc in top:
            continue
        contender = next(new)
        best = max(witnesses.get(contender, ()), key=lambda r: (r["jina_score"], r["chunk_id"]), default=None)
        incumbent, challenger = scores.get(doc), scores.get(contender)
        yield {"position": index, "incumbent_document_id": doc, "candidate_document_id": contender,
               "candidate_parent_score": challenger, "candidate_witness_score": best["jina_score"] if best else None,
               "incumbent_score": incumbent, "score_margin": challenger - incumbent if challenger is not None and
               incumbent is not None else None, "champion_chunk_id": best["chunk_id"] if best else None,
               "evidence_excerpt": best["evidence_excerpt"] if best else "", "reason": "ungated_control",
               "new_coverage_keys": []}


def run_arms(record, witness_record, configs, split, gold=None):
    witnesses = witness_record["witnesses"]
    candidates = record["candidates"]
    scores = fair_scores(record)
    baseline = record["baseline"][:10]
    original = recall(gold, baseline) if gold else None
    original_sources = {doc for doc, _ in gold} if gold else set()
    score_stats = record["scoring"]
    prep = perf_counter()
    comps = competitions(record, witnesses)
    preparation_seconds = perf_counter() - prep
    results = []
    for method in ("E0", "E1", "E2"):
        started = perf_counter()
        top = control(record, method)
        control_seconds = perf_counter() - started
        results.append((method, top, list(ungated_changes(baseline, top, witnesses, scores)), [],
                        score_stats["fallback_reason"] if method == "E2" else None,
                        record["baseline"] if method == "E0" else record["fixed"], control_seconds))
    for family, cap, config in configs:
        started = perf_counter()
        out = choose(comps["augmented" if family == "E5" else "fixed"], config)
        results.append((f"{family}-B{cap}", out.document_ids, [asdict(n) for n in out.replacements],
                        [asdict(n) for n in out.decisions], out.fallback_reason,
                        record["candidate_ids"] if family == "E5" else record["fixed"], perf_counter() - started))
        if out.fallback_reason and out.ranking != record["baseline"]:
            raise ValueError("Invalid full P5-B fallback")
    for method, top, replacements, decisions, reason, pool, elapsed in results:
        if (len(top) != 10 or len(set(top)) != 10 or top[:7] != baseline[:7] or not set(top) <= set(pool)
                or len(replacements) != len(set(top) - set(baseline)) or len(replacements) > 3):
            raise ValueError(f"Invalid protected Top 10: {record['query_id']} {method}")
        for entry in replacements:
            added, removed = entry["candidate_document_id"], entry["incumbent_document_id"]
            selected = next((r for r in witnesses.get(added, ())
                             if r["chunk_id"] == entry["champion_chunk_id"]), None)
            entry.update({"new_candidate_lanes": candidates[added]["lanes"], "new_candidate_title": candidates[added]["title"],
                          "witness_status": selected["status"] if selected else "unknown",
                          "witness_reasons": selected["reasons"] if selected else [],
                          "candidate_necessary_offline": added in original_sources if gold else None,
                          "incumbent_necessary_offline": removed in original_sources if gold else None})
        final = recall(gold, top) if gold else None
        candidate_recall = recall(gold, pool) if gold else None
        costs = [x for d in pool for x in score_stats["documents"].get(d, ())]
        model_enabled = method not in ("E0", "E1")
        uses_gate = method.startswith(("E3", "E4", "E5"))
        data = {
            "query_id": record["query_id"], "question": record["question"], "split": split, "method": method,
            "baseline_top10": baseline, "top10": top, "candidate_pool_size": len(pool), "replacements": replacements,
            "fallback_reason": reason, "baseline_complete": original == 1 if gold else None,
            "complete": final == 1 if gold else None, "recall_at10": final,
            "candidate_complete": candidate_recall == 1 if gold else None, "candidate_recall": candidate_recall,
            "recovery": original < 1 and final == 1 if gold else None,
            "regression": original == 1 and final < 1 if gold else None,
            "observable_wrong_replacements": sum(r["incumbent_necessary_offline"] and
                                                 not r["candidate_necessary_offline"] for r in replacements) if gold else None,
            "explicit_conflict_replacements": sum(r["witness_status"] == "conflict" for r in replacements),
            "expansion_only_replacements": sum(not candidates[r["candidate_document_id"]]["in_fixed"] for r in replacements),
            "candidates_assessed": len(decisions),
            "quality_passes": sum(d["quality_pass"] is True for d in decisions),
            "novelty_passes": sum(d["novelty_pass"] is True for d in decisions),
            "margin_passes": sum(d["margin_pass"] is True for d in decisions),
            "actual_jina_pairs_in_scope": len(costs) if model_enabled else 0,
            "new_jina_pair_uses_in_scope": sum(x["cache_source"] == "new_inference" for x in costs) if model_enabled else 0,
            "inference_seconds_shared": score_stats["inference_seconds"] if model_enabled else 0.0,
            "pipeline_seconds": elapsed + (score_stats["wall_seconds"] if model_enabled else 0.0)
                                + (witness_record["wall_seconds"] + preparation_seconds if uses_gate else 0.0),
        }
        yield data, decisions


def summarize(records, method, split, group="all"):
    def total(key):
        return sum(r[key] for r in records)

    gold = split != "null"
    gate_denominator = total("candidates_assessed")
    out = {
        "method": method, "split": split, "group": group, "questions": len(records),
        "replacement_count": sum(len(r["replacements"]) for r in records),
        "replacement_questions": sum(bool(r["replacements"]) for r in records),
        "wrong_replacements_observable": total("observable_wrong_replacements") if gold else None,
        "null_replacements": sum(len(r["replacements"]) for r in records) if not gold else None,
        "explicit_conflict_replacements": total("explicit_conflict_replacements"),
        "expansion_only_replacements": total("expansion_only_replacements"),
        "challenger_decisions": gate_denominator, "quality_gate_pass": total("quality_passes"),
        "quality_gate_pass_rate": total("quality_passes") / gate_denominator if gate_denominator else None,
        "novel_gate_pass": total("novelty_passes"),
        "evidence_gate_pass_rate": total("novelty_passes") / gate_denominator if gate_denominator else None,
        "margin_gate_pass": total("margin_passes"),
        "actual_jina_pairs_in_scope": total("actual_jina_pairs_in_scope"),
        "new_jina_pair_uses_in_scope": total("new_jina_pair_uses_in_scope"),
        "average_ms": mean(r["pipeline_seconds"] for r in records) * 1000,
        "p95_ms": float(np.percentile([r["pipeline_seconds"] for r in records], 95)) * 1000,
        "fallback_questions": sum(bool(r["fallback_reason"]) for r in records),
    }
    if gold:
        out.update({**transition_metrics((r["baseline_complete"] for r in records), (r["complete"] for r in records)),
                    "complete_coverage_at10": mean(r["complete"] for r in records),
                    "evidence_recall_at10": mean(r["recall_at10"] for r in records),
                    "candidate_complete_count": total("candidate_complete"),
                    "candidate_complete_coverage": mean(r["candidate_complete"] for r in records),
                    "candidate_evidence_recall": mean(r["candidate_recall"] for r in records)})
    return out


def compare(tune, validation, null, annotations, all_docs, frozen, deadline):
    summary_rows = {}
    configs = list(arm_configs(frozen))
    log_path, gate_path, candidate_path, witness_path = (
        OUT / name for name in ("replacement_log.jsonl", "gate_decisions.csv", "candidate_competition.csv",
                                "evidence_witnesses.jsonl"))
    temporary = [p.with_suffix(p.suffix + ".tmp") for p in (log_path, gate_path, candidate_path, witness_path)]
    with (temporary[0].open("w", encoding="utf-8", newline="\n") as log,
          temporary[1].open("w", encoding="utf-8", newline="") as decision_stream,
          temporary[2].open("w", encoding="utf-8", newline="") as candidate_stream,
          temporary[3].open("w", encoding="utf-8", newline="\n") as witness_stream):
        decisions_csv = None
        candidates_csv = None
        for stage, expected in (("development", tune | validation), ("null", null)):
            seen = set()
            for record, witnesses in joined(stage):
                check_time(deadline)
                qid = record["query_id"]
                if qid in seen or qid not in expected:
                    raise ValueError("Unexpected scored question")
                seen.add(qid)
                split = ("tune" if qid in tune else "validation") if stage != "null" else "null"
                keys = facts(annotations[qid]) if stage != "null" else None
                for doc in record["candidate_ids"]:
                    datum = record["candidates"][doc]
                    chosen = record["scoring"]["documents"].get(doc, ())
                    row = {"query_id": qid, "split": split, **datum,
                           "source": datum["metadata"]["source"], "document_date": datum["metadata"]["published_at"],
                           "source_url": datum["metadata"]["url"], "chunk_ids": compact_json([x["chunk_id"] for x in chosen]),
                           "fair_jina_score": max((x["jina_score"] for x in chosen), default=None),
                           "scoring_routes": compact_json([x["selection_route"] for x in chosen])}
                    for field in ("metadata", "expansion", "lanes"):
                        row[field] = compact_json(row[field])
                    if candidates_csv is None:
                        candidates_csv = csv.DictWriter(candidate_stream, fieldnames=list(row))
                        candidates_csv.writeheader()
                    candidates_csv.writerow(row)
                    for witness in witnesses["witnesses"].get(doc, ()):
                        witness_stream.write(compact_json({"split": split, **witness}) + "\n")
                for result, decisions in run_arms(record, witnesses, configs, split, keys):
                    method = result["method"]
                    if keys:
                        result["question_type"] = annotations[qid].question_type
                        result["fact_count"] = len(keys)
                    log.write(compact_json(result) + "\n")
                    summary_rows.setdefault(method, []).append(result)
                    for decision in decisions:
                        row = {"query_id": qid, "split": split, "method": method,
                               **{k: compact_json(v) if isinstance(v, tuple) else v for k, v in decision.items()}}
                        if decisions_csv is None:
                            decisions_csv = csv.DictWriter(decision_stream, fieldnames=list(row))
                            decisions_csv.writeheader()
                        decisions_csv.writerow(row)
                if len(seen) % 75 == 0:
                    print(f"offline {stage}: {len(seen)}", flush=True)
            if seen != expected:
                raise ValueError("Offline stage incomplete")
    for src, dest in zip(temporary, (log_path, gate_path, candidate_path, witness_path), strict=True):
        src.replace(dest)

    metrics = []
    for method, records in summary_rows.items():
        for split in ("tune", "validation", "development", "null"):
            subset = [r for r in records if r["split"] == split or (split == "development" and r["split"] != "null")]
            main = summarize(subset, method, split)
            metrics.append(main)
            if split != "null":
                ranking = {r["query_id"]: [RetrievalResult(r["query_id"], doc, method, 0.0, pos)
                                          for pos, doc in enumerate(r["top10"], 1)] for r in subset}
                ref = evaluate([annotations[r["query_id"]] for r in subset], all_docs, ranking)["custom"]
                if (abs(ref["Complete Evidence Coverage@10"] - main["complete_coverage_at10"]) > 1e-12
                        or abs(ref["Evidence Recall@10"] - main["evidence_recall_at10"]) > 1e-12):
                    raise ValueError("Source metric differs from project evaluator")
                for group_key in ("question_type", "fact_count"):
                    for group in sorted({r[group_key] for r in subset}):
                        metrics.append(summarize([r for r in subset if r[group_key] == group],
                                                 method, split, str(group)))
    csv_write(OUT / "experiment_metrics.csv", metrics)
    return [r for r in metrics if r["group"] == "all"]


def main():
    started = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Re-evaluate only saved data under a new offline time limit")
    parser.add_argument("--freeze-only", action="store_true", help="Freeze Tune parameters before the final replay")
    args = parser.parse_args()
    deadline = monotonic() + (7200 if args.replay else read(CLOCK)["deadline_epoch"] - time())
    tune, validation, null = cohorts()
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    annotations = {q.query_id: q for q in dataset.queries if q.query_id in tune | validation}
    if annotations.keys() != tune | validation or not all(q.is_answerable for q in annotations.values()):
        raise ValueError("Evaluation can see only the two frozen answerable development splits")
    frozen = freeze(tune, {q: annotations[q] for q in tune}, deadline)
    if args.freeze_only:
        print(frozen, flush=True)
        return
    results = compare(tune, validation, null, annotations, dataset.documents, frozen, deadline)
    write(RUN / "evaluation_summary.json", {"wall_seconds": perf_counter() - started, "metrics": results})
    base = next(r for r in results if r["method"] == "E0" and r["split"] == "development")
    if base["final_complete"] != 333 or base["candidate_complete_count"] != 405:
        raise ValueError("Frozen P5-B differs from previous rounds")
    for r in results:
        print(r["method"], r["split"], r.get("final_complete"), r["replacement_count"],
              r.get("regressed_queries"), r["quality_gate_pass"], flush=True)


if __name__ == "__main__":
    main()

