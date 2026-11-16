"""Verify EGE outputs and issue a manifest with input and artifact fingerprints."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import time

from analysis.besa_v0.experiment import read, rows, write
from analysis.reranking.pool import load_splits

from .evaluate import methods
from .run import BUDGET_SECONDS, CLOCK, DATA, OUT, ROOT, cohorts, corpus, fingerprint


def profile():
    files = ["pilot", "pilot_jina", "development", "development_jina", "null", "null_jina"]
    return {name: read(DATA / f"{name}_performance.json") for name in files}


def verify_traces(path, expected):
    seen = set()
    count = 0
    documents, blocks_by_doc = corpus()
    chunk_parent = {c.chunk_id: c.document_id for blocks in blocks_by_doc.values() for c in blocks}
    for row in rows(path):
        qid = row["query_id"]
        if qid in seen or qid not in expected:
            raise ValueError(f"Unexpected trace ID: {qid}")
        seen.add(qid)
        baseline = row["baseline"]
        for experiment in (row["question_only"], *row["variants"].values()):
            calls = experiment["queries"]
            novel = experiment["new_candidates"]
            if (experiment["baseline_order"] != baseline or len(calls) > 5 or len(novel) > 16
                    or max((v["hop"] for v in calls), default=0) > 2
                    or len({tuple(v["term"]["term_tokens"]) for v in calls}) != len(calls)):
                raise ValueError(f"Trace budget or duplication error: {qid}")
            new_ids = {x["document_id"] for x in novel}
            if new_ids & set(baseline) or len(novel) != len(new_ids):
                raise ValueError(f"Duplicate expanded parent: {qid}")
            for v in novel:
                if (len(v["path"]) != len(set(v["path"])) or v["document_id"] not in documents
                        or chunk_parent[v["chunk_id"]] != v["document_id"]):
                    raise ValueError(f"Path or Chunk mapping error: {qid}")
            if any(len(x["hits"]) > 7 or x["question"] != row["question"] for x in calls):
                raise ValueError(f"Invalid expansion search provenance: {qid}")
            count += 1
    if seen != expected:
        raise ValueError("Some expected trace IDs are missing")
    return count


def main():
    integrity = read(DATA / "historical_inputs_before.json")
    after = {path: fingerprint(ROOT / path) for path in integrity}
    if after != integrity:
        raise ValueError("Historical inputs changed during the experiment")
    tune, validation, null = cohorts()
    dev = tune | validation
    config = read(OUT / "experiment_config.json")
    expected_methods = set(methods(config))
    counts = Counter()
    seen = set()
    _, chunk_map = corpus()
    valid_witnesses = {doc: {c.chunk_id for c in chunks} for doc, chunks in chunk_map.items()}
    for row in rows(OUT / "candidate_expansion_results.jsonl"):
        qid = row["query_id"]
        if qid in seen or qid not in dev or set(row["offline_comparisons"]) != expected_methods:
            raise ValueError(f"Incomplete comparison: {qid}")
        seen.add(qid)
        for name, result in row["offline_comparisons"].items():
            top = result["top10"]
            rank = result["full_ranking"]
            witnesses = result["winner_chunks"]
            if (len(top) != 10 or len(set(top)) != 10 or rank[:10] != top
                    or not set(top) <= set(result["candidate_ids"]) or set(top) != set(witnesses)
                    or any(chunk not in valid_witnesses[doc] for doc, chunk in witnesses.items())):
                raise ValueError(f"Invalid ranking or missing Chunk: {name}/{qid}")
            if result["ranking_fallback_reason"] and rank != row["p5b_initial_candidates"]:
                raise ValueError(f"Invalid P5-B fallback: {name}/{qid}")
            counts[name] += 1
    if seen != dev:
        raise ValueError("Development comparison incomplete")
    null_diagnostics = [r["query_id"] for r in rows(DATA / "null_diagnostics.jsonl")]
    if len(null_diagnostics) != 301 or set(null_diagnostics) != null:
        raise ValueError("Unanswerable cohort mismatch")
    stage_times = profile()
    for stage, target in (("development", dev), ("null", null)):
        seen_batches = set()
        for row in rows(DATA / f"{stage}_jina.jsonl"):
            batch = row["batch"]
            if (row["query_id"] in seen_batches or row["query_id"] not in target or batch is None
                    or batch["actual_pairs"] > 8 or batch["fallback_reason"]):
                raise ValueError(f"Jina coverage, budget or backend invalid: {stage}")
            seen_batches.add(row["query_id"])
        if seen_batches != target:
            raise ValueError(f"Incomplete Jina batches: {stage}")
    cases = list(rows(OUT / "failure_reclassification.jsonl"))
    if len({r["query_id"] for r in cases}) != 115 or any(r["query_id"] not in dev for r in cases):
        raise ValueError("Incomplete offline failure audit")
    budget = read(CLOCK)
    out = OUT / "experiment_manifest.json"
    now = time()
    if not out.exists() and now > budget["deadline_epoch"]:
        raise TimeoutError("Initial manifest verification exceeded the experiment deadline")
    ended = read(out)["budget"]["first_verification_epoch"] if out.exists() else now
    traced_variants = {
        "development": verify_traces(DATA / "development_expansion.jsonl", dev),
        "null": verify_traces(DATA / "null_expansion.jsonl", null),
    }
    outputs = [p for p in OUT.glob("*") if p.is_file() and p != out]
    outputs.extend(p for p in (OUT / "tests").glob("*.py") if p.is_file())
    outputs.extend(DATA / file for file in (
        "pilot_expansion.jsonl", "pilot_jina.jsonl", "development_expansion.jsonl",
        "development_jina.jsonl", "null_expansion.jsonl", "null_jina.jsonl", "null_diagnostics.jsonl",
        "historical_inputs_before.json", "evaluation_wall.json", "budget_clock.json",
    ))
    outputs.extend(DATA / f"{name}_performance.json" for name in stage_times)
    sums = {key: sum(stage_times[name][key] for name in stage_times if key in stage_times[name])
            for key in ("wall_seconds", "cpu_seconds", "actual_pairs", "new_pairs", "historical_hits",
                        "local_hits", "inference_seconds", "embedding_requests", "dense_seconds")}
    write(out, {
        "protocol": "EGE-v0 all comparisons, recorded Pilot 10, frozen 451 development and 301 unanswerable",
        "timestamp_utc": datetime.fromtimestamp(ended, timezone.utc).isoformat(),
        "cohort_counts": {"tune": len(tune), "validation": len(validation), "null": len(null),
                          "answerable_heldout_run": 0},
        "answerable_heldout_ids_excluded": len(load_splits()["answerable_heldout"]),
        "budget": {"maximum_seconds": BUDGET_SECONDS, "first_verification_epoch": ended,
                   "wall_seconds_start_to_first_verification": ended - budget["started_epoch"],
                   "within_budget": ended <= budget["deadline_epoch"],
                   "sum_successful_scoring_and_expansion_stage_times": sums["wall_seconds"],
                   "last_offline_evaluation_seconds": read(DATA / "evaluation_wall.json")["wall_seconds"]},
        "historical_inputs_sha256_and_size": after, "historical_inputs_unchanged": True,
        "checked_development_rankings": dict(counts), "checked_runtime_traces": traced_variants,
        "failure_reclassification_queries": len(cases),
        "experiment_totals": sums,
        "stage_performance": {name: {k: v for k, v in report.items()
                                     if k not in {"variants", "question_only_config"}}
                              for name, report in stage_times.items()},
        "artifacts": {p.relative_to(OUT).as_posix(): fingerprint(p) for p in sorted(outputs)},
        "reproduction": [
            "python -B -m analysis.ege_v0.audit",
            "python -B -m analysis.ege_v0.run pilot",
            "python -B -m analysis.ege_v0.run development",
            "python -B -m analysis.ege_v0.evaluate freeze",
            "python -B -m analysis.ege_v0.run development-jina",
            "python -B -m analysis.ege_v0.evaluate final",
            "python -B -m analysis.ege_v0.run null",
            "python -B -m analysis.ege_v0.run null-jina",
            "python -B -m analysis.ege_v0.no_answer",
            "python -B -m analysis.ege_v0.verify",
        ],
    })
    print("Verified", sum(counts.values()), "rankings,", sum(traced_variants.values()), "runtime variant traces,",
          len(after), "unchanged historical inputs and", len(null_diagnostics), "null diagnostics")


if __name__ == "__main__":
    main()

