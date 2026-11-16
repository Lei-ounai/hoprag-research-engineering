"""One-off feasibility timing for five reviewed, question-only plans."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from types import SimpleNamespace

from analysis.reranking.backend import FastEmbedBackend, peak_rss_bytes
from analysis.reranking.cache import RerankCache, score_pool
from analysis.reranking.core import TEMPLATE, aggregate, make_pairs
from analysis.reranking.pool import iter_pools, load_splits, read_manifest
from analysis.reranking_optimization.policies import select
from analysis.reranking_optimization.runner import _lower_priority, identity

OUTPUT = Path(__file__).resolve().parent
ANALYSIS = OUTPUT.parent
HISTORY = ANALYSIS / "reranking"
MODEL_SPEC = HISTORY / "model_jina-unified-256-3defdef1ae34.json"
BATCH_SIZE = 16
THREADS = 4


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan_samples(path):
    packet = read(path)
    if set(packet) != {"queries"} or not isinstance(packet["queries"], list) or len(packet["queries"]) != 5:
        raise ValueError("Timing requires five manually reviewed questions")
    samples = {}
    for query in packet["queries"]:
        if set(query) != {"query_id", "question", "needs"}:
            raise ValueError("Only query_id, question, and needs may be used")
        qid, question, needs = query["query_id"], query["question"], query["needs"]
        if not question.strip() or qid in samples or not 2 <= len(needs) <= 4:
            raise ValueError(f"Invalid reviewed plan: {qid}")
        seen = set()
        for need in needs:
            keys = {"need_id", "need_text", "need_role", "depends_on", "confidence"}
            if set(need) != keys or not need["need_text"].strip() or not need["need_role"].strip():
                raise ValueError(f"Invalid need record: {qid}")
            if not isinstance(need["depends_on"], list) or set(need["depends_on"]) - seen:
                raise ValueError(f"Dependencies must reference earlier needs: {qid}")
            if need["need_id"] in seen:
                raise ValueError(f"Duplicate need ID: {qid}")
            seen.add(need["need_id"])
        if len({need["need_text"] for need in needs}) != len(needs):
            raise ValueError(f"Duplicate need texts: {qid}")
        samples[qid] = (question, needs)
    return samples


def project(pair_rate, r6_manifest, observed_mean):
    inventory = r6_manifest["initial_preflight"]
    scenarios = {}
    for scope in ("answerable_dev", "answerable_heldout"):
        pairs = inventory[scope]["P5"]["total_pairs"]
        scenarios[scope] = {
            "queries": inventory[scope]["P5"]["queries"],
            "original_question_chunk_pairs": pairs,
            "cold_context": "all original P5 chunks per need; independent cache initially empty",
            "assume_needs": {
                str(n): {"chunk_pairs": pairs * n, "seconds": pairs * n / pair_rate}
                for n in (2, observed_mean, 3, 4)
            },
        }
    return scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=OUTPUT / "scoring_reviewed_plans.json")
    args = parser.parse_args()
    if not args.plans.resolve().is_relative_to(OUTPUT):
        parser.error("Reviewed inputs must be in the feasibility directory")
    plans = plan_samples(args.plans)
    splits = load_splits()
    if not set(plans) <= set(splits["answerable_dev"]):
        raise ValueError("Only development samples are allowed")
    read_manifest(verify=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    _lower_priority()
    expected = read(MODEL_SPEC)
    frozen = read(ANALYSIS / "reranking_optimization" / "frozen_reranking_config.json")
    if (frozen["policy"] != "P5" or frozen["template"] != TEMPLATE
            or frozen["model"] != identity(expected)):
        raise ValueError("Historical model/config identity changed")
    start = perf_counter()
    cpu_start = process_time()
    model = FastEmbedBackend(
        expected["model_id"], HISTORY / "model_cache", threads=THREADS,
        context_mode="unified", batch_size=BATCH_SIZE, unified_limit=256,
    )
    if identity(model.metadata) != identity(expected):
        raise ValueError("Loaded Jina differs from the frozen local model")
    rows = []
    counters = Counter()
    sample_started = perf_counter()
    cache_path = OUTPUT / "experiment_cache" / "need_scores.sqlite3"
    with RerankCache(cache_path, model.metadata) as cache:
        for original in iter_pools(plans):
            question, needs = plans[original.query_id]
            if question != original.question:
                raise ValueError(f"Question differs from the frozen scoring pool: {original.query_id}")
            chosen = select(original, "P5")
            pairs = make_pairs(chosen)
            if len(chosen.candidates) > 30:
                raise ValueError("Document budget exceeded")
            counters.update({"queries": 1, "needs": len(needs), "documents": len(chosen.candidates)})
            for need in needs:
                query_started = perf_counter()
                proxy = SimpleNamespace(query_id=original.query_id, question=need["need_text"])
                scores, inferred, inference_seconds = score_pool(model, cache, proxy, pairs, BATCH_SIZE)
                rankings = aggregate(chosen, pairs, (score.score for score in scores))
                row = {
                    "query_id": original.query_id, "need_id": need["need_id"],
                    "need_text": need["need_text"], "documents": len(chosen.candidates),
                    "chunk_pairs": len(pairs), "new_pairs": inferred,
                    "cache_hits": len(pairs) - inferred,
                    "truncated_pairs": sum(score.truncated for score in scores),
                    "input_tokens_min": min(score.input_tokens for score in scores),
                    "input_tokens_median": median(score.input_tokens for score in scores),
                    "input_tokens_max": max(score.input_tokens for score in scores),
                    "inference_seconds": inference_seconds, "wall_seconds": perf_counter() - query_started,
                    "ranking": rankings,
                }
                rows.append(row)
                counters.update({key: row[key] for key in (
                    "chunk_pairs", "new_pairs", "cache_hits", "truncated_pairs",
                    "inference_seconds", "wall_seconds",
                )})
    if counters["queries"] != 5 or not all(isfinite(r["score"]) for row in rows for r in row["ranking"]):
        raise ValueError("Unexpected sample result")
    profiles = OUTPUT / "need_scoring_runs.json"
    prior = read(profiles) if profiles.exists() else []
    rate = counters["new_pairs"] / counters["wall_seconds"] if counters["new_pairs"] else None
    reference = rate or next((p["new_pairs_per_scoring_second"] for p in prior if p["new_pairs"]), None)
    r6 = read(ANALYSIS / "reranking_optimization" / "experiment_manifest.json")
    report = {
        **counters, "process_wall_seconds": perf_counter() - start,
        "scoring_wall_seconds": perf_counter() - sample_started, "cpu_seconds": process_time() - cpu_start,
        "peak_working_set_bytes": peak_rss_bytes(), "new_pairs_per_scoring_second": rate,
        "total_need_document_pairs": sum(row["documents"] for row in rows),
        "mean_needs": counters["needs"] / 5, "model": model.metadata,
        "python": os.sys.version, "fastembed": version("fastembed"), "onnxruntime": version("onnxruntime"),
        "template": TEMPLATE, "cache": str(cache_path.relative_to(OUTPUT)),
        "projection": project(reference, r6, counters["needs"] / 5) if reference else None,
    }
    save(profiles, [*prior, report])
    save(OUTPUT / "need_scoring_scores.json", rows)
    print(json.dumps({k: v for k, v in report.items() if k != "model"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
