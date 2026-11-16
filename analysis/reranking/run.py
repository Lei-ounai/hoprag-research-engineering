"""Run local, frozen-candidate Cross-Encoder experiments."""

from argparse import ArgumentParser
from collections import Counter
import csv
import ctypes
from ctypes import wintypes
import json
import os
from random import Random
from pathlib import Path
from time import perf_counter, process_time

from .cache import RerankCache, score_pool
from .core import FINAL_K, FUSION_K, TEMPLATE, aggregate, fuse, make_pairs
from .pool import iter_pools, load_splits


OUTPUT = Path(__file__).resolve().parent
MODEL_CACHE = OUTPUT / "model_cache"
RESULTS = OUTPUT / "reranking_results.jsonl"
BENCHMARK = OUTPUT / "model_benchmark.csv"
DEVELOPMENT = OUTPUT / "development_config.json"
HELDOUT = OUTPUT / "heldout_test_config.json"
MODELS = {
    "minilm": "Xenova/ms-marco-MiniLM-L-6-v2",
    "jina": "jinaai/jina-reranker-v1-turbo-en",
}
SETTINGS = {
    "threads": 2,
    "batch_size": 16,
    "unified_limit": 256,
    "template": TEMPLATE,
    "aggregation": "maximum over all original chunks, one rank per parent",
    "fusion_k": FUSION_K,
    "candidate_branch_depth": 100,
    "top_k": FINAL_K,
    "development_gate_hours": 4,
    "model_ids": MODELS,
    "development_modes": ["minilm:unified", "jina:unified", "jina:native"],
}


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(obj, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    tmp.replace(path)


def pre_register():
    if DEVELOPMENT.exists():
        data = json.loads(DEVELOPMENT.read_bytes())
        if any(data.get(key) != value for key, value in SETTINGS.items()):
            raise ValueError("Development configuration differs from preregistered settings")
    else:
        write_json(DEVELOPMENT, {
            **SETTINGS, "seed": 42, "stages": ["dev_5", "dev_20", "dev_100"],
            "heldout_policy": "Use only after selection and config freeze; no heldout-driven tuning",
        })


def peak_memory_mib():
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (key, ctypes.c_size_t)
            for key in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")
        ]
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not measure(get_current_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize / (1024 * 1024)


def _run_id(backend, model, mode):
    metadata = backend.metadata
    return f"{model}-{mode}-{metadata['effective_max_tokens']}-{metadata['model_sha256'][:12]}"


def _write_stage_results(new_rows, run_id, stage):
    tmp = RESULTS.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as target:
        if RESULTS.exists():
            with RESULTS.open(encoding="utf-8") as existing:
                for line in existing:
                    row = json.loads(line)
                    if (row["run_id"], row["split"]) != (run_id, stage):
                        target.write(line)
        for row in new_rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(RESULTS)


def _record_benchmark(row):
    existing = []
    if BENCHMARK.exists():
        with BENCHMARK.open(encoding="utf-8", newline="") as stream:
            existing = [
                item for item in csv.DictReader(stream)
                if (item["run_id"], item["split"]) != (row["run_id"], row["split"])
            ]
    tmp = BENCHMARK.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerows([*existing, row])
    tmp.replace(BENCHMARK)


def _benchmark_projection(run_id, stage):
    with BENCHMARK.open(encoding="utf-8", newline="") as stream:
        pilots = [row for row in csv.DictReader(stream) if row["split"] == "dev_100"]
    rows = [row for row in pilots if row["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("Finish the 100-query development stage before a full cohort")
    if stage == "answerable_dev":
        unified = {row["model"]: row for row in pilots if row["context_mode"] == "unified"}
        if unified.keys() != set(MODELS.values()):
            raise RuntimeError("Finish both unified 100-query pilots before a complete development comparison")
        hours = sum(float(row["estimated_dev_hours"]) for row in unified.values())
        if rows[0]["context_mode"] != "unified":
            hours += float(rows[0]["estimated_dev_hours"])
        if hours > SETTINGS["development_gate_hours"]:
            raise RuntimeError(f"Complete development comparison projects {hours:.2f}h, above the 4h gate")
    return rows[0]


def _allow_stage(run_id, stage):
    if stage == "answerable_dev":
        _benchmark_projection(run_id, stage)
    if stage in ("answerable_heldout", "null_heldout"):
        frozen = json.loads(HELDOUT.read_bytes())
        if frozen.get("state") != "frozen" or run_id not in frozen.get("permitted_run_ids", ()):
            raise RuntimeError("Heldout Cross-Encoder run is not authorized by a frozen selection")
        if RESULTS.exists():
            with RESULTS.open(encoding="utf-8") as existing:
                for line in existing:
                    record = json.loads(line)
                    if (record["run_id"], record["split"]) == (run_id, stage):
                        raise RuntimeError(f"Heldout {run_id}/{stage} has already been completed")


def _stage(backend, cache, run_id, short_name, stage, query_ids, pair_counts, expected_pairs=None):
    started = perf_counter()
    cpu_started = process_time()
    rows = []
    stats = Counter()
    lengths = []
    models = {
        "minilm": ("B3", "B5_minilm"),
        "jina": ("B4", "B5_jina"),
    }
    ce_method, fused_method = models[short_name]
    for index, pool in enumerate(iter_pools(query_ids), 1):
        pairs = make_pairs(pool)
        scores, actual, model_seconds = score_pool(backend, cache, pool, pairs, SETTINGS["batch_size"])
        stats["total_pairs"] += len(pairs)
        stats["actual_inferred_pairs"] += actual
        stats["inference_seconds"] += model_seconds
        stats["truncated_pairs"] += sum(score.truncated for score in scores)
        lengths.extend(score.input_tokens for score in scores)
        ce = aggregate(pool, pairs, (score.score for score in scores))
        for method, ranking in ((ce_method, ce), (fused_method, fuse(pool, ce))):
            rows.append({
                "run_id": run_id, "split": stage, "method": method,
                "query_id": pool.query_id, "candidate_count": len(pool.candidates),
                "top10": ranking[:FINAL_K], "ranking": ranking,
            })
        if len(query_ids) <= 5 or index % 5 == 0:
            print(f"{run_id} {stage} {index}/{len(query_ids)} queries, "
                  f"{stats['actual_inferred_pairs']} new pairs, {perf_counter()-started:.1f}s", flush=True)

    if len(rows) != 2 * len(query_ids):
        raise ValueError(f"Incomplete cohort {stage}")
    elapsed = perf_counter() - started
    cpu_seconds = process_time() - cpu_started
    if stats["total_pairs"] != (pair_counts[stage] if expected_pairs is None else expected_pairs):
        raise ValueError(f"Candidate pair count changed for {stage}")
    _write_stage_results(rows, run_id, stage)
    actual = stats["actual_inferred_pairs"]
    per_pair = elapsed / actual if actual else None
    if per_pair is None and BENCHMARK.exists():
        with BENCHMARK.open(encoding="utf-8", newline="") as old:
            for row in csv.DictReader(old):
                if (row["run_id"], row["split"]) == (run_id, stage) and int(row["actual_inferred_pairs"]):
                    per_pair = float(row["wall_seconds"]) / int(row["actual_inferred_pairs"])
                    break
    projections = {
        key: round(per_pair * pair_counts[cohort] / 3600, 4) if per_pair is not None else ""
        for key, cohort in (
            ("estimated_dev_hours", "answerable_dev"),
            ("estimated_heldout_hours", "answerable_heldout"),
            ("estimated_null_dev_hours", "null_dev"),
            ("estimated_null_heldout_hours", "null_heldout"),
        )
    }
    metadata = backend.metadata
    bench = {
        "run_id": run_id,
        "split": stage,
        "model": metadata["model_id"],
        "context_mode": metadata["context_mode"],
        "effective_max_tokens": metadata["effective_max_tokens"],
        "queries": len(query_ids),
        **{key: stats[key] for key in ("total_pairs", "actual_inferred_pairs", "truncated_pairs", "inference_seconds")},
        "cache_hits": stats["total_pairs"] - actual,
        "cache_hit_rate": round(1 - actual / stats["total_pairs"], 8),
        "token_min": min(lengths),
        "token_max": max(lengths),
        "token_mean": round(sum(lengths) / len(lengths), 3),
        "load_seconds": metadata["load_seconds"],
        "wall_seconds": round(elapsed, 4),
        "wall_seconds_per_query": round(elapsed / len(query_ids), 4),
        "new_pairs_per_wall_second": round(actual / elapsed, 4),
        "new_pairs_per_inference_second": round(actual / stats["inference_seconds"], 4) if stats["inference_seconds"] else 0,
        "cpu_seconds": round(cpu_seconds, 4),
        "cpu_pct_one_core": round(100 * cpu_seconds / elapsed, 2),
        "cpu_pct_machine": round(100 * cpu_seconds / elapsed / (os.cpu_count() or 1), 2),
        "process_peak_working_set_mib": round(peak_memory_mib(), 2),
        **projections,
    }
    _record_benchmark(bench)
    print(json.dumps(bench, ensure_ascii=True, allow_nan=False), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--context", choices=("unified", "native"), default="unified")
    parser.add_argument("--stages", nargs="+", default=["dev_5", "dev_20", "dev_100"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--null-dev-pilot", type=int, help="Limit a separate null_dev pilot to at most 20 frozen development queries")
    args = parser.parse_args()
    pre_register()
    if args.prepare:
        return
    if args.model is None or f"{args.model}:{args.context}" not in SETTINGS["development_modes"]:
        parser.error("Choose a preregistered model and context mode")
    splits = load_splits()
    if any(stage not in splits or not isinstance(splits[stage], list) for stage in args.stages):
        parser.error("Unknown cohort")
    manifest = json.loads((OUTPUT / "candidate_pool_manifest.json").read_bytes())
    pair_counts = {name: item["candidate_chunk_pairs"] for name, item in manifest["cohorts"].items()}
    pilot_pairs = None
    if args.null_dev_pilot is not None:
        if args.stages != ["null_dev"] or not 1 <= args.null_dev_pilot <= 20:
            parser.error("--null-dev-pilot needs only the null_dev stage and a size from 1 to 20")
        query_ids = sorted(Random(42).sample(splits["null_dev"], args.null_dev_pilot))
        pilot_pairs = sum(len(make_pairs(pool)) for pool in iter_pools(query_ids))
        pilot_path = OUTPUT / "null_dev_pilot.json"
        plan = {"seed": 42, "query_ids": query_ids, "pair_count": pilot_pairs}
        if pilot_path.exists() and json.loads(pilot_path.read_bytes()) != plan:
            raise RuntimeError("The no-answer pilot cohort has already been frozen")
        if not pilot_path.exists():
            write_json(pilot_path, plan)
        splits["null_dev"] = query_ids
    from .backend import FastEmbedBackend

    backend = FastEmbedBackend(
        MODELS[args.model], MODEL_CACHE, threads=SETTINGS["threads"],
        context_mode=args.context, batch_size=SETTINGS["batch_size"],
        unified_limit=SETTINGS["unified_limit"],
    )
    run_id = _run_id(backend, args.model, args.context)
    write_json(OUTPUT / f"model_{run_id}.json", backend.metadata)
    with RerankCache(OUTPUT / "rerank_cache" / "scores.sqlite3", backend.metadata) as cache:
        for stage in args.stages:
            _allow_stage(run_id, stage)
            _stage(backend, cache, run_id, args.model, stage, splits[stage], pair_counts, pilot_pairs)


if __name__ == "__main__":
    main()
