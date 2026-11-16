"""Run BESA-v0 on the established development split without Gold inputs."""

import argparse
from collections import defaultdict
from contextlib import closing, nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
import sqlite3
from time import perf_counter, process_time

from analysis.reranking.backend import FastEmbedBackend, peak_rss_bytes
from analysis.reranking_optimization.runner import _lower_priority, identity

from .experiment import (
    BUDGETS, MAX_SECONDS, MODEL_SPEC, OUT, R5, Checkpoints, assemble_scoring,
    cache_source, development_inputs, frozen_split, historical_scores, read, write, write_rows,
)
from .planner import plan
from .scoring import DenseVectors, NeedScoreCache, schedule, score_schedule

OUTPUT = OUT / "run"
CLOCK = OUTPUT / "budget_clock.json"
CHUNKS = OUT.parents[1] / "results" / "baseline" / "chunks.jsonl"
THREADS = 4
BATCH = 16


class TimedBackend:
    def __init__(self, model, deadline):
        self.model = model
        self.deadline = deadline

    def score(self, query, texts):
        if datetime.now(timezone.utc) >= self.deadline:
            raise TimeoutError("Development scoring deadline reached")
        return self.model.score(query, texts)


def _chunks():
    from hoprag.retrievers import Chunk

    chunks = defaultdict(list)
    with CHUNKS.open(encoding="utf-8") as stream:
        for line in stream:
            chunk = Chunk(**json.loads(line))
            chunks[chunk.document_id].append(chunk)
    return dict(chunks)


def _scope():
    hashes = [
        sha256((OUT / name).read_bytes()).hexdigest()
        for name in ("run.py", "experiment.py", "planner.py", "scoring.py")
    ]
    model_hash = read(MODEL_SPEC)["model_sha256"]
    return sha256((model_hash + ":".join(hashes)).encode()).hexdigest()


def _checked_plan(question):
    try:
        return plan(question)
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        from .planner import Plan

        return Plan(question=question, needs=(), accepted=False, fallback_reason="planner_error")


def _scored_record(pool, baseline, vectors, backend, cache):
    started = perf_counter()
    p = _checked_plan(pool.question)
    planner_seconds = perf_counter() - started

    def compute():
        if cache is None:
            raise OSError("Score cache unavailable")
        if backend is None:
            raise RuntimeError("Local model unavailable")
        planned = schedule(pool, p.needs, {d: i for i, d in enumerate(baseline, 1)}, vectors, limit=BUDGETS[-1])
        if len(planned) > BUDGETS[-1]:
            raise ValueError("Schedule exceeds Jina pair budget")
        if {pair.need_id for pair in planned} != {need.need_id for need in p.needs}:
            raise ValueError("One or more needs were not assigned any chunk")
        return score_schedule(backend, cache, pool.query_id, planned, batch_size=BATCH)

    before_scoring = perf_counter()
    scored, fallback = assemble_scoring(pool, baseline, p, compute)
    if fallback == "scoring_error:TimeoutError":
        raise TimeoutError("Development deadline reached; committed scores retained")
    return {
        "query_id": pool.query_id, "plan": asdict(p), "fallback_reason": fallback,
        "baseline": baseline, "scored_pairs": [asdict(row) for row in scored],
        "planner_seconds": planner_seconds, "scorer_wall_seconds": perf_counter() - before_scoring,
        "wall_seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "development", "warm-pilot"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    split = frozen_split()
    write(OUTPUT / "development_split.json", split)
    pilot_ids = sorted(split["tune"], key=lambda q: sha256(f"pilot:42:{q}".encode()).digest())[:20]
    wanted = set(pilot_ids if args.stage != "development" else split["tune"] + split["validation"])
    started = perf_counter()
    cpu_started = process_time()
    scope = _scope()
    if not CLOCK.exists():
        now = datetime.now(timezone.utc)
        write(CLOCK, {"started_utc": now.isoformat(), "deadline_utc": (now + timedelta(seconds=MAX_SECONDS)).isoformat()})
    deadline = datetime.fromisoformat(read(CLOCK)["deadline_utc"])
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(THREADS))
    _lower_priority()
    vectors = DenseVectors.load(cache_source(), _chunks())
    spec = read(MODEL_SPEC)
    model, backend_error = None, None
    try:
        model = FastEmbedBackend(spec["model_id"], R5 / "model_cache", threads=THREADS,
                                 context_mode="unified", batch_size=BATCH, unified_limit=256)
        if identity(model.metadata) != identity(spec):
            raise ValueError("Model identity mismatch")
    except (RuntimeError, ValueError, OSError, ImportError, KeyError) as exc:
        model = None
        backend_error = str(exc)
    backend = TimedBackend(model, deadline) if model is not None else None
    cache_error = None
    try:
        cache_object = NeedScoreCache(OUTPUT / "cache" / "scores.sqlite3",
                                      dict(zip(("audit", "r6", "r5"), historical_scores(), strict=True)), spec)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        cache_error = str(exc)
        cache_object = nullcontext(None)

    output = []
    resumed = 0
    with closing(Checkpoints(OUTPUT / "checkpoints.sqlite3")) as checkpoints, cache_object as cache:
        try:
            for i, (pool, baseline, _) in enumerate(development_inputs(wanted), 1):
                warm = args.stage == "warm-pilot"
                row = None if warm else checkpoints.get(scope, pool.query_id)
                if row is None:
                    row = _scored_record(pool, baseline, vectors, backend, cache)
                    if not warm:
                        checkpoints.put(scope, row)
                else:
                    resumed += 1
                output.append(row)
                if i % 25 == 0 or i == len(wanted):
                    print(f"{args.stage}: {i}/{len(wanted)} questions", flush=True)
        except TimeoutError:
            write_rows(OUTPUT / f"{args.stage}_partial.jsonl", checkpoints.export(scope))
            raise

    if {r["query_id"] for r in output} != wanted or len(output) != len(wanted):
        raise ValueError("Cohort mismatch")
    write_rows(OUTPUT / f"{args.stage}_scores.jsonl", output)
    metrics = {
        "stage": args.stage, "scope": scope, "queries": len(output), "resumed_questions": resumed,
        "accepted_questions": sum(r["plan"]["accepted"] for r in output),
        "runtime_fallbacks": sum(r["fallback_reason"] is not None for r in output),
        "score_pairs": sum(len(r["scored_pairs"]) for r in output),
        "new_pairs": sum(pair["cache_source"] == "new_inference" for r in output for pair in r["scored_pairs"]),
        "scoring_wall_seconds": sum(r["scorer_wall_seconds"] for r in output),
        "wall_seconds": perf_counter() - started, "cpu_seconds": process_time() - cpu_started,
        "peak_working_set_bytes": peak_rss_bytes(), "backend_error": backend_error,
        "cache_error": cache_error, "threads": THREADS, "batch_size": BATCH,
        "model_revision": spec["revision"], "model_sha256": spec["model_sha256"],
        "model_context_tokens": spec["effective_max_tokens"], "embedding_requests": 0,
    }
    write(OUTPUT / f"{args.stage}_performance.json", metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
