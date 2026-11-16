from __future__ import annotations

import json
import os
import platform
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .config import BenchmarkConfig
from .dataset import load_dataset, stratified_sample
from .embeddings import OllamaEmbedder
from .evaluation import evaluate
from .graph_regressions import run_graph_regressions
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever


def _write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(data, target, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        target.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _ollama_status(config: BenchmarkConfig, endpoint: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{config.ollama_url.rstrip('/')}/api/{endpoint}", timeout=10) as response:
        return json.load(response)


def _model_info(config: BenchmarkConfig) -> dict:
    model = config.embedding_model
    names = {model} if ":" in model else {model, f"{model}:latest"}
    for item in _ollama_status(config, "tags").get("models", ()):
        if item.get("name") in names or item.get("model") in names:
            digest = item.get("digest")
            if not isinstance(digest, str) or not digest:
                raise RuntimeError(f"Ollama reported no digest for {model!r}")
            return {key: item.get(key) for key in ("name", "digest", "size", "details")}
    raise RuntimeError(f"Already-installed Ollama model {model!r} was not found")


def _stats_delta(current: dict, previous: dict) -> dict:
    return {key: value - previous.get(key, 0) for key, value in current.items()}


def _evaluate_stage(config, queries, documents, bm25, dense, hybrid, output, label):
    methods = {"bm25": {}, "dense": {}, "hybrid": {}}
    timings = {name: 0.0 for name in methods}
    rows = []
    for query in queries:
        started = perf_counter()
        lexical = bm25.search(query, config.fusion_depth)
        timings["bm25"] += perf_counter() - started

        started = perf_counter()
        vector = dense.search(query, config.fusion_depth)
        timings["dense"] += perf_counter() - started

        started = perf_counter()
        fused = hybrid.fuse(query, lexical, vector, config.top_k)
        timings["hybrid"] += perf_counter() - started

        methods["bm25"][query.query_id] = lexical[:config.top_k]
        methods["dense"][query.query_id] = vector[:config.top_k]
        methods["hybrid"][query.query_id] = fused
        rows.append({
            "query_id": query.query_id,
            "question_type": query.question_type,
            "evidence_count": len(query.evidence_list),
            "bm25": [asdict(result) for result in lexical],
            "dense": [asdict(result) for result in vector],
            "hybrid": [asdict(result) for result in fused],
        })

    metrics_started = perf_counter()
    metrics = {method: evaluate(queries, documents, rankings) for method, rankings in methods.items()}
    metric_seconds = perf_counter() - metrics_started
    _write_jsonl(output / f"rankings_{label}.jsonl", rows)
    strata = Counter((query.question_type, len(query.evidence_list)) for query in queries)
    return {
        "query_count": len(queries),
        "query_ids": [query.query_id for query in queries],
        "strata": [
            {"question_type": kind, "evidence_count": count, "queries": total}
            for (kind, count), total in sorted(strata.items())
        ],
        "ranking_seconds": timings,
        "metric_seconds": metric_seconds,
        "metrics": metrics,
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    started = perf_counter()
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "dependencies": {name: version(name) for name in ("numpy", "rank-bm25")},
        "llm_tokens_and_cost": "not applicable",
        "stages": {},
    }
    _write_json(output / "run.json", manifest)
    try:
        dataset = load_dataset(config.raw_directory)
        _write_json(output / "dataset_audit.json", dataset.audit)
        _write_json(output / "graph_regressions.json", run_graph_regressions(config.example_corpus))
        print(f"Dataset: {len(dataset.documents)} documents, {len(dataset.queries)} queries", flush=True)

        model = _model_info(config)
        manifest["ollama_model"] = model
        cache_path = config.cache_directory / f"embeddings-{model['digest']}.sqlite3"
        manifest["embedding_cache"] = str(cache_path)
        config.cache_directory.mkdir(parents=True, exist_ok=True)
        _write_json(output / "run.json", manifest)

        step_started = perf_counter()
        bm25 = BM25Retriever(dataset.documents)
        manifest["bm25_build_seconds"] = perf_counter() - step_started
        last_progress = perf_counter()

        def embedding_progress(done, total):
            nonlocal last_progress
            now = perf_counter()
            if done in (0, total) or now - last_progress >= 30:
                print(f"Embeddings available: {done}/{total} unique texts", flush=True)
                last_progress = now

        with OllamaEmbedder(
            cache_path,
            base_url=config.ollama_url,
            model=config.embedding_model,
            batch_size=config.batch_size,
            dimensions=config.embedding_dimensions,
            timeout=config.embedding_timeout_seconds,
            keep_alive=config.embedding_keep_alive,
            progress=embedding_progress,
        ) as embedder:
            print(f"Embedding/indexing all {len(dataset.documents)} documents", flush=True)
            step_started = perf_counter()
            dense = DenseRetriever(dataset.documents, embedder, max_chunk_bytes=config.max_chunk_bytes)
            manifest["dense_build_seconds"] = perf_counter() - step_started
            manifest["dense_chunks"] = len(dense.chunks)
            manifest["corpus_embedding"] = asdict(embedder.stats)
            manifest["dense_matrix_bytes_float32"] = len(dense.chunks) * config.embedding_dimensions * 4
            _write_jsonl(output / "chunks.jsonl", (asdict(chunk) for chunk in dense.chunks))
            _write_json(output / "run.json", manifest)
            hybrid = HybridRetriever(bm25, dense, rrf_k=config.rrf_k)

            for size in config.stage_sizes:
                stage_started = perf_counter()
                label = str(size) if size is not None else "full"
                queries = dataset.queries if size is None else stratified_sample(dataset.queries, size, seed=config.seed)
                print(f"Evaluating {label}: {len(queries)} queries", flush=True)
                stats_before = asdict(embedder.stats)
                dense.prepare_queries(queries)
                result = _evaluate_stage(config, queries, dataset.documents, bm25, dense, hybrid, output, label)
                result["embedding"] = _stats_delta(asdict(embedder.stats), stats_before)
                result["elapsed_seconds"] = perf_counter() - stage_started
                _write_json(output / f"metrics_{label}.json", result)
                manifest["stages"][label] = result
                _write_json(output / "run.json", manifest)
                print(f"Completed {label} in {result['elapsed_seconds']:.2f}s", flush=True)
            manifest["total_embedding"] = asdict(embedder.stats)

        try:
            manifest["ollama_running_models"] = _ollama_status(config, "ps").get("models", [])
        except OSError as exc:
            manifest["ollama_resource_probe_error"] = str(exc)
        manifest["state"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["state"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["elapsed_seconds"] = perf_counter() - started
        _write_json(output / "run.json", manifest)
    return manifest
