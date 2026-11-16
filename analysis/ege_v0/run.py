"""Generate and checkpoint Gold-free expansion traces for the frozen cohorts."""

from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import file_digest, sha256
from itertools import product
from pathlib import Path
from time import monotonic, perf_counter, process_time, time
import json
import os

from hoprag.models import Document
from hoprag.retrievers import Chunk
from analysis.besa_v0.experiment import OUT as BESA, R5, R6, MODEL_SPEC, cache_source, read, rows, write, write_rows
from analysis.reranking.pool import iter_pools, load_splits, read_manifest
from analysis.reranking_optimization.policies import select, weighted_fuse

from .expansion import ExpansionConfig, LexicalIndex, expand, through_hop
from .ranking import Scorer, local_jina_backend


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / "run"
SPLIT = BESA / "run" / "development_split.json"
DEV_RANKS = R6 / "development_rankings_answerable_dev.jsonl"
NULL_RANKS = R6 / "no_answer_results.jsonl"
CANDIDATE_DOCS = R5 / "candidate_documents.jsonl"
JINA_HISTORY = R6 / "rerank_cache" / "scores.sqlite3"
CLOCK = DATA / "budget_clock.json"
PILOT_LIMIT = 10
BUDGET_SECONDS = 7200
RESERVE_SECONDS = 90
LEXICAL = {f"s{seeds}-{style}": ExpansionConfig(seed_count=seeds, query_style=style)
           for seeds, style in product((3, 5), ("q1", "q2", "q3"))}
QUESTION_ONLY = ExpansionConfig(max_hops=1)
JINA_PARENTS = 8
JINA_CHUNKS = 1
HISTORICAL_INPUTS = (
    CANDIDATE_DOCS, R5 / "candidate_pools.jsonl", R5 / "candidate_pool_manifest.json",
    R5 / "splits.json", MODEL_SPEC, JINA_HISTORY, R5 / "rerank_cache" / "scores.sqlite3",
    R6 / "frozen_reranking_config.json", R6 / "heldout_run_lock.json", DEV_RANKS, NULL_RANKS, SPLIT,
    BESA / "run" / "selected_results.jsonl", BESA / "run" / "development_scores.jsonl",
    ROOT / "results" / "baseline" / "run.json", ROOT / "results" / "baseline" / "chunks.jsonl",
    ROOT / "results" / "baseline" / "rankings_full.jsonl",
    ROOT / "data" / "raw" / "multihop_rag" / "corpus.json",
    ROOT / "data" / "raw" / "multihop_rag" / "MultiHopRAG.json",
)


def fingerprint(path):
    with path.open("rb") as stream:
        return {"sha256": file_digest(stream, "sha256").hexdigest(), "bytes": path.stat().st_size}


def cohorts():
    splits = load_splits()
    dev = read(SPLIT)
    tune, valid = set(dev["tune"]), set(dev["validation"])
    null = set(splits["null_dev"]) | set(splits["null_heldout"])
    if (len(tune), len(valid), len(null)) != (271, 180, 301) or tune & valid or (tune | valid) & null:
        raise ValueError("Unexpected experimental cohorts")
    if tune | valid != set(splits["answerable_dev"]) or (tune | valid | null) & set(splits["answerable_heldout"]):
        raise ValueError("Requested cohort deviates from the historical answerable development split")
    return tune, valid, null


def corpus():
    docs, chunks = {}, {}
    for row in rows(CANDIDATE_DOCS):
        doc = row["document_id"]
        docs[doc] = Document(doc, row["title"], row["text"],
                             {"source": row["article_source"], "url": row["article_url"]})
        chunks[doc] = tuple(Chunk(**chunk) for chunk in row["chunks"])
    if len(docs) != 609 or sum(map(len, chunks.values())) != 2378:
        raise ValueError("Frozen candidate corpus does not match 609 parents and 2,378 chunks")
    return docs, chunks


def fixed_pool(candidates, limit):
    return tuple(c.document_id for c in candidates
                 if any(rank is not None and rank <= limit for rank in (c.bm25_rank, c.dense_rank)))


def inputs(requested):
    tune, valid, null = cohorts()
    requested = set(requested)
    if not requested <= tune | valid | null:
        raise ValueError("EGE takes only the frozen development and null cohorts")
    wanted_null = requested & null
    archives = {r["query_id"]: r for r in rows(DEV_RANKS)
                if r["pool_id"] == "P5" and r["query_id"] in requested}
    archives.update({r["query_id"]: r for r in rows(NULL_RANKS) if r["query_id"] in wanted_null
                     and r["pool_id"] == "P5" and r["scored"]})
    if archives.keys() != requested:
        raise ValueError("Missing a frozen P5-B question")
    for orig in iter_pools(requested):
        saved = archives[orig.query_id]
        p5 = select(orig, "P5")
        jina = saved["jina_ranking"]
        full = [r["document_id"] for r in weighted_fuse(p5, jina, 2, 1, 60)]
        archived_order = (saved["optimized_ranking"] if orig.query_id in wanted_null
                          else saved["final_top10_by_rrf_weight"]["2"])
        expected = [r["document_id"] for r in archived_order]
        if full[:len(expected)] != expected or set(fixed_pool(orig.candidates, 15)) != set(full):
            raise ValueError(f"P5-B differs from its archive for {orig.query_id}")
        start = perf_counter()
        fixed = {str(limit): fixed_pool(orig.candidates, limit) for limit in (30, 50, 100)}
        elapsed = perf_counter() - start
        yield orig.query_id, orig.question, full, jina, fixed, elapsed


def stage_scope(label, settings, source_state):
    local = {name: fingerprint(OUT / name)["sha256"] for name in ("run.py", "expansion.py", "ranking.py")}
    encoded = json.dumps([label, settings, source_state, local], sort_keys=True, allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def save_file(path, records):
    write_rows(path, sorted(records, key=lambda row: row["query_id"]))


class CheckpointLog:
    """Append one atomic logical record after a completed question. Repair a partial final line."""

    def __init__(self, path, scope):
        self.path = path
        self.scope = scope
        self.completed = {}
        if not path.exists():
            return
        end = 0
        with path.open("rb+") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    stream.seek(end)
                    stream.truncate()
                    break
                payload = json.loads(line)
                if payload["scope"] != scope:
                    raise ValueError(f"Checkpoint input signature changed: {path.name}")
                record = payload["record"]
                if record["query_id"] in self.completed:
                    raise ValueError(f"Duplicate checkpoint question: {record['query_id']}")
                self.completed[record["query_id"]] = record
                end += len(line)

    def put(self, record):
        query_id = record["query_id"]
        if query_id in self.completed:
            raise ValueError(f"Question already checkpointed: {query_id}")
        payload = json.dumps({"scope": self.scope, "record": record}, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
        with self.path.open("ab") as stream:
            stream.write(payload.encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.completed[query_id] = record


def initial_state():
    DATA.mkdir(parents=True, exist_ok=True)
    before = DATA / "historical_inputs_before.json"
    if not before.exists():
        manifest = read_manifest(R5)
        for path, expected in manifest["source_artifacts"].items():
            if fingerprint(ROOT / path) != expected:
                raise ValueError(f"Changed historical source: {path}")
        if fingerprint(R6 / "frozen_reranking_config.json")["sha256"] != read(R6 / "heldout_run_lock.json")["frozen_config_sha256"]:
            raise ValueError("Historical P5-B lock does not match")
        expected_model = read(MODEL_SPEC)
        paths = (*HISTORICAL_INPUTS, cache_source(), Path(expected_model["model_path"]))
        integrity = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in paths}
        if integrity[Path(expected_model["model_path"]).relative_to(ROOT).as_posix()]["sha256"] != expected_model["model_sha256"]:
            raise ValueError("Existing Jina ONNX artifact does not match the locked digest")
        write(before, integrity)
    if not CLOCK.exists():
        started = time()
        write(CLOCK, {"started_epoch": started, "deadline_epoch": started + BUDGET_SECONDS})
    return read(before), monotonic() + read(CLOCK)["deadline_epoch"] - time()


def check_stage(stage, deadline, estimate=None):
    remaining = deadline - monotonic()
    if remaining < (estimate or 0) + RESERVE_SECONDS:
        raise TimeoutError(f"{stage}: {max(0, remaining):.1f} seconds left; committed progress retained")


def run_lexical(stage, deadline, integrity):
    tune, valid, null = cohorts()
    pilot = sorted(tune, key=lambda qid: sha256(f"ege-pilot-42:{qid}".encode()).digest())[:PILOT_LIMIT]
    frozen = read(OUT / "experiment_config.json") if stage == "null" else None
    variants = LEXICAL if frozen is None else {
        key: LEXICAL[key] for key in sorted(set(frozen["seed_variants"].values()) | {frozen["lexical_winner"]["variant"]})
    }
    wanted = {"pilot": set(pilot), "development": tune | valid, "null": null}[stage]
    variant_settings = {name: asdict(value) for name, value in variants.items()}
    scope = stage_scope(stage, variant_settings, integrity)
    log = CheckpointLog(DATA / f"{stage}_checkpoint.jsonl", scope)
    if not log.completed.keys() <= wanted:
        raise ValueError("Checkpoint is outside its frozen stage")
    pending = wanted - log.completed.keys()
    start, cpu = perf_counter(), process_time()
    construction_time = 0.0
    if pending:
        check_stage(stage, deadline)
        docs, chunks = corpus()
        index = LexicalIndex(docs.values(), chunks)
        construction_time = perf_counter() - start
        for offset, (qid, question, baseline, jina, fixed, fixed_seconds) in enumerate(inputs(pending), 1):
            check_stage(stage, deadline)
            result = {key: expand(index, qid, question, baseline, jina, config=config)
                      for key, config in variants.items()}
            ctrl = expand(index, qid, question, baseline, jina, config=QUESTION_ONLY, question_only=True)
            log.put({"query_id": qid, "question": question, "baseline": baseline, "jina": jina,
                     "fixed": fixed, "fixed_seconds": fixed_seconds, "question_only": ctrl, "variants": result})
            if offset % 50 == 0 or offset == len(pending):
                print(f"{stage}: {len(log.completed)}/{len(wanted)}", flush=True)
    records = log.completed.values()
    result_path = DATA / f"{stage}_expansion.jsonl"
    save_file(result_path, records)
    if log.completed.keys() != wanted:
        raise ValueError("Incomplete expansion snapshot")
    performance = {
        "queries": len(wanted), "resumed": len(wanted) - len(pending), "variants": variant_settings,
        "question_only_config": asdict(QUESTION_ONLY), "index_build_seconds": construction_time,
        "wall_seconds": perf_counter() - start, "cpu_seconds": process_time() - cpu,
        "embedding_requests": 0, "dense_seconds": 0.0,
        "bm25_seconds": sum(run["bm25_seconds"] for r in records
                            for run in (r["question_only"], *r["variants"].values())),
    }
    dest = DATA / f"{stage}_performance.json"
    if pending or not dest.exists():
        write(dest, performance)
    return result_path


def winner_run(row, selection):
    return through_hop(row["variants"][selection["variant"]], selection["hops"])


def ordered_novel(run):
    ordered = sorted(run["new_candidates"], key=lambda row: (-row["score"], row["hop"],
                                                           row["retrieval_rank"], row["document_id"]))
    return tuple(row["document_id"] for row in ordered)


def score_jina(stage, source, deadline, integrity, *, pilot=False):
    started, cpu = perf_counter(), process_time()
    records = list(rows(source))
    frozen = None if pilot else read(OUT / "experiment_config.json")["lexical_winner"]
    selection = frozen or {"variant": "s3-q1", "hops": 2}
    label = f"{stage}_jina"
    scope = stage_scope(label, [selection, JINA_PARENTS, JINA_CHUNKS], integrity)
    log = CheckpointLog(DATA / f"{label}_checkpoint.jsonl", scope)
    if not log.completed.keys() <= {r["query_id"] for r in records}:
        raise ValueError("Mismatch in Jina checkpoint cohort")
    pending = [row for row in records if row["query_id"] not in log.completed]
    backend_issue = None
    if pending:
        check_stage(label, deadline)
        docs, chunks = corpus()
        metadata = read(MODEL_SPEC)
        # All artifacts are present locally. Set offline mode before importing the runtime.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        try:
            backend = local_jina_backend(R5 / "model_cache", metadata, threads=4, batch_size=16)
            scoring = Scorer(DATA / "cache" / "jina.sqlite3",
                             (JINA_HISTORY, R5 / "rerank_cache" / "scores.sqlite3"), metadata, backend,
                             max_parents=JINA_PARENTS, chunks_per_parent=JINA_CHUNKS, batch_size=16)
        except (RuntimeError, OSError, ImportError, ValueError) as error:
            scoring = nullcontext(None)
            backend_issue = f"{type(error).__name__}: {error}"
        with scoring as scorer:
            for offset, row in enumerate(pending, 1):
                check_stage(label, deadline)
                run = winner_run(row, selection)
                witnesses = {c["document_id"]: c["chunk_id"] for c in run["new_candidates"]}
                batch = (scorer.score(row["query_id"], row["question"], ordered_novel(run), docs,
                                      expansion_witnesses=witnesses, chunks=chunks,
                                      deadline=deadline - RESERVE_SECONDS) if scorer else None)
                if batch and batch.fallback_reason == "deadline_exhausted":
                    check_stage(label, deadline)
                log.put({"query_id": row["query_id"], "batch": asdict(batch) if batch else None,
                         "backend_issue": backend_issue})
                if offset % 100 == 0 or offset == len(pending):
                    print(f"{label}: {len(log.completed)}/{len(records)}", flush=True)
    result_path = DATA / f"{label}.jsonl"
    save_file(result_path, log.completed.values())
    batches = [r["batch"] for r in log.completed.values() if r["batch"] is not None]
    perf = DATA / f"{label}_performance.json"
    stats = {
        "queries": len(records), "batches": len(batches), "resumed": len(records) - len(pending),
        "wall_seconds": perf_counter() - started, "cpu_seconds": process_time() - cpu,
        "backend_issue": backend_issue,
        **{key: sum(b[key] for b in batches) for key in ("actual_pairs", "historical_hits", "local_hits",
                                                       "new_pairs", "inference_seconds")},
    }
    if pending or not perf.exists():
        write(perf, stats)
    return read(perf)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "development", "development-jina", "null", "null-jina"))
    args = parser.parse_args()
    if not (OUT / "FAILURE_RECLASSIFICATION.md").is_file():
        parser.error("Complete the offline failure audit before starting EGE experiments")
    integrity, deadline = initial_state()
    if args.stage == "pilot":
        source = run_lexical("pilot", deadline, integrity)
        stats = score_jina("pilot", source, deadline, integrity, pilot=True)
        write(OUT / "pilot_performance.json", {
            "expansion": read(DATA / "pilot_performance.json"), "jina": stats,
            "upper_new_pair_budget_at_8_per_dev_query": JINA_PARENTS * 451,
            "seconds_remaining_after_pilot": deadline - monotonic(),
        })
    else:
        pilot = read(OUT / "pilot_performance.json")
        if args.stage in ("development", "null"):
            if args.stage == "development":
                check_stage("development estimate", deadline,
                            3 * (451 / PILOT_LIMIT) * pilot["expansion"]["wall_seconds"])
            run_lexical(args.stage, deadline, integrity)
        else:
            stage = args.stage.split("-")[0]
            check_stage(args.stage, deadline,
                        3 * ((451 if stage == "development" else 301) / PILOT_LIMIT) *
                        max(1.0, pilot["jina"]["inference_seconds"]))
            score_jina(stage, DATA / f"{stage}_expansion.jsonl", deadline, integrity)


if __name__ == "__main__":
    main()

