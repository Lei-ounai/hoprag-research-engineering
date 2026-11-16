"""Score the frozen Top 30 and archived one-hop candidate union without labels."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import monotonic, perf_counter, process_time, time
import json
import os

from analysis.besa_v0.experiment import MODEL_SPEC, R5, cache_source, read, rows, write, write_rows
from analysis.besa_v0.scoring import DenseVectors
from analysis.ege_v0.ranking import local_jina_backend
from analysis.ege_v0.run import (DATA as EGE_DATA, HISTORICAL_INPUTS, OUT as EGE,
                                  CheckpointLog, cohorts, corpus, fingerprint, inputs)
from analysis.reranking.pool import iter_pools
from hoprag.dataset import _record_key
from hoprag.models import Document

from .scoring import FairScorer


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = OUT / "run"
CLOCK = RUN / "budget_clock.json"
BUDGET_SECONDS = 7200
RESERVE_SECONDS = 120
CHUNKS = 3
BATCH = 32
MODEL_BATCH = 16
THREADS = 4
CANDIDATE_LIMIT = 76
HISTORIES = {
    "round8": EGE_DATA / "cache" / "jina.sqlite3",
    "round6": R5.with_name("reranking_optimization") / "rerank_cache" / "scores.sqlite3",
    "round5": R5 / "rerank_cache" / "scores.sqlite3",
}
RAW_CORPUS = ROOT / "data" / "raw" / "multihop_rag" / "corpus.json"
PILOT_COUNT = 10


def documents_and_chunks():
    frozen, chunks = corpus()
    metadata, seen = {}, set()
    with RAW_CORPUS.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    for index, item in enumerate(raw, 1):
        key = _record_key(item)
        if key in seen:
            continue
        seen.add(key)
        parent = f"doc-{index:04d}"
        if item["title"] != frozen[parent].title or item["body"] != frozen[parent].text:
            raise ValueError(f"Original corpus mapping changed: {parent}")
        metadata[parent] = {key: item[key] for key in ("source", "published_at", "url")}
    if metadata.keys() != frozen.keys():
        raise ValueError("Incomplete original corpus metadata")
    docs = {doc: Document(doc, data.title, data.text, metadata[doc]) for doc, data in frozen.items()}
    return docs, chunks


def from_round8(wanted):
    tune, validation, null = cohorts()
    if not wanted <= tune | validation | null:
        raise ValueError("The runtime excludes all answerable holdout IDs")
    name = read(EGE / "experiment_config.json")["lexical_winner"]["variant"]
    output = {}
    groups = (("development", wanted - null), ("null", wanted & null))
    for label, selected in groups:
        if not selected:
            continue
        for row in rows(EGE_DATA / f"{label}_expansion.jsonl"):
            qid = row["query_id"]
            if qid not in selected:
                continue
            phase = next((h for h in row["variants"][name]["phases"] if h["hop"] == 1), None)
            expansion = {}
            if phase:
                requests = {call["request_id"]: call["query"] for call in phase["queries"]}
                for candidate in phase["new_candidates"]:
                    doc = candidate["document_id"]
                    expansion[doc] = {
                        "term": candidate["term"], "query": requests[candidate["request_id"]],
                        "chunk_id": candidate["chunk_id"], "seed_document_id": candidate["seed_document_id"],
                        "path": candidate["path"], "request_id": candidate["request_id"],
                        "rank": candidate["retrieval_rank"], "score": candidate["score"],
                    }
            output[qid] = expansion
    if output.keys() != wanted:
        raise ValueError("Archived first-hop candidate cohort is incomplete")
    return output


def competition_inputs(wanted, docs):
    expansion = from_round8(wanted)
    ranked = {item.query_id: item for item in iter_pools(wanted)}
    for qid, question, baseline, jina, fixed, _ in inputs(wanted):
        assert question == ranked[qid].question
        original = {c.document_id: c for c in ranked[qid].candidates}
        baseline_rank = {doc: rank for rank, doc in enumerate(baseline, 1)}
        history = {r["document_id"]: r for r in jina}
        fixed_ids = list(dict.fromkeys([*fixed["30"], *baseline]))
        union = list(dict.fromkeys([*fixed_ids, *expansion[qid]]))
        if len(union) > CANDIDATE_LIMIT:
            raise ValueError("Candidate union exceeds the fixed scoring budget")
        fixed_set = set(fixed_ids)
        candidates = {}
        for doc in union:
            ranks = original.get(doc)
            old = history.get(doc, {})
            exp = expansion[qid].get(doc)
            candidates[doc] = {
                "document_id": doc, "title": docs[doc].title, "metadata": docs[doc].metadata,
                "p5b_rank": baseline_rank.get(doc), "bm25_rank": ranks.bm25_rank if ranks else None,
                "dense_rank": ranks.dense_rank if ranks else None, "rrf_rank": ranks.rrf_rank if ranks else None,
                "archived_jina_score": old.get("score"), "archived_jina_chunk_id": old.get("winning_chunk_id"),
                "in_fixed": doc in fixed_set, "expansion": exp,
                "lanes": [lane for lane, rank in (("bm25", ranks.bm25_rank if ranks else None),
                                                 ("dense", ranks.dense_rank if ranks else None))
                          if rank is not None and rank <= 30] + (["round8_one_hop"] if exp else []),
            }
        # Keep common candidates identical in E4 and E5 for a clean expansion ablation.
        terms = {doc: [value["term"]] for doc, value in expansion[qid].items() if doc not in fixed_set}
        yield {"query_id": qid, "question": question, "baseline": baseline, "fixed": fixed_ids,
               "candidate_ids": union, "candidates": candidates}, terms


def input_paths():
    model = read(MODEL_SPEC)
    return tuple(dict.fromkeys((
        *HISTORICAL_INPUTS, *HISTORIES.values(), cache_source(), Path(model["model_path"]),
        EGE / "EGE_V0_REPORT.md", EGE / "experiment_config.json", EGE / "experiment_manifest.json",
        EGE / "failure_reclassification.csv", EGE / "expansion_metrics.csv",
        EGE / "candidate_expansion_results.jsonl", EGE / "expansion_paths.jsonl",
        EGE / "final_ranking_comparison.csv", EGE / "no_answer_expansion_analysis.md",
        EGE_DATA / "development_expansion.jsonl", EGE_DATA / "null_expansion.jsonl",
    )))


def snapshot():
    return {path.relative_to(ROOT).as_posix(): fingerprint(path) for path in input_paths()}


def deadline_start():
    RUN.mkdir(exist_ok=True, parents=True)
    before = RUN / "historical_inputs_before.json"
    if not before.exists():
        write(before, snapshot())
    if not CLOCK.exists():
        now = time()
        write(CLOCK, {"started_epoch": now, "deadline_epoch": now + BUDGET_SECONDS})
    return read(before), monotonic() + read(CLOCK)["deadline_epoch"] - time()


def check_time(deadline, requirement=0.0):
    seconds = deadline - monotonic()
    if seconds <= RESERVE_SECONDS + requirement:
        raise TimeoutError(f"Remaining experimental time {max(0, seconds):.1f}s. Committed scores retained.")


def estimate_remaining(pilot):
    # Cached hits can vary between cohorts, so estimate with all scheduled pairs as cold.
    speed = pilot["new_pairs"] / max(0.01, pilot["inference_seconds"])
    if pilot["score_fallbacks"]:
        raise ValueError("Pilot scoring fallback requires investigation")
    speed = speed if pilot["new_pairs"] >= 8 and speed >= 3 else 25.0
    pairs = pilot["actual_pairs"] / pilot["queries"]
    return (pairs / speed + 0.06) * 1.4


def run(stage):
    tune, validation, null = cohorts()
    pilot = set(sorted(tune, key=lambda item: sha256(f"egr-pilot-42:{item}".encode()).digest())[:PILOT_COUNT])
    wanted = {"pilot": pilot, "development": tune | validation, "null": null}[stage]
    before, deadline = deadline_start()
    scope = sha256(json.dumps({
        "stage": stage, "sources": before, "chunks": CHUNKS, "batch": BATCH,
        "code": {p: fingerprint(OUT / p)["sha256"] for p in ("run.py", "scoring.py")},
    }, sort_keys=True).encode()).hexdigest()
    log = CheckpointLog(RUN / f"{stage}_checkpoint.jsonl", scope)
    if not log.completed.keys() <= wanted:
        raise ValueError("Checkpoint differs from frozen cohort")
    pending = wanted - log.completed.keys()
    start, cpu_start = perf_counter(), process_time()
    if pending:
        estimate = estimate_remaining(read(RUN / "pilot_performance.json")) * len(pending) if stage != "pilot" else 0.0
        check_time(deadline, estimate)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(THREADS))
        if os.name == "nt":
            from ctypes import windll
            kernel = windll.kernel32
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)
        docs, chunks = documents_and_chunks()
        vectors = DenseVectors.load(cache_source(), chunks)
        metadata = read(MODEL_SPEC)
        backend = local_jina_backend(R5 / "model_cache", metadata, threads=THREADS, batch_size=MODEL_BATCH)
        with FairScorer(RUN / "scores.sqlite3", HISTORIES, metadata, backend, vectors=vectors,
                        candidates=docs, chunks=chunks, chunk_budget=CHUNKS, batch_size=BATCH, max_parents=CANDIDATE_LIMIT) as scorer:
            for index, (record, terms) in enumerate(competition_inputs(pending, docs), 1):
                check_time(deadline)
                score = scorer.score(record["query_id"], record["question"], record["candidate_ids"],
                                     expansion_terms=terms, deadline=deadline - RESERVE_SECONDS)
                if score.get("fallback_reason") == "deadline_exhausted":
                    raise TimeoutError("Committed pairs and questions remain available")
                log.put({**record, "scoring": score})
                if index % 25 == 0 or index == len(pending):
                    print(f"{stage}: {len(log.completed)}/{len(wanted)}", flush=True)
    if log.completed.keys() != wanted:
        raise ValueError("Not all questions completed")
    save = RUN / f"{stage}_scores.jsonl"
    if pending or not save.exists():
        write_rows(save, (log.completed[qid] for qid in sorted(wanted)))
    report = RUN / f"{stage}_performance.json"
    if pending or not report.exists():
        profiles = [r["scoring"] for r in log.completed.values()]
        write(report, {"queries": len(wanted), "resumed": len(wanted) - len(pending),
                       "wall_seconds": perf_counter() - start, "cpu_seconds": process_time() - cpu_start,
                       "embedding_requests": 0, "threads": THREADS, "batch_size": BATCH, "model_batch_size": MODEL_BATCH,
                       "score_fallbacks": sum(bool(r["fallback_reason"]) for r in profiles),
                       **{key: sum(r.get(key, 0) for r in profiles) for key in
                          ("actual_pairs", "new_pairs", "historical_hits", "local_hits",
                           "inference_seconds", "truncated_pairs")}})
    print(read(report), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "development", "null"))
    args = parser.parse_args()
    if not (OUT / "round8_audit.json").is_file():
        parser.error("Run the preceding round-eight audit")
    run(args.stage)


if __name__ == "__main__":
    main()

