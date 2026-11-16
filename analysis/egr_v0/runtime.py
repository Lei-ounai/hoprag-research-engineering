"""Gold-free witness verification and protected Top 10 controls."""

from argparse import ArgumentParser
from dataclasses import asdict
from itertools import chain
from time import perf_counter

from analysis.besa_v0.experiment import rows, write, write_rows
from analysis.ege_v0.ranking import fuse
from analysis.reranking.core import _rank

from .evidence import profile_question, verify
from .run import CLOCK, RUN, CHUNKS, check_time, documents_and_chunks
from .selector import GateConfig, prepare


def make_witnesses(record, chunk_map, source_names):
    profile = profile_question(record["question"], source_names)
    output = {}
    for doc, chosen in record["scoring"]["documents"].items():
        if not 0 < len(chosen) <= CHUNKS or len({x["chunk_id"] for x in chosen}) != len(chosen):
            raise ValueError("Fair scoring budget or witness identity invalid")
        candidate = record["candidates"][doc]
        origin = candidate["expansion"]
        result = []
        for score in chosen:
            chunk = chunk_map[score["chunk_id"]]
            if chunk.document_id != doc:
                raise ValueError(f"Invalid witness parent: {chunk.chunk_id}")
            witness = verify(profile, record["query_id"], doc, chunk.chunk_id, chunk.text,
                             title=candidate["title"], metadata=candidate["metadata"],
                             expansion_queries=(origin["query"], origin["term"]) if origin else (),
                             jina_score=score["jina_score"],
                             model_visible_chunk_chars=score["model_visible_chunk_chars"])
            row = asdict(witness)
            if row["evidence_excerpt"]:
                start, end = row["evidence_start"], row["evidence_end"]
                if chunk.text[start:end] != row["evidence_excerpt"]:
                    raise ValueError(f"Excerpt does not match original Chunk: {chunk.chunk_id}")
            result.append(row)
        output[doc] = result
    if len(output) != len(record["candidate_ids"]) and not record["scoring"]["fallback_reason"]:
        raise ValueError("Missing fair-scored candidate without a fallback reason")
    return {"query_id": record["query_id"], "profile": asdict(profile), "witnesses": output}


def control(record, method):
    baseline = record["baseline"]
    if method == "E0":
        return baseline[:10]
    fixed = record["fixed"]
    if method == "E1":
        old = set(baseline)
        return fuse(baseline, [doc for doc in fixed if doc not in old],
                    mode="rrf", expansion_weight=0.88, anchor=7).document_ids
    if method != "E2":
        raise ValueError("Unknown protected control")
    if record["scoring"]["fallback_reason"]:
        return baseline[:10]
    scores = fair_scores(record)
    rrf_order = sorted(fixed, key=lambda doc: (record["candidates"][doc]["rrf_rank"], doc))
    ce_ranks = {r["document_id"]: r["rank"] for r in _rank({doc: scores[doc] for doc in fixed})}
    ranked = _rank({doc: 2 / (60 + rank) + 1 / (60 + ce_ranks[doc])
                    for rank, doc in enumerate(rrf_order, 1)})
    protected = baseline[:7]
    tail = [r["document_id"] for r in ranked if r["document_id"] not in protected][:3]
    # Keep incumbent positions when retained, permitting only 8–10 replacements.
    challengers = iter(doc for doc in tail if doc not in baseline[:10])
    reordered = [doc if doc in tail else next(challengers) for doc in baseline[7:10]]
    return [*protected, *reordered]


def fair_scores(record):
    return {doc: max(row["jina_score"] for row in types)
            for doc, types in record["scoring"]["documents"].items()}


def competitions(record, witnesses):
    scores = fair_scores(record)
    return {scope: prepare(record["baseline"], scores, witnesses,
                           challengers=record[ids], failure_reason=record["scoring"]["fallback_reason"])
            for scope, ids in (("fixed", "fixed"), ("augmented", "candidate_ids"))}


def gate_config(cap, quality, margin, novelty):
    return GateConfig(max_replacements=cap, min_quality=quality, min_margin=margin,
                      require_margin=margin > 0, novelty_overlap=novelty)


def main():
    import time
    from analysis.besa_v0.experiment import read

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "development", "null"))
    stage = parser.parse_args().stage
    deadline = time.monotonic() + read(CLOCK)["deadline_epoch"] - time.time()
    started = perf_counter()
    docs, chunks = documents_and_chunks()
    chunk_map = {c.chunk_id: c for c in chain.from_iterable(chunks.values())}
    names = tuple(sorted({d.metadata["source"] for d in docs.values() if d.metadata["source"]}))
    timing = []
    scores = RUN / f"{stage}_scores.jsonl"
    if not scores.is_file():
        raise FileNotFoundError(scores)

    def records():
        for record in rows(scores):
            check_time(deadline)
            start = perf_counter()
            outcome = make_witnesses(record, chunk_map, names)
            outcome["wall_seconds"] = perf_counter() - start
            timing.append(outcome["wall_seconds"])
            yield outcome
            if len(timing) % 50 == 0:
                print(f"{stage} witness: {len(timing)}", flush=True)

    write_rows(RUN / f"{stage}_witnesses.jsonl", records())
    print(f"{stage} witness: {len(timing)}", flush=True)
    write(RUN / f"{stage}_witness_performance.json", {
        "queries": len(timing), "verification_seconds": sum(timing), "wall_seconds": perf_counter() - started,
    })


if __name__ == "__main__":
    main()

