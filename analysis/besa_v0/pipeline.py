"""Apply a frozen BESA-v0 selector to Gold-free score records."""

import argparse
import json
from time import perf_counter

from .experiment import OUT, cache_source, development_inputs, fallback_record, guarded_selection, read, rows
from .scoring import DenseVectors, ScoredPair, summarize_scores
from .selector import select

MODES = ("C", "D", "E")
MMR_WEIGHT = 0.75
OUTPUT = OUT / "run"


def candidate_inputs(pool, baseline, witnesses, score_record, vectors):
    if baseline != score_record["baseline"]:
        raise ValueError("Scored record does not match replayed P5-B order")
    return {
        "pool": pool, "baseline": baseline, "witnesses": witnesses, "raw": score_record,
        "vectors": vectors.parent_vectors(pool),
        "pairs": [ScoredPair(**pair) for pair in score_record["scored_pairs"]],
        "candidates": {d.document_id: d for d in pool.candidates},
        "chunk_vectors": {c.chunk_id: vectors.chunk_vectors[c.chunk_id] for d in pool.candidates for c in d.chunks},
    }


def assemble(item, *, mode, budget=0, threshold=0.0, need_weight=0.55, repeat_weight=0.0):
    baseline = item["baseline"]
    raw = item["raw"]
    if mode == "A":
        return fallback_record(baseline, None), None, 0.0
    reason = raw["fallback_reason"] if mode in MODES else None
    if reason is not None:
        return fallback_record(baseline, reason), None, 0.0
    start = perf_counter()
    score = None
    if mode in MODES:
        score = summarize_scores(item["pairs"], budget, threshold=threshold)
        support = score["supports"]
        if not any(cell["support"] > 0 for need in support.values() for cell in need.values()):
            return fallback_record(baseline, "no_scores_above_gate"), score, perf_counter() - start
    else:
        support = None
    kwargs = dict(
        mode=mode, supports=support, parent_vectors=item["vectors"],
        chunk_vectors=item["chunk_vectors"], lambda_prior=1.0 - need_weight,
        lambda_need=need_weight, lambda_repeat=repeat_weight,
    )
    if mode == "B":
        kwargs["mmr_lambda"] = MMR_WEIGHT
    method = guarded_selection(item["pool"], baseline, select, **kwargs)
    return method, score, perf_counter() - start



def main():
    from .run import _chunks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_id")
    parser.add_argument("--arm", help="A, B, or a Tune-frozen arm such as C-B30")
    args = parser.parse_args()
    frozen = read(OUTPUT / "tune_freeze.json")
    arm = args.arm or frozen["selected_arm"]
    params = {"mode": arm} if arm in ("A", "B") else {
        key: value for key, value in frozen["frozen"][arm].items() if key != "config_id"
    }
    pool, baseline, witnesses = next(development_inputs([args.query_id]))
    record = next(row for row in rows(OUTPUT / "development_scores.jsonl") if row["query_id"] == args.query_id)
    vectors = DenseVectors.load(cache_source(), _chunks())
    item = candidate_inputs(pool, baseline, witnesses, record, vectors)
    selected, score, seconds = assemble(item, **params)
    print(json.dumps({
        "query_id": pool.query_id, "arm": arm, "plan": record["plan"], **selected,
        "scoring": {key: value for key, value in score.items() if key != "pairs"} if score else None,
        "selection_seconds": seconds,
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

