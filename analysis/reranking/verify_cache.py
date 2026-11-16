"""Replay dev_5 rankings from the score cache without loading a model."""

import json
from time import perf_counter

from .cache import RerankCache, score_pool
from .core import aggregate, fuse, make_pairs
from .pool import iter_pools, load_splits
from .run import OUTPUT, RESULTS, write_json


class CacheOnly:
    def score(self, query, texts):
        raise AssertionError(f"Warm replay unexpectedly needs {len(texts)} inferences")


def main():
    stored = {}
    with RESULTS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "dev_5":
                stored[row["run_id"], row["query_id"], row["method"]] = row["ranking"]
    pools = list(iter_pools(load_splits()["dev_5"]))
    results = []
    for run_id in sorted({key[0] for key in stored}):
        started = perf_counter()
        model = json.loads((OUTPUT / f"model_{run_id}.json").read_bytes())
        methods = ("B3", "B5_minilm") if run_id.startswith("minilm") else ("B4", "B5_jina")
        n = 0
        with RerankCache(OUTPUT / "rerank_cache" / "scores.sqlite3", model) as cache:
            for pool in pools:
                pairs = make_pairs(pool)
                scores, actual, _ = score_pool(CacheOnly(), cache, pool, pairs, model["batch_size"])
                assert actual == 0
                ce = aggregate(pool, pairs, (score.score for score in scores))
                for method, ranking in zip(methods, (ce, fuse(pool, ce)), strict=True):
                    if ranking != stored[run_id, pool.query_id, method]:
                        raise ValueError(f"Warm ranking mismatch: {run_id} {pool.query_id} {method}")
                n += len(pairs)
        results.append({
            "run_id": run_id, "queries": len(pools), "cached_pairs": n,
            "inferred_pairs": 0, "ranking_identical": True, "elapsed_seconds": perf_counter() - started,
        })
    write_json(OUTPUT / "cache_reopen_check.json", results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
