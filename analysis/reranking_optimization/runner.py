"""CPU reranking with frozen query IDs, inherited scores, and a four-hour deadline."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
import json
from pathlib import Path
from random import Random
from time import perf_counter, process_time
import ctypes
import os

from analysis.reranking.cache import digest, score_pool
from analysis.reranking.core import TEMPLATE, aggregate, make_pairs
from analysis.reranking.pool import iter_pools, load_splits, read_manifest

from .policies import DEV_SUPERSET, POLICIES, POLICY_UNION, select
from .scoring import LayeredCache

OUTPUT = Path(__file__).resolve().parent
PARENT = OUTPUT.parent / "reranking"
HISTORY = PARENT / "rerank_cache" / "scores.sqlite3"
SCORES = OUTPUT / "rerank_cache" / "scores.sqlite3"
SPEC = PARENT / "model_jina-unified-256-3defdef1ae34.json"
CLOCK = OUTPUT / "budget_clock.json"
FROZEN = OUTPUT / "frozen_reranking_config.json"
THREADS = 4
BATCH_SIZE = 16
HOURS = 4
SEED = 42
COHORTS = ("answerable_dev", "answerable_heldout", "null_dev", "null_heldout")
FILES = {
    "pilot": "pilot_scores.jsonl",
    "development": "dev_scores.jsonl",
    "heldout": "heldout_scores.jsonl",
    "no-answer": "null_scores.jsonl",
    "no-answer-dev": "null_scores.jsonl",
}
IDENTITY_FIELDS = (
    "model_id", "revision", "model_sha256", "tokenizer_sha256",
    "effective_max_tokens", "context_mode", "truncation_strategy", "truncation_side", "batch_size",
)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc)


def start_clock():
    if not CLOCK.exists():
        start = now()
        write(CLOCK, {
            "started_utc": start.isoformat(), "deadline_utc": (start + timedelta(hours=HOURS)).isoformat(),
            "budget_hours": HOURS,
        })
    return read(CLOCK)


def remaining_seconds():
    if not CLOCK.exists():
        return HOURS * 3600
    return (datetime.fromisoformat(read(CLOCK)["deadline_utc"]) - now()).total_seconds()


def _lower_priority():
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError()


def identity(meta):
    return {k: meta[k] for k in IDENTITY_FIELDS}


def pilot_ids(splits):
    remaining = sorted(set(splits["answerable_dev"]) - set(splits["dev_100"]))
    return sorted(Random(SEED).sample(remaining, 20))


def _keys(pool):
    by_doc = {candidate.document_id: [] for candidate in pool.candidates}
    for pair in make_pairs(pool):
        by_doc[pair.document_id].append(digest(pair.text))
    return by_doc


def plan(meta, development_policy=DEV_SUPERSET):
    splits = load_splits()
    scopes = {qid: scope for scope in COHORTS for qid in splits[scope]}
    cohort = {name: {p: Counter() for p in (*POLICIES, DEV_SUPERSET, POLICY_UNION)} for name in COHORTS}
    membership = OUTPUT / "candidate_membership.jsonl"
    pending = membership.with_suffix(".jsonl.tmp")
    with LayeredCache(SCORES if SCORES.exists() else None, HISTORY, meta) as cache:
        with pending.open("w", encoding="utf-8", newline="\n") as target:
            for pool in iter_pools():
                scope = scopes[pool.query_id]
                known = cache.get(digest(pool.question))
                by_doc = _keys(pool)
                pools = {}
                for p in (*POLICIES, DEV_SUPERSET, POLICY_UNION):
                    if p in (DEV_SUPERSET, POLICY_UNION) and scope != "answerable_dev":
                        continue
                    selected = select(pool, p)
                    ids = [c.document_id for c in selected.candidates]
                    pools[p] = ids
                    hashes = {h for doc in ids for h in by_doc[doc]}
                    counts = cohort[scope][p]
                    counts["candidate_documents"] += len(ids)
                    counts["total_pairs"] += sum(len(by_doc[doc]) for doc in ids)
                    counts["unique_pairs"] += len(hashes)
                    counts["missing_pairs"] += len(hashes - known.keys())
                    counts["queries"] += 1
                target.write(json.dumps({"query_id":pool.query_id,"cohort":scope,"policies":pools}) + "\n")
    pending.replace(membership)
    measurements = [
        read(OUTPUT / f"{stage}_performance.json")["new_pairs_per_wall_second"]
        for stage in ("pilot", "development", "heldout")
        if (OUTPUT / f"{stage}_performance.json").is_file()
    ]
    speed = min((value for value in measurements if value > 0), default=None)
    left = remaining_seconds()
    projections = {}
    dev = cohort["answerable_dev"][development_policy]["missing_pairs"]
    for p in POLICIES:
        heldout = cohort["answerable_heldout"][p]["missing_pairs"]
        null_dev = cohort["null_dev"][p]["missing_pairs"]
        null_heldout = cohort["null_heldout"][p]["missing_pairs"]
        projections[p] = {
            "remaining_dev_pairs": dev,
            "remaining_heldout_pairs": heldout,
            "remaining_null_dev_pairs": null_dev,
            "remaining_null_heldout_pairs": null_heldout,
            "dev_and_heldout_seconds": (dev + heldout) / speed if speed else None,
            "dev_heldout_and_all_null_seconds": (dev + heldout + null_dev + null_heldout) / speed if speed else None,
        }
    manifest = {
        "split_counts": {scope: len(splits[scope]) for scope in COHORTS},
        "historical_answerable_dev_cache_queries": len(splits["dev_100"]),
        "pilot_query_ids": pilot_ids(splits),
        "split_sha256": sha(PARENT / "splits.json"),
        "candidate_manifest_sha256": sha(PARENT / "candidate_pool_manifest.json"),
        "candidate_content_and_original_ranks": "../reranking/candidate_pool_manifest.json",
        "candidate_membership": membership.name,
        "candidate_membership_sha256": sha(membership),
        "model": identity(meta), "threads": THREADS, "cpu_priority": "below normal on Windows",
        "template": TEMPLATE, "pool_policy": "all original chunks, max raw score per parent",
        "weights_rrf": [1, 2, 3, 4], "weight_jina": 1, "rrf_k": 60,
        "rank_policy": "RRF and Jina ranks are compact within the same selected pool; original global ranks are retained in metadata; missing lane contributes zero",
        "development_scoring_policy": development_policy,
        "scoring_scope": "RRF Top50 union BM25 Top50 union Dense Top50; POLICY_UNION is the exact smaller six-policy union",
        "cohort_pairs": cohort, "projections": projections,
        "measured_pairs_per_second": speed, "seconds_left_at_plan": left,
        "budget_clock": read(CLOCK) if CLOCK.exists() else None,
    }
    path = OUTPUT / "experiment_manifest.json"
    if path.exists():
        previous = read(path)
        manifest["initial_preflight"] = previous.get("initial_preflight", previous["cohort_pairs"])
    else:
        manifest["initial_preflight"] = cohort
    write(path, manifest)
    print(json.dumps({"speed":speed,"seconds_left":left,"projections":projections}), flush=True)


def _stage_spec(stage, splits, meta, development_policy=DEV_SUPERSET):
    if stage == "pilot":
        return DEV_SUPERSET, pilot_ids(splits)
    if stage == "development":
        return development_policy, splits["answerable_dev"]
    frozen = read(FROZEN)
    if (frozen["state"] != "frozen" or frozen["model"] != identity(meta) or frozen["template"] != TEMPLATE
            or frozen["split_sha256"] != sha(PARENT / "splits.json")
            or frozen["candidate_manifest_sha256"] != sha(PARENT / "candidate_pool_manifest.json")
            or frozen["policy"] not in POLICIES or frozen["weight_rrf"] not in (1, 2, 3, 4)
            or frozen["weight_jina"] != 1 or frozen["rrf_k"] != 60):
        raise ValueError("Frozen heldout config does not match scoring inputs")
    ids = {
        "heldout": splits["answerable_heldout"],
        "no-answer": [*splits["null_dev"], *splits["null_heldout"]],
        "no-answer-dev": splits["null_dev"],
    }[stage]
    return frozen["policy"], ids


class BudgetBackend:
    def __init__(self, backend):
        self.backend = backend

    def score(self, query, texts):
        if remaining_seconds() <= 5:
            raise TimeoutError("Four-hour scoring deadline reached; committed batches are reusable")
        return self.backend.score(query, texts)


def require_budget(stage, policy):
    manifest = read(OUTPUT / "experiment_manifest.json")
    inventory = manifest["cohort_pairs"]
    scopes = {
        "development": ["answerable_dev"], "heldout": ["answerable_heldout"],
        "no-answer": ["null_dev", "null_heldout"], "no-answer-dev": ["null_dev"],
    }[stage]
    needed = sum(inventory[scope][policy]["missing_pairs"] for scope in scopes)
    speed = manifest["measured_pairs_per_second"]
    if needed and (speed is None or needed / speed + 60 > remaining_seconds()):
        raise TimeoutError(f"{stage} {policy} does not fit the remaining four-hour budget")


def lock_heldout():
    path = OUTPUT / "heldout_run_lock.json"
    frozen_sha = sha(FROZEN)
    if path.exists():
        if read(path)["frozen_config_sha256"] != frozen_sha:
            raise ValueError("Cannot resume heldout after changing the frozen configuration")
    else:
        write(path, {"frozen_config_sha256": frozen_sha, "started_utc": now().isoformat()})


def run(stage, meta, development_policy=DEV_SUPERSET):
    from analysis.reranking.backend import FastEmbedBackend, peak_rss_bytes

    splits = load_splits()
    policy, ids = _stage_spec(stage, splits, meta, development_policy)
    outfile = OUTPUT / FILES[stage]
    profile_path = OUTPUT / f"{stage}_performance.json"
    if outfile.exists() and profile_path.exists():
        raise ValueError(f"Cohort already completed: {outfile}")
    if stage != "pilot" and not (OUTPUT / "pilot_performance.json").is_file():
        raise ValueError("Complete the 20-query throughput pilot before full scoring")
    if stage != "pilot" and not profile_path.exists():
        require_budget(stage, policy)
    if stage == "heldout":
        lock_heldout()
    read_manifest(verify=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    _lower_priority()
    start_clock()
    clock_started = perf_counter()
    cpu_started = process_time()
    backend = FastEmbedBackend(
        meta["model_id"], PARENT / "model_cache", threads=THREADS,
        context_mode="unified", batch_size=BATCH_SIZE, unified_limit=256,
    )
    if identity(backend.metadata) != identity(meta):
        raise ValueError("Local model or preprocessing differs from fifth-round Jina")
    stage_id = "|".join((stage, policy, sha(PARENT / "splits.json"), sha(PARENT / "candidate_pool_manifest.json")))
    totals = Counter()
    active = Counter()
    rows = []
    with LayeredCache(SCORES, HISTORY, backend.metadata) as cache:
        for index, full_pool in enumerate(iter_pools(ids), 1):
            row = cache.checkpoint_get(stage_id, full_pool.query_id)
            if row is None:
                t = perf_counter()
                pool = select(full_pool, policy)
                pairs = make_pairs(pool)
                counts = cache.inspect(pool, pairs)
                old = cache.get(digest(pool.question))
                unseen = set()
                new_by_doc = Counter()
                for pair in pairs:
                    key = digest(pair.text)
                    if key not in old and key not in unseen:
                        new_by_doc[pair.document_id] += 1
                        unseen.add(key)
                scored, inferred, infer_seconds = score_pool(BudgetBackend(backend), cache, pool, pairs, BATCH_SIZE)
                if inferred != counts.missing_pairs:
                    raise ValueError("Pair inventory changed while scoring")
                ce = aggregate(pool, pairs, (s.score for s in scored))
                row = {
                    "query_id": pool.query_id,
                    "scores": {r["document_id"]: r["score"] for r in ce},
                    "winning_chunks": {r["document_id"]: r["winning_chunk_id"] for r in ce},
                    "candidate_documents": len(ce), "pair_count": len(pairs), "new_pairs": inferred,
                    "cache_hits": len(pairs) - inferred, "history_hits": counts.historical_hits,
                    "new_pairs_by_document": dict(new_by_doc),
                    "inference_seconds": infer_seconds, "wall_seconds": perf_counter() - t,
                    "truncated_pairs": sum(s.truncated for s in scored),
                }
                cache.checkpoint_put(stage_id, pool.query_id, row)
                active.update({key: row[key] for key in ("new_pairs", "wall_seconds", "inference_seconds")})
            for key in ("pair_count", "new_pairs", "cache_hits", "history_hits", "inference_seconds", "wall_seconds", "truncated_pairs"):
                totals[key] += row[key]
            rows.append(row)
            if index == len(ids) or index % 10 == 0:
                print(f"{stage} {index}/{len(ids)} queries, {totals['new_pairs']} inferred pairs, "
                      f"{remaining_seconds()/3600:.2f}h left", flush=True)
    if len(rows) != len(ids) or {row["query_id"] for row in rows} != set(ids):
        raise ValueError("Scoring output does not match the frozen cohort")
    temp = outfile.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
    elapsed = perf_counter() - clock_started
    profile = {
        "stage": stage, "policy": policy, "queries": len(rows), "query_ids": sorted(ids),
        **dict(totals), "process_wall_seconds": elapsed,
        "process_cpu_seconds": process_time() - cpu_started,
        "model_load_seconds": backend.metadata["load_seconds"], "peak_working_set_bytes": peak_rss_bytes(),
        "active_new_pairs": active["new_pairs"], "active_query_wall_seconds": active["wall_seconds"],
        "new_pairs_per_wall_second": totals["new_pairs"] / totals["wall_seconds"] if totals["wall_seconds"] else 0,
        "cache_hit_rate": totals["cache_hits"] / totals["pair_count"],
        "remaining_budget_seconds": remaining_seconds(),
    }
    if not profile_path.exists():
        write(profile_path, profile)
    temp.replace(outfile)
    print(json.dumps({k:v for k,v in profile.items() if k != "query_ids"}), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("plan", *FILES))
    parser.add_argument("--development-policy", choices=(DEV_SUPERSET, POLICY_UNION), default=DEV_SUPERSET)
    args = parser.parse_args()
    meta = read(SPEC)
    if args.stage == "plan":
        read_manifest(verify=True)
        plan(meta, args.development_policy)
    else:
        run(args.stage, meta, args.development_policy)


if __name__ == "__main__":
    main()
