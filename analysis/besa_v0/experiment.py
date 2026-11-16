"""Gold-free BESA development inputs, baseline replay, and scoring checkpoints."""

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from analysis.reranking.pool import iter_pools, load_splits
from analysis.reranking_optimization.policies import select as select_pool, weighted_fuse

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
R6 = OUT.parent / "reranking_optimization"
R5 = OUT.parent / "reranking"
AUDIT = OUT.parent / "besa_design_feasibility"
ARCHIVED = R6 / "development_rankings_answerable_dev.jsonl"
MODEL_SPEC = R5 / "model_jina-unified-256-3defdef1ae34.json"
BUDGETS = (30, 60, 90)
MAX_SECONDS = 7200
TOP_K = 10


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_rows(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def development_ids():
    ids = load_splits()["answerable_dev"]
    if len(set(ids)) != len(ids) or len(ids) != 451:
        raise ValueError("Unexpected frozen development split")
    return tuple(sorted(ids))


def frozen_split():
    ids = development_ids()
    ordered = sorted(ids, key=lambda qid: sha256(f"42:{qid}".encode()).digest())
    return {
        "method": "SHA-256(42:query_id), ascending; first 60% for tuning",
        "tune": sorted(ordered[:271]), "validation": sorted(ordered[271:]),
    }


def development_inputs(requested: Iterable[str] | None = None):
    ids = set(development_ids())
    requested = ids if requested is None else set(requested)
    if not requested <= ids:
        raise ValueError("This experiment only accepts frozen development IDs")
    saved = {r["query_id"]: r for r in rows(ARCHIVED) if r["pool_id"] == "P5" and r["query_id"] in requested}
    if saved.keys() != requested:
        raise ValueError("Missing archived P5-B development inputs")
    frozen = read(R6 / "frozen_reranking_config.json")
    if (frozen["policy"], frozen["weight_rrf"], frozen["weight_jina"], frozen["rrf_k"]) != ("P5", 2, 1, 60):
        raise ValueError("Frozen P5-B configuration mismatch")
    for raw in iter_pools(requested):
        pool = select_pool(raw, "P5")
        row = saved[pool.query_id]
        jina = row["jina_ranking"]
        full = weighted_fuse(pool, jina, 2, 1, 60)
        if full[:TOP_K] != row["final_top10_by_rrf_weight"]["2"]:
            raise ValueError(f"Historical P5-B replay diverged: {pool.query_id}")
        witnesses = {item["document_id"]: item["winning_chunk_id"] for item in jina}
        baseline = [item["document_id"] for item in full]
        if set(witnesses) != set(baseline) or len(baseline) != len(pool.candidates):
            raise ValueError("Historical ranking missing candidates")
        yield pool, baseline, witnesses


def cache_source():
    manifest = read(ROOT / "results" / "baseline" / "run.json")
    return ROOT / "data" / "cache" / Path(manifest["embedding_cache"]).name


def historical_scores():
    return (
        AUDIT / "experiment_cache" / "need_scores.sqlite3",
        R6 / "rerank_cache" / "scores.sqlite3",
        R5 / "rerank_cache" / "scores.sqlite3",
    )


def fallback_record(baseline, reason):
    return {
        "ranking": list(baseline), "document_ids": list(baseline[:TOP_K]),
        "choices": [], "fallback_reason": reason,
    }


def guarded_selection(pool, baseline, selector, **kwargs):
    try:
        output = selector(pool, baseline, **kwargs)
        top = output["document_ids"]
        if len(top) != min(TOP_K, len(baseline)) or len(top) != len(set(top)) or not set(top) <= set(baseline):
            raise ValueError("Invalid selector output")
        if output["fallback_reason"]:
            return fallback_record(baseline, output["fallback_reason"])
        selected = set(top)
        return {
            **output,
            "ranking": [*top, *(doc for doc in baseline if doc not in selected)],
        }
    except (TypeError, ValueError, ArithmeticError, KeyError) as error:
        return fallback_record(baseline, f"selector_invalid:{type(error).__name__}")


def assemble_scoring(pool, baseline, plan, scorer):
    if not plan.accepted:
        return [], plan.fallback_reason or "planner_rejected"
    if plan.question != pool.question or not 2 <= len(plan.needs) <= 4:
        return [], "invalid_need_plan"
    seen = set()
    for need in plan.needs:
        if not need.need_text.strip() or need.need_id in seen or not set(need.depends_on) <= seen:
            return [], "invalid_need_plan"
        seen.add(need.need_id)
    try:
        return scorer(), None
    except TimeoutError:
        raise
    except (OSError, sqlite3.Error):
        return [], "cache_or_io_error"
    except Exception as error:
        return [], f"scoring_error:{type(error).__name__}"


class Checkpoints:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS question_runs (
            scope TEXT NOT NULL, query_id TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(scope, query_id)
        ) WITHOUT ROWID""")

    def close(self):
        self.db.close()

    def get(self, scope, query_id):
        row = self.db.execute("SELECT payload FROM question_runs WHERE scope=? AND query_id=?",
                              (scope, query_id)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, scope, row):
        with self.db:
            self.db.execute("INSERT INTO question_runs VALUES (?,?,?)",
                            (scope, row["query_id"], json.dumps(row, ensure_ascii=False, allow_nan=False)))

    def export(self, scope):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM question_runs WHERE scope=? ORDER BY query_id", (scope,)
        )]
