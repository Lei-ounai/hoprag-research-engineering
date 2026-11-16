"""Verify all EGR artifacts, limits, cohorts, and historical hashes."""

from collections import Counter
from datetime import datetime, timezone
from time import time
import csv
import sqlite3

from analysis.besa_v0.experiment import read, rows, write
from analysis.ege_v0.run import fingerprint

from .run import CLOCK, OUT, RUN, cohorts, documents_and_chunks, snapshot


def main():
    tune, valid, null = cohorts()
    allowed = {"tune": tune, "validation": valid, "null": null}
    historical = read(RUN / "historical_inputs_before.json")
    if historical != snapshot():
        raise ValueError("A frozen historical file or model digest changed")
    docs, chunks = documents_and_chunks()
    chunk_index = {chunk.chunk_id: chunk for parent in chunks.values() for chunk in parent}
    scored = {}
    lane_budget = Counter()
    score_stages = ("pilot", "development", "null")
    profile = {stage: read(RUN / f"{stage}_performance.json") for stage in score_stages}
    for stage, cohort in (("development", tune | valid), ("null", null)):
        seen = set()
        for record in rows(RUN / f"{stage}_scores.jsonl"):
            qid = record["query_id"]
            if qid not in cohort or qid in seen:
                raise ValueError(f"Out-of-scope scoring record: {qid}")
            seen.add(qid)
            doc_scores = record["scoring"]["documents"]
            candidate_ids = set(record["candidate_ids"])
            scored[qid] = {}
            if (record["scoring"]["fallback_reason"] or candidate_ids != doc_scores.keys()
                    or not set(record["baseline"]) <= set(record["fixed"]) <= candidate_ids
                    or len(candidate_ids) != len(record["candidate_ids"])):
                raise ValueError(f"Unexpected pool or scoring mismatch: {qid}")
            for doc, entries in doc_scores.items():
                expected = min(3, len(chunks[doc]))
                if len(entries) != expected or len({x["chunk_id"] for x in entries}) != expected:
                    raise ValueError(f"Budget mismatch: {qid}/{doc}")
                meta = record["candidates"][doc]
                prior = meta["p5b_rank"]
                lane = ("incumbent_8_10" if prior and 8 <= prior <= 10 else
                        "reserve_11_15" if prior and 11 <= prior <= 15 else
                        "top7" if prior and prior <= 7 else
                        "other_fixed" if meta["in_fixed"] else "one_hop_only")
                lane_budget[f"{lane}:{expected} chunks"] += 1
                for item in entries:
                    c = chunk_index[item["chunk_id"]]
                    if (c.document_id != doc or not 0 < item["model_visible_chunk_chars"] <= len(c.text)
                            or item["model_visible_query_chars"] != len(record["question"])):
                        raise ValueError(f"Missing model scope: {qid}/{c.chunk_id}")
                    scored[qid][c.chunk_id] = item["jina_score"]
            if len(scored[qid]) != record["scoring"]["actual_pairs"]:
                raise ValueError(f"Pair count mismatch: {qid}")
        if seen != cohort:
            raise ValueError("Scored cohort incomplete")

    current_witnesses = Counter()
    witness_seen = set()
    for row in rows(OUT / "evidence_witnesses.jsonl"):
        chunk = chunk_index[row["chunk_id"]]
        qid, pos = row["query_id"], row["chunk_id"]
        key = (qid, pos)
        if (qid not in allowed[row["split"]] or key in witness_seen or chunk.document_id != row["document_id"]
                or scored[qid][pos] != row["jina_score"]):
            raise ValueError(f"Invalid exported witness: {key}")
        witness_seen.add(key)
        if row["evidence_excerpt"]:
            start, end = row["evidence_start"], row["evidence_end"]
            if chunk.text[start:end] != row["evidence_excerpt"] or end > row["model_visible_chunk_chars"]:
                raise ValueError(f"Invalid original text offsets: {key}")
        current_witnesses[f"{row['split']}:{row['status']}"] += 1
    if len(witness_seen) != sum(len(r) for r in scored.values()):
        raise ValueError("Missing evidence witnesses")

    rankings = set()
    n_swaps = Counter()
    methods = {"E0", "E1", "E2"} | {f"E{i}-B{cap}" for i in (3, 4, 5) for cap in (1, 2, 3)}
    for row in rows(OUT / "replacement_log.jsonl"):
        qid, method, split, top = (row[k] for k in ("query_id", "method", "split", "top10"))
        baseline = row["baseline_top10"]
        if (qid not in allowed[split] or method not in methods or (qid, method) in rankings
                or len(top) != 10 or len(set(top)) != 10 or top[:7] != baseline[:7]
                or (split == "null" and any(row[key] is not None for key in
                                            ("recall_at10", "candidate_recall", "complete", "regression")))):
            raise ValueError(f"Invalid isolated/protected output: {qid}/{method}")
        rankings.add((qid, method))
        maximum = int(method[-1]) if "-B" in method else (0 if method == "E0" else 3)
        swaps = row["replacements"]
        if len(swaps) > maximum or len({x["candidate_document_id"] for x in swaps}) != len(swaps):
            raise ValueError("Replacement budget or uniqueness violation")
        for swap in swaps:
            given = swap["position"] - 1
            if (given not in (7, 8, 9) or baseline[given] != swap["incumbent_document_id"]
                    or top[given] != swap["candidate_document_id"]):
                raise ValueError(f"Replacement log does not reconstruct the result: {qid}/{method}")
            if "-B" in method and ((qid, swap["champion_chunk_id"]) not in witness_seen
                                  or swap["score_margin"] <= 0 or not swap["new_coverage_keys"]):
                raise ValueError(f"Gate-listed replacement lacks a comparable witness: {qid}/{method}")
            n_swaps[f"{split}:{method}"] += 1
    if len(rankings) != len(scored) * len(methods):
        raise ValueError("Incomplete ranking comparison table")
    with (OUT / "gate_decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = Counter((row["split"], row["method"], row["status"]) for row in csv.DictReader(stream))
    with sqlite3.connect((RUN / "scores.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Independent local Jina cache failed quick_check")
    before = read(CLOCK)
    manifest = OUT / "experiment_manifest.json"
    ended = read(manifest)["budget"]["first_verification_epoch"] if manifest.exists() else time()
    if ended > before["deadline_epoch"]:
        raise TimeoutError("First verified conclusion exceeded the experiment budget")
    files = [p for p in OUT.glob("*") if p.is_file() and p != manifest]
    files += sorted((OUT / "tests").glob("test_*.py"))
    files += [RUN / name for name in ("budget_clock.json", "historical_inputs_before.json", "evaluation_summary.json",
                                      "pilot_scores.jsonl", "development_scores.jsonl", "null_scores.jsonl")]
    files += sorted(RUN.glob("*_performance.json"))
    write(manifest, {
        "cohorts": {"tune": len(tune), "validation": len(valid), "null": len(null), "answerable_heldout_run": 0},
        "inputs_verified_unchanged": len(historical), "historical_artifacts": historical,
        "budget": {"maximum_seconds": before["deadline_epoch"] - before["started_epoch"],
                   "wall_seconds_start_to_first_verification": ended - before["started_epoch"],
                   "first_verification_epoch": ended,
                   "timestamp_utc": datetime.fromtimestamp(ended, timezone.utc).isoformat(),
                   "within_budget": True},
        "profile": profile, "embedding_requests": 0, "candidate_chunk_budgets": dict(lane_budget),
        "ranking_records_verified": len(rankings), "witness_records_verified": len(witness_seen),
        "witness_statuses": dict(current_witnesses), "corpus_parent_documents": len(docs),
        "replacement_counts": dict(n_swaps), "gate_decisions": {":".join(k): v for k, v in decisions.items()},
        "artifacts": {p.relative_to(OUT).as_posix(): fingerprint(p) for p in sorted(files)},
    })
    print(f"Verified {len(historical)} historical files, {len(rankings)} rankings, {len(witness_seen)} original witnesses")


if __name__ == "__main__":
    main()

