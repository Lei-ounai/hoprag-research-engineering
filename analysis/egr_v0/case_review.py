"""Compare the real round-eight changes against round-nine fair evidence."""

from collections import Counter

from analysis.besa_v0.experiment import read, rows, write

from .run import OUT
from .evaluate import joined


def review(baseline, incoming, witnesses, candidates):
    return [{"document_id": doc, "activity": "incumbent" if doc in baseline else "challenger",
             "in_fixed_pool": candidates.get(doc, {}).get("in_fixed", False),
             "archived_expansion": candidates.get(doc, {}).get("expansion"),
             "verdicts": [{"chunk_id": w["chunk_id"], "status": w["status"], "reasons": w["reasons"],
                           "excerpt": w["evidence_excerpt"], "score": w["jina_score"],
                           "model_visible_chunk_chars": w["model_visible_chunk_chars"],
                           "features": w["features"]} for w in witnesses.get(doc, ())]}
            for doc in incoming]


def main():
    before = read(OUT / "round8_audit.json")["development_cases"]
    observed = {}
    for record, current in joined("development"):
        qid = record["query_id"]
        if qid not in before:
            continue
        previous = before[qid]
        ws, candidates = current["witnesses"], record["candidates"]
        baseline = record["baseline"][:10]
        prior_changes = {}
        for old in ("e5", "e6"):
            result = previous[old]
            if result is None:
                continue
            top = result["top10"].split()
            added = [d for d in top if d not in baseline]
            removed = [d for d in baseline if d not in top]
            prior_changes[old] = {"transition": result["transition"], "new_docs": review(baseline, added, ws, candidates),
                                  "old_docs": review(baseline, removed, ws, candidates)}
        observed[qid] = {"question": record["question"], "split": previous["split"],
                         "eighth_candidate_recovered": review(baseline, previous["new_sources_e3"], ws, candidates),
                         "eighth_replacement_experiment": prior_changes}
    if observed.keys() != before.keys():
        raise ValueError("A round-eight review case is missing in round nine")
    counts = Counter()
    for case in observed.values():
        historical = case["eighth_replacement_experiment"].get("e5")
        if historical and historical["transition"] == "regressed":
            counts["e5_regression_questions"] += 1
            for doc in historical["new_docs"]:
                verdicts = doc["verdicts"]
                counts["e5_regression_added_documents"] += 1
                counts["e5_regression_added_scored"] += bool(verdicts)
                counts["e5_regression_added_supported"] += any(r["status"] == "supported" for r in verdicts)
    write(OUT / "round8_case_review.json", {"counts": dict(counts), "records": observed})
    print(dict(counts))


if __name__ == "__main__":
    main()

