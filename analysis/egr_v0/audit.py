"""Revisit the recorded round-eight changes before running EGR."""

from collections import Counter
import csv
from pathlib import Path

from analysis.besa_v0.experiment import rows, write
from analysis.ege_v0.run import OUT as EGE


OUT = Path(__file__).resolve().parent
RECOVERY = "q-1781"
REGRESSIONS = {"q-0880", "q-2416"}
NULL_SAMPLES = {"q-0014", "q-0113", "q-1326", "q-1451"}


def main():
    transitions = {"E5": {}, "E6": {}}
    with (EGE / "final_ranking_comparison.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["method"]
            if label in transitions and row["transition"] != "unchanged":
                transitions[label][row["query_id"]] = row
    assert Counter(x["transition"] for x in transitions["E5"].values()) == {"regressed": 30, "recovered": 3}
    assert set(transitions["E6"]) == REGRESSIONS | {RECOVERY}
    assert transitions["E6"][RECOVERY]["transition"] == "recovered"
    records = {}
    for row in rows(EGE / "candidate_expansion_results.jsonl"):
        methods = row["offline_comparisons"]
        qid = row["query_id"]
        outcome = methods["E3"]["candidate_complete"] and not methods["E0"]["candidate_complete"]
        if not outcome and qid not in transitions["E5"] and qid not in transitions["E6"]:
            continue
        support = {
            label: [{key: hit[key] for key in ("document_id", "chunk_id", "term", "path", "retrieval_rank")}
                    for hit in row["lexical_controls"][label]["new_candidates"]]
            for label in ("E3", "E5")
        }
        records[qid] = {"question": row["question"], "split": row["split"], "one_hop_candidate_recovered": outcome,
                        "new_sources_e3": methods["E3"]["new_required_source_ids"],
                        "e5": transitions["E5"].get(qid), "e6": transitions["E6"].get(qid),
                        "expansion_witnesses": support}
    candidate_recovered = sorted(q for q, info in records.items() if info["one_hop_candidate_recovered"])
    assert len(candidate_recovered) == 6
    rerank = {qid: item for qid, item in records.items() if qid in transitions["E6"]}
    routes = {}
    for row in rows(EGE / "expansion_paths.jsonl"):
        if row["query_id"] in rerank and row["method"] == "E5":
            routes.setdefault(row["query_id"], []).append({"term": row["term"], "hits": row["hits"]})
    assert set(routes) == set(rerank)
    null_report = (EGE / "no_answer_expansion_analysis.md").read_text(encoding="utf-8")
    assert all(qid in null_report for qid in NULL_SAMPLES)
    write(OUT / "round8_audit.json", {
        "candidate_recovered_e3": candidate_recovered, "lexical_regressions": sorted(q for q, r in transitions["E5"].items()
                                                                                   if r["transition"] == "regressed"),
        "jina_recovered": [RECOVERY], "jina_regressed": sorted(REGRESSIONS),
        "reviewed_null_case_ids": sorted(NULL_SAMPLES), "development_cases": records,
    })
    print("Reviewed 6 one-hop pool recoveries, 30 lexical regressions, 1 Jina recovery and 2 Jina regressions")


if __name__ == "__main__":
    main()

