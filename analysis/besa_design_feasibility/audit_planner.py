"""Human annotation of the 35 frozen, question-only planner outputs."""

from collections import Counter
from hashlib import sha256
import json

from .planner_probe import OUT, ROOT, _rows, _write


def main() -> None:
    ratings_path = OUT / "planner_manual_ratings.txt"
    ratings = {}
    for line in ratings_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        query_id, status, risks, links, rationale = line.split("|", 4)
        if query_id in ratings:
            raise ValueError(f"Duplicate rating: {query_id}")
        ratings[query_id] = status, risks, links, rationale

    inputs = {row["query_id"]: row["question"] for row in _rows(OUT / "planner_inputs.jsonl")}
    outputs = {row["query_id"]: row for row in _rows(OUT / "planner_outputs.jsonl")}
    if inputs.keys() != outputs.keys() or outputs.keys() != ratings.keys():
        raise ValueError("Manual ratings must cover the full frozen input and output sample")
    raw = json.loads((ROOT / "data/raw/multihop_rag/MultiHopRAG.json").read_bytes())
    split = json.loads((ROOT / "analysis/reranking/splits.json").read_bytes())
    corpus = json.loads((ROOT / "data/raw/multihop_rag/corpus.json").read_bytes())
    document_ids = {row["url"]: f"doc-{index:04d}" for index, row in enumerate(corpus, 1)}
    if len(document_ids) != len(corpus):
        raise ValueError("Source URL identity is ambiguous")
    evidence = []
    for query_id, question in inputs.items():
        if query_id not in split["answerable_dev"]:
            raise ValueError(f"Question not in the original development split: {query_id}")
        gold = raw[int(query_id.removeprefix("q-")) - 1]
        if gold["query"] != question:
            raise ValueError(f"Changed input question: {query_id}")
        status, risks, links, rationale = ratings[query_id]
        mapping = {}
        if links != "none":
            for link in links.split():
                need_id, positions = link.split(":")
                mapping[need_id] = [f"{query_id}-e{int(position):02d}" for position in positions.split(",")]
        evidence_document = {
            f"{query_id}-e{index:02d}": document_ids[row["url"]]
            for index, row in enumerate(gold["evidence_list"], 1)
        }
        required = list(evidence_document)
        mapped = {evidence_id for ids in mapping.values() for evidence_id in ids}
        by_need = {need["need_id"]: need for need in outputs[query_id]["needs"]}
        if mapping.keys() - by_need.keys() or mapped - set(required):
            raise ValueError(f"Invalid evidence map: {query_id}")
        if any(by_need[need_id]["need_role"] in {"join", "original_question_fallback"} for need_id in mapping):
            raise ValueError("Join and fallback nodes are not counted as independent retrieval needs")
        if (status == "original_fallback") != (outputs[query_id]["disposition"] == "fallback_original"):
            raise ValueError(f"Human ratings disagree with the planner fallback record: {query_id}")
        evidence.append({
            "query_id": query_id,
            "question_type": gold["question_type"],
            "gold_evidence_count": len(required),
            "status": status,
            "risks": risks.split(",") if risks else [],
            "need_to_gold_evidence_ids": mapping,
            "need_to_gold_source_document_ids": {
                need_id: list(dict.fromkeys(evidence_document[item] for item in ids))
                for need_id, ids in mapping.items()
            },
            "gold_evidence_ids_without_a_matching_need": [item for item in required if item not in mapped],
            "rationale": rationale,
        })
    _write(OUT / "planner_eval.jsonl", evidence)
    states = Counter(row["status"] for row in evidence)
    risks = Counter(risk for row in evidence for risk in row["risks"])
    by_cell = Counter((row["question_type"], row["gold_evidence_count"]) for row in evidence)
    summary = {
        "rows": len(evidence),
        "status_counts": states,
        "risk_counts": risks,
        "by_stratum": {f"{kind}:{count}": total for (kind, count), total in sorted(by_cell.items())},
        "has_complete_gold_alignment": sum(not row["gold_evidence_ids_without_a_matching_need"] for row in evidence),
        "usable_complete_gold_alignment": sum(
            row["status"] == "usable" and not row["gold_evidence_ids_without_a_matching_need"] for row in evidence
        ),
        "sha256": {
            name: sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("planner_inputs.jsonl", "planner_outputs.jsonl", "planner_manual_ratings.txt")
        },
    }
    (OUT / "planner_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
