"""Offline provenance comparison for the frozen five-question cost probe."""

import json
from pathlib import Path

from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize
from analysis.reranking.pool import iter_pools
from analysis.reranking_optimization.policies import select


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
# Human alignment of question-derived needs to annotated source excerpts.
ALIGNMENT = {
    "q-1400": (1, 2), "q-1787": (3, 2, 1), "q-1407": (2, 1),
    "q-1648": (1, 2), "q-1492": (1, 2),
}


def main():
    scores = json.loads((OUT / "need_scoring_scores.json").read_bytes())
    assert set(ALIGNMENT) == {row["query_id"] for row in scores}
    data = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    qmap = {q.query_id: q for q in data.queries if q.query_id in ALIGNMENT}
    docs = {d.document_id: d for d in data.documents}
    candidates = {
        pool.query_id: {c.document_id: c for c in select(pool, "P5").candidates}
        for pool in iter_pools(ALIGNMENT)
    }
    rows = []
    for row in scores:
        qid, nid = row["query_id"], row["need_id"]
        ordinal = int(nid.removeprefix("n"))
        evidence = qmap[qid].evidence_list[ALIGNMENT[qid][ordinal - 1] - 1]
        ranked = row["ranking"]
        hit = next((doc for doc in ranked if doc["document_id"] == evidence.source_document_id), None)
        best = ranked[0]
        fact = _normalize(evidence.evidence_text)
        if hit:
            source = candidates[qid][hit["document_id"]]
            win = next(c for c in source.chunks if c.chunk_id == hit["winning_chunk_id"])
            chunk_hit = fact in _normalize(win.text)
            any_chunk = any(fact in _normalize(chunk.text) for chunk in source.chunks)
        else:
            chunk_hit = any_chunk = None
        rows.append({
            "query_id": qid, "need_id": nid, "evidence_id": evidence.evidence_id,
            "human_aligned_source_document_id": evidence.source_document_id,
            "source_rank": hit["rank"] if hit else None,
            "source_winning_chunk_contains_literal_fact": chunk_hit,
            "source_any_chunk_contains_literal_fact": any_chunk,
            "top_document_id": best["document_id"], "top_title": docs[best["document_id"]].title,
            "top_article_source": docs[best["document_id"]].metadata.get("source"),
            "top_document_score": best["score"],
        })
    summary = {
        "warning": "Hand-selected five development examples; source rank and literal clause are diagnostics, not entailment or generalization.",
        "aligned_needs": len(rows),
        "source_in_pool": sum(row["source_rank"] is not None for row in rows),
        "source_at_1": sum(row["source_rank"] == 1 for row in rows),
        "source_at_3": sum(0 < (row["source_rank"] or 0) <= 3 for row in rows),
        "source_at_10": sum(0 < (row["source_rank"] or 0) <= 10 for row in rows),
        "winning_chunk_literal": sum(row["source_winning_chunk_contains_literal_fact"] is True for row in rows),
        "rows": rows,
    }
    (OUT / "need_scoring_offline_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
