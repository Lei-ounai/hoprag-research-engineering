"""Offline-only audit of literal necessary facts in historical EGR model windows."""

from collections import Counter
from statistics import median
from time import perf_counter

from analysis.besa_v0.experiment import rows, write
from analysis.egr_v0.run import ROOT, RUN, documents_and_chunks
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize

from .build import OUT


def summarize_visibility():
    start = perf_counter()
    _, chunks = documents_and_chunks()
    chunk_by_id = {block.chunk_id: block for blocks in chunks.values() for block in blocks}
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    annotations = {q.query_id: q for q in dataset.queries}
    counters, model = Counter(), {}
    for stage in ("development", "null"):
        stats = Counter()
        sizes, lengths = [], []
        for record in rows(RUN / f"{stage}_scores.jsonl"):
            stats["questions"] += 1
            docs = record["scoring"]["documents"]
            for selections in docs.values():
                stats["query_document_pairs"] += 1
                stats[f"parents_scored_{len(selections)}_chunks"] += 1
                for chunk in selections:
                    stats["scored_pairs"] += 1
                    stats["truncated_pairs"] += chunk["truncated"]
                    stats["pairs_with_complete_question"] += len(record["question"]) == chunk["model_visible_query_chars"]
                    sizes.append(chunk["model_visible_chunk_chars"])
                    lengths.append(chunk["input_tokens"])
            if stage == "null":
                continue
            fixed = set(record["fixed"])
            baseline_top10 = set(record["baseline"][:10])
            for fact in annotations[record["query_id"]].evidence_list:
                doc = fact.source_document_id
                fixed_hit = doc in fixed
                challenger = fixed_hit and doc not in baseline_top10
                counters["annotated_facts"] += 1
                counters["in_fixed_pool"] += fixed_hit
                counters["in_combined_pool"] += doc in docs
                counters["fixed_challenger_facts"] += challenger
                if doc not in docs:
                    continue
                expected = _normalize(fact.evidence_text)
                any_block = any(expected in _normalize(block.text) for block in chunks[doc])
                selected = docs[doc]
                any_selected = any(expected in _normalize(chunk_by_id[row["chunk_id"]].text) for row in selected)
                any_visible = any(expected in _normalize(chunk_by_id[row["chunk_id"]].text[:row["model_visible_chunk_chars"]])
                                  for row in selected)
                counters["literal_any_original_chunk"] += any_block
                counters["literal_scored_full_chunk"] += any_selected
                counters["literal_model_visible_prefix"] += any_visible
                counters["fixed_challenger_literal_any_original_chunk"] += challenger and any_block
                counters["fixed_challenger_literal_scored_full_chunk"] += challenger and any_selected
                counters["fixed_challenger_literal_model_visible_prefix"] += challenger and any_visible
        model[stage] = {**stats, "model_visible_chunk_char_range": [min(sizes), median(sizes), max(sizes)],
                        "input_token_range": [min(lengths), median(lengths), max(lengths)]}
    return {"model_input_profile": model, "development_gold_offline_literal_counts": dict(counters),
            "matching": "Historical normalization removes plain spaces and newline; case and punctuation preserved",
            "unit": "one annotated fact; source mapping is first checked independently",
            "semantic_ground_truth": False, "new_jina_pairs": 0, "new_embeddings": 0,
            "wall_seconds": perf_counter() - start}


def main():
    report = summarize_visibility()
    write(OUT / "literal_visibility_summary.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()

