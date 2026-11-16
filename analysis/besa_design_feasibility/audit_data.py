"""Audit BGE vectors and parent aggregation on frozen P5 candidates without scoring a new rank."""

from collections import Counter
from csv import DictReader
from hashlib import file_digest
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from analysis.rankings import load_cached_vectors, load_chunks, text_hash
from analysis.reranking.pool import iter_pools, load_splits, read_manifest
from analysis.reranking_optimization.policies import select
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize as literal_normalize
from hoprag.retrievers import _normalize as unit_normalize


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results" / "baseline"
ROUND6 = ROOT / "analysis" / "reranking_optimization"
OUTPUT = Path(__file__).resolve().parent


def describe(values):
    series = np.asarray(values, dtype=np.float64)
    return {
        "count": len(series),
        "mean": float(series.mean()),
        "p50": float(np.median(series)),
        "p95": float(np.percentile(series, 95)),
        "max": float(series.max()),
    }


def score_gap(maximum, centroid):
    return {
        **describe(np.asarray(maximum) - centroid),
        "centroid_higher": int(np.count_nonzero(np.asarray(centroid) - maximum > 1e-6)),
        "maximum_higher": int(np.count_nonzero(np.asarray(maximum) - centroid > 1e-6)),
    }


def source_statistics(pools, queries, texts, parent_slices, query_cosines):
    stats = Counter()
    for pool in pools:
        candidate_ids = {doc.document_id for doc in pool.candidates}
        for item in queries[pool.query_id].evidence_list:
            stats["development_facts"] += 1
            if item.source_document_id not in candidate_ids:
                stats["facts_missing_parent_from_p5"] += 1
                continue
            stats["facts_with_p5_parent"] += 1
            fact = literal_normalize(item.evidence_text)
            parent = parent_slices[item.source_document_id]
            matching = np.fromiter((fact in text for text in texts[parent]), dtype=bool)
            if not matching.any():
                stats["p5_facts_crossing_literal_chunk_boundaries"] += 1
                continue
            stats["p5_facts_contained_in_one_or_more_chunks"] += 1
            scores = query_cosines[pool.query_id][parent]
            winning_index = int(np.argmax(scores))
            if matching[winning_index]:
                stats["p5_facts_in_dense_winning_chunk"] += 1
            else:
                stats["p5_facts_absent_from_dense_winning_chunk"] += 1
    return dict(stats)


def stored_pool_rows():
    with (ROUND6 / "candidate_pool_metrics.csv").open(encoding="utf-8", newline="") as stream:
        return {
            row["split"]: {
                "queries": int(row["query_count"]),
                "mean_candidates": float(row["mean_candidate_documents"]),
                "total_chunk_pairs": int(row["total_candidate_pairs"]),
                "complete_pool_queries": int(row["candidate_complete_queries"]),
                "complete_pool_coverage": float(row["candidate_complete_evidence_coverage"]),
                "pool_evidence_recall": float(row["candidate_evidence_recall"]),
            }
            for row in DictReader(stream) if row["pool_id"] == "P5"
        }


def cache_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run():
    read_manifest(verify=True)
    baseline = json.loads((BASELINE / "run.json").read_bytes())
    frozen = json.loads((ROUND6 / "frozen_reranking_config.json").read_bytes())
    if (frozen["policy"], frozen["scheme"], frozen["weight_rrf"], frozen["weight_jina"]) != ("P5", "B", 2, 1):
        raise ValueError("Round 6 P5-B identity changed")
    splits = load_splits()
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    documents = {doc.document_id: doc for doc in dataset.documents}
    query_by_id = {query.query_id: query for query in dataset.queries}
    chunks = load_chunks(BASELINE / "chunks.jsonl", documents, baseline["config"]["max_chunk_bytes"])
    cache_path = ROOT / "data" / "cache" / Path(baseline["embedding_cache"]).name
    before_sha = cache_sha(cache_path)
    vectors, cache = load_cached_vectors(
        cache_path, baseline["config"]["embedding_model"],
        baseline["config"]["embedding_dimensions"], chunks, dataset.queries,
    )
    if cache["read_error"] or any(cache[key] for key in (
        "missing_corpus_vectors", "missing_query_vectors", "invalid_expected_vectors",
    )):
        raise ValueError(f"Incomplete vector cache: {cache}")

    parent_ids = sorted(chunks)
    parent_index = {doc_id: index for index, doc_id in enumerate(parent_ids)}
    offsets = np.cumsum([0, *(len(chunks[doc_id]) for doc_id in parent_ids)])
    parent_slices = {
        doc_id: slice(int(offsets[index]), int(offsets[index + 1]))
        for index, doc_id in enumerate(parent_ids)
    }
    ordered = [chunk for doc_id in parent_ids for chunk in chunks[doc_id]]
    text_literals = [literal_normalize(chunk.text) for chunk in ordered]
    matrix = unit_normalize(np.stack([vectors[text_hash(chunk.text)] for chunk in ordered]))
    averaged = np.stack([matrix[parent_slices[doc_id]].mean(axis=0) for doc_id in parent_ids])
    centroids = unit_normalize(averaged)

    pools = [select(pool, "P5") for pool in iter_pools(splits["answerable_dev"])]
    p5_source_counts = stored_pool_rows()
    if len(pools) != p5_source_counts["answerable_dev"]["queries"]:
        raise ValueError("Audited P5 development cohort does not match stored summary")
    query_cosines = {}
    pooling_query_max, pooling_query_centroid = [], []
    co_candidate_pairs = set()
    candidate_chunk_counts = Counter()
    for pool in pools:
        query_vector = unit_normalize(vectors[text_hash(pool.question)][None, :])[0]
        chunk_scores = np.einsum("ij,j->i", matrix, query_vector)
        centroid_scores = np.einsum("ij,j->i", centroids, query_vector)
        query_cosines[pool.query_id] = chunk_scores
        candidate_ids = sorted(doc.document_id for doc in pool.candidates)
        candidate_chunk_counts.update(len(chunks[doc_id]) for doc_id in candidate_ids)
        max_scores = np.maximum.reduceat(chunk_scores, offsets[:-1])
        indexes = [parent_index[doc_id] for doc_id in candidate_ids]
        pooling_query_max.extend(max_scores[indexes])
        pooling_query_centroid.extend(centroid_scores[indexes])
        co_candidate_pairs.update(combinations(candidate_ids, 2))

    pooling_graph_max, pooling_graph_centroid = [], []
    for left, right in sorted(co_candidate_pairs):
        left_matrix = matrix[parent_slices[left]]
        right_matrix = matrix[parent_slices[right]]
        pooling_graph_max.append(float((left_matrix @ right_matrix.T).max()))
        pooling_graph_centroid.append(float(centroids[parent_index[left]] @ centroids[parent_index[right]]))

    cache["sha256_unchanged"] = before_sha == cache_sha(cache_path)
    if not cache["sha256_unchanged"]:
        raise ValueError("Vector cache changed during this audit")
    return {
        "corpus": {
            "documents": len(documents), "chunks": len(ordered),
            "queries": len(dataset.queries),
            "document_metadata_keys": sorted({key for doc in documents.values() for key in doc.metadata}),
            "explicit_document_links": sum("links" in doc.metadata for doc in documents.values()),
            "chunk_count_distribution": dict(sorted(Counter(map(len, chunks.values())).items())),
            "duplicate_chunk_ids": len(ordered) - len({chunk.chunk_id for chunk in ordered}),
            "chunk_parent_map_verified_against_saved_partition": True,
        },
        "embedding_cache": cache,
        "p5_stored_aggregate": p5_source_counts,
        "p5_development_candidate_chunk_count_distribution": dict(sorted(candidate_chunk_counts.items())),
        "p5_development_direct_query_max_minus_centroid": score_gap(pooling_query_max, pooling_query_centroid),
        "p5_development_distinct_co_candidate_parent_pairs": len(co_candidate_pairs),
        "p5_development_parent_pair_max_minus_centroid": score_gap(pooling_graph_max, pooling_graph_centroid),
        "p5_development_gold_fact_diagnostic": source_statistics(
            pools, query_by_id, text_literals, parent_slices, query_cosines,
        ),
    }


if __name__ == "__main__":
    report = json.dumps(run(), ensure_ascii=False, indent=2) + "\n"
    (OUTPUT / "data_audit.json").write_text(report, encoding="utf-8")
    print(report, end="")
