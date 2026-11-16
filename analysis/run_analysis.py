"""Reproduce fourth-round curves and attributions from frozen experiment data."""

from collections import Counter, defaultdict
import csv
from itertools import product
import json
from pathlib import Path

from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize
from hoprag.retrievers import BM25Retriever, _tokens

from .rankings import (
    METHODS,
    CachedDenseDiagnostics,
    compact_ranking,
    iter_rankings,
    load_cached_vectors,
    load_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline"
OUTPUT = ROOT / "analysis"
CUTOFFS = (1, 5, 10, 20, 50, 100)
INTERNAL_CUTOFFS = (1, 4, 5, 10, 20, 50, 100)
CATEGORIES = (
    "CANDIDATE_MISS", "RANKING_FAILURE", "CHUNK_MAPPING_FAILURE",
    "EVALUATION_MISMATCH", "CORPUS_GAP", "UNCERTAIN",
)


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def covered(rank, cutoff):
    return rank is not None and rank <= cutoff


def fact_index(queries, bodies, chunks):
    facts = {_normalize(e.evidence_text) for q in queries for e in q.evidence_list}
    exact = {
        fact: tuple(doc_id for doc_id, body in bodies.items() if fact in body)
        for fact in sorted(facts)
    }
    normalized_chunks = {
        doc_id: [(chunk.chunk_id, _normalize(chunk.text)) for chunk in items]
        for doc_id, items in chunks.items()
    }
    return exact, normalized_chunks


def first_rank(doc_ids, ranks):
    return min((ranks[doc_id] for doc_id in doc_ids if doc_id in ranks), default=None)


def prepare_evidence(query, ranks, exact_facts, chunks):
    result = []
    for evidence in query.evidence_list:
        fact = _normalize(evidence.evidence_text)
        literal_docs = exact_facts[fact]
        doc_id = evidence.source_document_id
        custom_docs = (doc_id,) if doc_id else literal_docs
        first = {m: first_rank(custom_docs, ranks[m]) for m in METHODS}
        supporting = [chunk_id for chunk_id, text in chunks.get(doc_id, ()) if fact in text]
        result.append({
            "evidence_id": evidence.evidence_id,
            "text": evidence.evidence_text,
            "source_document_id": doc_id,
            "source_url": evidence.metadata.get("url"),
            "first_ranks": {m: rank if covered(rank, 100) else None for m, rank in first.items()},
            "literal_first_ranks": {m: first_rank(literal_docs, ranks[m]) for m in METHODS},
            "rrf_pool_rank": first["rrf"],
            "rrf_pool_present": first["rrf"] is not None,
            "chunk_status": "contained" if supporting else "cross_boundary" if doc_id in literal_docs else "missing",
            "supporting_chunk_ids": supporting,
            "source_contains_literal": doc_id in literal_docs,
            "literal_document_ids": literal_docs,
        })
    return result


def metric_values(query, evidence, method, cutoff):
    found = sum(covered(e["first_ranks"][method], cutoff) for e in evidence)
    gold_doc_ids = {e["source_document_id"] for e in evidence if e["source_document_id"]}
    source_ranks = sorted({
        e["first_ranks"][method] for e in evidence if covered(e["first_ranks"][method], cutoff)
    })
    literal = {
        _normalize(e["text"]): e["literal_first_ranks"][method]
        for e in evidence if covered(e["literal_first_ranks"][method], cutoff)
    }
    fact_ap = sum(1 / rank for rank in literal.values()) / min(len(query.evidence_list), cutoff)
    return {
        "hits_at_k_literal": int(bool(literal)),
        "mrr_at_k_literal": 1 / min(literal.values()) if literal else 0.0,
        "map_at_k_fact_formula": fact_ap,
        "fact_map_queries_above_one": int(fact_ap > 1),
        "map_at_k_source_document": (
            sum(i / rank for i, rank in enumerate(source_ranks, start=1)) / min(len(gold_doc_ids), cutoff)
        ),
        "hits_at_k_source": int(bool(found)),
        "evidence_recall_at_k": found / len(evidence),
        "complete_evidence_coverage_at_k": int(found == len(evidence)),
        "evidence_entries_found": found,
    }


def classify(evidence):
    if covered(evidence["first_ranks"]["rrf"], 10):
        return None, None
    if not evidence["source_document_id"]:
        # Without a unique source, distinguish an absent literal/source from an unresolved origin.
        if not evidence["literal_document_ids"] and evidence["source_url"]:
            return "CORPUS_GAP", "SOURCE_UNRESOLVED_AND_NO_LITERAL_OCCURRENCE"
        return "UNCERTAIN", "SOURCE_UNRESOLVED"
    if not evidence["source_contains_literal"]:
        return "UNCERTAIN", "SOURCE_URL_RESOLVES_BUT_LITERAL_UNCONFIRMED"
    if evidence["first_ranks"]["rrf"] is not None:
        return "RANKING_FAILURE", "RRF_RANK_11_TO_100"
    if evidence["rrf_pool_present"]:
        return "CANDIDATE_MISS", "IN_FUSION_POOL_BELOW_RRF_100"
    return "CANDIDATE_MISS", "OUTSIDE_BOTH_BRANCH_TOP100"


def question_category(evidence):
    priorities = (
        "CORPUS_GAP", "CHUNK_MAPPING_FAILURE", "EVALUATION_MISMATCH",
        "UNCERTAIN", "CANDIDATE_MISS", "RANKING_FAILURE",
    )
    seen = {e["category"] for e in evidence}
    return next(item for item in priorities if item in seen)


def complement(combinations, scope, evidence, cutoff):
    patterns = [tuple(int(covered(e["first_ranks"][m], cutoff)) for m in METHODS) for e in evidence]
    combinations[scope, "evidence", cutoff].update(patterns)
    complete = tuple(int(all(pattern[i] for pattern in patterns)) for i in range(len(METHODS)))
    combinations[scope, "question_complete", cutoff][complete] += 1


def complement_rows(combinations):
    return [
        {
            "scope": scope, "unit": unit, "k": cutoff,
            "bm25_hit": pattern[0], "dense_hit": pattern[1], "rrf_hit": pattern[2],
            "count": counts[pattern], "denominator": sum(counts.values()),
        }
        for (scope, unit, cutoff), counts in sorted(combinations.items())
        for pattern in product((0, 1), repeat=3)
    ]


def assert_baseline(metrics, baseline):
    for method in METHODS:
        old = baseline["metrics"]["hybrid" if method == "rrf" else method]
        comparisons = (
            (4, "hits_at_k_literal", old["official"]["Hits@4"]),
            (10, "hits_at_k_literal", old["official"]["Hits@10"]),
            (10, "mrr_at_k_literal", old["official"]["MRR@10"]),
            (10, "map_at_k_fact_formula", old["official"]["MAP@10"]),
            (4, "evidence_recall_at_k", old["custom"]["Evidence Recall@4"]),
            (10, "evidence_recall_at_k", old["custom"]["Evidence Recall@10"]),
            (4, "complete_evidence_coverage_at_k", old["custom"]["Complete Evidence Coverage@4"]),
            (10, "complete_evidence_coverage_at_k", old["custom"]["Complete Evidence Coverage@10"]),
        )
        for cutoff, name, expected in comparisons:
            if abs(metrics[method, cutoff][name] - expected) > 1e-12:
                raise ValueError(f"Original metric mismatch: {method} {name} @{cutoff}")


def main():
    manifest = read_json(BASELINE / "run.json")
    config = manifest["config"]
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    if dataset.audit["files"] != read_json(BASELINE / "dataset_audit.json")["files"]:
        raise ValueError("Raw dataset bytes no longer match the baseline audit")
    documents = {d.document_id: d for d in dataset.documents}
    bodies = {doc_id: _normalize(doc.text) for doc_id, doc in documents.items()}
    chunks = load_chunks(BASELINE / "chunks.jsonl", documents, config["max_chunk_bytes"])
    literal, normalized_chunks = fact_index(dataset.queries, bodies, chunks)

    cache = ROOT / "data" / "cache" / Path(manifest["embedding_cache"]).name
    vectors, cache_audit = load_cached_vectors(
        cache, config["embedding_model"], config["embedding_dimensions"], chunks, dataset.queries,
    )
    write_json(OUTPUT / "cache_audit.json", cache_audit)
    replay = None
    if cache_audit["missing_corpus_vectors"] or cache_audit["missing_query_vectors"] or cache_audit["read_error"]:
        print(f"Read-only Dense diagnostics skipped: {cache_audit}", flush=True)
    else:
        replay = CachedDenseDiagnostics(chunks, vectors)

    bm25 = BM25Retriever(dataset.documents)
    document_terms = {doc_id: set(_tokens(f"{d.title}\n{d.text}")) for doc_id, d in documents.items()}
    metrics = defaultdict(Counter)
    combinations = defaultdict(Counter)
    question_categories, evidence_categories = Counter(), Counter()
    subtypes, by_question_subtype, stages, strata = Counter(), Counter(), Counter(), Counter()
    chunk_diagnostics, lexical_diagnostics, match_differences = Counter(), Counter(), Counter()
    union_diagnostics = Counter()
    branch_pool_miss_details = []
    top10_evidence_rows = []
    total_questions = total_evidence = failure_count = 0
    rows = iter_rankings(BASELINE / "rankings_full.jsonl", dataset.queries, documents.keys(), config)
    failure_path = OUTPUT / "failure_cases.jsonl"
    pending_failures = failure_path.with_suffix(".jsonl.tmp")
    with pending_failures.open("w", encoding="utf-8", newline="\n") as target:
        for query, rankings in rows:
            if query.question_type == "null_query":
                continue
            total_questions += 1
            total_evidence += len(query.evidence_list)
            ranks = {m: {row.document_id: row.rank for row in rankings[m]} for m in METHODS}
            evidence = prepare_evidence(query, ranks, literal, normalized_chunks)
            unique_keys = {(e["source_document_id"], _normalize(e["text"])) for e in evidence}
            if len(unique_keys) != len(evidence):
                raise ValueError(f"Unexpected duplicate required evidence for {query.query_id}")

            for method in METHODS:
                for cutoff in INTERNAL_CUTOFFS:
                    metrics[method, cutoff].update(metric_values(query, evidence, method, cutoff))
                for e in evidence:
                    if covered(e["first_ranks"][method], 10) != covered(e["literal_first_ranks"][method], 10):
                        match_differences[method] += 1
            for cutoff in CUTOFFS:
                complement(combinations, "all_answerable", evidence, cutoff)
            union_diagnostics["complete_using_union_of_branch_top10"] += int(all(
                any(covered(e["first_ranks"][m], 10) for m in ("bm25", "dense")) for e in evidence
            ))
            union_diagnostics["complete_in_union_of_branch_top100"] += int(all(e["rrf_pool_present"] for e in evidence))
            for e in evidence:
                first = e["first_ranks"]
                top10_evidence_rows.append({
                    "query_id": query.query_id,
                    "evidence_id": e["evidence_id"],
                    "source_document_id": e["source_document_id"],
                    **{f"{m}_rank_in_top100": first[m] for m in METHODS},
                    **{f"{m}_hit_at10": int(covered(first[m], 10)) for m in METHODS},
                })
                if covered(first["rrf"], 10) and not any(covered(first[m], 10) for m in ("bm25", "dense")):
                    branch_count = sum(first[m] is not None for m in ("bm25", "dense"))
                    union_diagnostics[f"rrf_only_top10_evidence_from_{branch_count}_branches"] += 1

            missing = []
            for e in evidence:
                e["category"], e["subtype"] = classify(e)
                if e["category"] is not None:
                    missing.append(e)
            if not missing:
                continue

            failure_count += 1
            category = question_category(missing)
            question_categories[category] += 1
            evidence_categories.update(e["category"] for e in missing)
            subtypes.update(e["subtype"] for e in missing)
            by_question_subtype.update({e["subtype"] for e in missing})
            strata[query.question_type, len(evidence)] += 1
            for cutoff in CUTOFFS:
                complement(combinations, "rrf_top10_failures", evidence, cutoff)
            complete_cutoff = next(
                (k for k in (20, 50, 100) if all(covered(e["first_ranks"]["rrf"], k) for e in evidence)),
                None,
            )
            stages[str(complete_cutoff) if complete_cutoff else "incomplete_at_100"] += 1

            full_bm25 = {row.document_id: row.rank for row in bm25.search(query, len(documents))}
            for row in rankings["bm25"]:
                if full_bm25.get(row.document_id) != row.rank:
                    raise ValueError(f"Full BM25 replay rank mismatch: {query.query_id}")
            dense_scores = replay.scores(query.question, rankings["dense"]) if replay else None
            terms = set(_tokens(query.question))
            for e in evidence:
                doc_id = e["source_document_id"]
                e["bm25_rank_full"] = full_bm25.get(doc_id)
                e["shared_query_fact_terms"] = sorted(terms & set(_tokens(e["text"])))
                e["shared_query_source_term_count"] = len(terms & document_terms.get(doc_id, set()))
                e["dense_diagnostic"] = (
                    replay.inspect(dense_scores, doc_id, e["supporting_chunk_ids"])
                    if dense_scores is not None and doc_id in documents else None
                )
                if e["category"] is None:
                    continue
                chunk_diagnostics[e["chunk_status"]] += 1
                if e["dense_diagnostic"] is not None:
                    chunk_diagnostics["missing_dense_winner_contains_fact"] += int(
                        e["dense_diagnostic"]["winning_chunk_contains_fact"]
                    )
                if e["subtype"] == "OUTSIDE_BOTH_BRANCH_TOP100":
                    lexical_diagnostics["pool_miss_evidence_entries"] += 1
                    lexical_diagnostics["pool_miss_has_any_bm25_candidate"] += int(e["bm25_rank_full"] is not None)
                    lexical_diagnostics["pool_miss_no_shared_fact_terms"] += int(not e["shared_query_fact_terms"])
                    branch_pool_miss_details.append({
                        "query_id": query.query_id,
                        "evidence_id": e["evidence_id"],
                        "source_document_id": doc_id,
                        "bm25_rank_full": e["bm25_rank_full"],
                        "dense_rank_full": (e["dense_diagnostic"] or {}).get("document_rank_full"),
                        "chunk_status": e["chunk_status"],
                        "shared_query_source_term_count": e["shared_query_source_term_count"],
                    })

            result = {
                "query_id": query.query_id,
                "question": query.question,
                "question_type": query.question_type,
                "answer": query.answer,
                "required_evidence_count": len(evidence),
                "matched_evidence_count": len(evidence) - len(missing),
                "missing_evidence": missing,
                "category": category,
                "subtypes": sorted({e["subtype"] for e in missing}),
                "rrf_first_complete_cutoff": complete_cutoff,
                "evidence": evidence,
                "rankings": {method: compact_ranking(rankings[method]) for method in METHODS},
                "ranking_prefix_cutoffs": [10, 20, 50, 100],
            }
            target.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")

    if (total_questions, total_evidence, failure_count) != (2255, 6084, 741):
        raise ValueError(f"Original question/evidence/failure counts changed: {total_questions, total_evidence, failure_count}")

    averages = {
        key: {
            name: value if name in ("fact_map_queries_above_one", "evidence_entries_found") else value / total_questions
            for name, value in counts.items()
        }
        for key, counts in metrics.items()
    }
    assert_baseline(averages, read_json(BASELINE / "metrics_full.json"))
    pending_failures.replace(failure_path)
    curve = [
        {
            "method": method, "k": k, "answerable_queries": total_questions,
            **averages[method, k],
            "complete_questions": int(metrics[method, k]["complete_evidence_coverage_at_k"]),
        }
        for method in METHODS for k in CUTOFFS
    ]
    write_csv(OUTPUT / "coverage_curve.csv", curve)
    write_csv(OUTPUT / "retrieval_complementarity.csv", complement_rows(combinations))
    write_csv(OUTPUT / "evidence_complementarity_top10.csv", top10_evidence_rows)
    summary = {
        "answerable_queries": total_questions,
        "evidence_entries": total_evidence,
        "rrf_top10_failures": failure_count,
        "question_categories": {key: question_categories[key] for key in CATEGORIES},
        "missing_evidence_categories": {key: evidence_categories[key] for key in CATEGORIES},
        "missing_evidence_subtypes": dict(subtypes),
        "questions_by_subtype_nonexclusive": dict(by_question_subtype),
        "first_complete_cutoff_among_top10_failures": dict(stages),
        "rrf_top10_failure_strata": [
            {"question_type": kind, "evidence_count": number, "questions": n}
            for (kind, number), n in sorted(strata.items())
        ],
        "top10_evidence_source_vs_literal_disagreements": {m: match_differences[m] for m in METHODS},
        "missing_evidence_chunk_diagnostics": dict(chunk_diagnostics),
        "candidate_pool_miss_lexical_diagnostics": dict(lexical_diagnostics),
        "candidate_pool_miss_details": branch_pool_miss_details,
        "branch_union_diagnostics": dict(union_diagnostics),
        "cache": cache_audit,
        "baseline_top10_ranks_and_existing_metrics_reproduced": True,
    }
    write_json(OUTPUT / "analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
