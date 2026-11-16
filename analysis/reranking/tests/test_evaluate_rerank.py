from __future__ import annotations

import unittest

from analysis.reranking.evaluate import (
    Group, _metrics_and_details, complete_at_10,
    validate_reranking, worst_rank,
)
from hoprag.models import Document, Evidence, Query, RetrievalResult


class RerankingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.documents = tuple(
            Document(f"doc-{i}", f"Title {i}", f"fact number {i} here.")
            for i in range(1, 13)
        )
        self.bodies = {doc.document_id: doc.text.replace(" ", "") for doc in self.documents}
        self.q1 = Query("q1", "Question?", "Answer", "inference_query", (
            Evidence("e1", "fact number 1 here", "doc-1"),
            Evidence("e2", "number 1 here", "doc-1"),
        ))
        self.q2 = Query("q2", "Other question?", "Answer", "comparison_query", (
            Evidence("e3", "fact number 11 here", "doc-11"),
        ))
        self.null = Query("q-null", "Unavailable answer?", "Insufficient information.", "null_query")
        self.queries = {q.query_id: q for q in (self.q1, self.q2, self.null)}
        self.before = list(range(1, 13))
        self.after = [11, *range(2, 11), 12, 1]

    def ranking(self, query, order, method="B2"):
        return [
            RetrievalResult(query.query_id, f"doc-{doc}", method, 0.5 / rank, rank)
            for rank, doc in enumerate(order, start=1)
        ]

    def test_recovery_regression_and_individual_evidence(self):
        cohort = (self.q1, self.q2)
        baseline = {q.query_id: {"B2": self.ranking(q, self.before)} for q in cohort}
        after = {q.query_id: self.ranking(q, self.after, "B3") for q in cohort}
        group = Group("dev_5", "test-run", "B3", tuple(after), after)
        score, details, nulls = _metrics_and_details(group, self.queries, self.documents, self.bodies, baseline, 5)
        self.assertEqual(score["candidate_pool_evidence_coverage"], 1.0)
        self.assertEqual(score["complete_evidence_coverage_at_10"], 0.5)
        self.assertEqual(score["reranking_recovery_rate"], 0.5)
        self.assertEqual(score["rrf_failure_rescue_rate"], 1.0)
        self.assertEqual(score["ranking_regression_rate"], 1.0)
        self.assertEqual(score["evidence_lost_from_top10"], 2)
        self.assertEqual(score["evidence_gained_into_top10"], 1)
        self.assertEqual(score["evidence_worsened_by_at_least_10"], 2)
        self.assertFalse(score["complete_split"])
        self.assertEqual([(d["rrf_rank"], d["rank"]) for d in details], [(1, 12), (1, 12), (11, 1)])
        self.assertEqual(nulls, [])

    def test_missing_rank_is_not_complete(self):
        self.assertIsNone(worst_rank([1, None]))
        self.assertFalse(complete_at_10([1, None]))
        self.assertEqual(worst_rank([1, 12]), 12)
        self.assertFalse(complete_at_10([1, 12]))
        self.assertTrue(complete_at_10([1, 10]))

    def test_requires_every_parent_and_checks_top10(self):
        original = self.ranking(self.q1, self.before)
        scores = [
            {"document_id": row.document_id, "rank": row.rank, "score": row.score}
            for row in original
        ]
        record = {"query_id": "q1", "method": "B3", "ranking": scores, "top10": scores[:10]}
        self.assertEqual(len(validate_reranking(record, original)), 12)
        with self.assertRaises(ValueError):
            validate_reranking({**record, "ranking": scores[:10]}, original)
        with self.assertRaises(ValueError):
            validate_reranking({**record, "top10": scores[1:11]}, original)
        with self.assertRaises(ValueError):
            validate_reranking({**record, "ranking": [*scores[:-1], {**scores[-1], "document_id": "doc-1"}]}, original)

    def test_null_excluded_and_scores_reported(self):
        baseline = {self.null.query_id: {"B2": self.ranking(self.null, self.before)}}
        after = {self.null.query_id: self.ranking(self.null, self.after, "B3")}
        group = Group("null_dev", "test-run", "B3", tuple(after), after)
        score, details, nulls = _metrics_and_details(group, self.queries, self.documents, self.bodies, baseline, 1)
        self.assertIsNone(score["hits_at_10"])
        self.assertIsNone(score["ranking_regression_rate"])
        self.assertEqual(score["null_queries"], 1)
        self.assertEqual(details, [])
        self.assertEqual(nulls[0]["candidate_count"], 12)
        self.assertEqual(nulls[0]["top1_document_id"], "doc-11")
        self.assertEqual(nulls[0]["top10_shared_with_rrf"], 9)


if __name__ == "__main__":
    unittest.main()
