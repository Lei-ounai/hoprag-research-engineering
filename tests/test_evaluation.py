import unittest
from pathlib import Path

from hoprag.evaluation import evaluate
from hoprag.graph_regressions import run_graph_regressions
from hoprag.models import Document, Evidence, Query, RetrievalResult


def _document(document_id, text):
    return Document(document_id, f"Title {document_id}", text, {})


def _evidence(evidence_id, fact, document_id=None):
    return Evidence(evidence_id, fact, document_id, {})


def _query(query_id, evidence, question_type="inference_query"):
    return Query(query_id, f"Question {query_id}", "answer", question_type, tuple(evidence))


def _ranking(query_id, document_ids):
    return [
        RetrievalResult(query_id, document_id, "test", 1.0 / rank, rank)
        for rank, document_id in enumerate(document_ids, start=1)
    ]


class RetrievalMetricTests(unittest.TestCase):
    def test_hand_computed_metrics_and_cutoffs(self):
        texts = [
            "irrelevant one",
            "start Alpha\nfact finish",
            "another copy of Alpha fact",
            "irrelevant four",
            "Beta fact",
            "irrelevant six",
            "irrelevant seven",
            "irrelevant eight",
            "irrelevant nine",
            "Gamma fact",
            "Delta fact",
        ]
        ids = [f"d{rank}" for rank in range(1, 12)]
        documents = [_document(doc_id, text) for doc_id, text in zip(ids, texts, strict=True)]
        queries = [
            _query("complete", [
                _evidence("a", "Alpha fact", "d2"),
                _evidence("b", "Beta fact", "d5"),
                _evidence("c", "Gamma fact", "d10"),
            ]),
            _query("too_late", [_evidence("d", "Delta fact", "d11")]),
            _query("unanswerable", [], "null_query"),
        ]
        metrics = evaluate(queries, documents, {
            "complete": _ranking("complete", ids),
            "too_late": _ranking("too_late", ids),
            "unanswerable": _ranking("unanswerable", ids[:5]),
        })

        self.assertEqual(metrics["answerable_queries"], 2)
        self.assertEqual(metrics["official"]["Hits@4"], 1 / 2)
        self.assertEqual(metrics["official"]["Hits@10"], 1 / 2)
        self.assertAlmostEqual(metrics["official"]["MAP@10"], (1 / 2 + 1 / 5 + 1 / 10) / 3 / 2)
        self.assertEqual(metrics["official"]["MRR@10"], 1 / 2 / 2)
        self.assertEqual(metrics["custom"]["Evidence Recall@4"], 1 / 3 / 2)
        self.assertEqual(metrics["custom"]["Evidence Recall@10"], 1 / 2)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@4"], 0)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@10"], 1 / 2)
        self.assertEqual(metrics["unanswerable"], {
            "query_count": 1,
            "queries_with_results": 1,
            "queries_without_results": 0,
            "average_returned_documents@4": 4.0,
            "average_returned_documents@10": 5.0,
        })

    def test_official_fact_matching_and_custom_source_matching_are_distinct(self):
        documents = [
            _document("source", "Same fact"),
            _document("copy", "Same fact and Fallback fact"),
        ]
        query = _query("q", [
            _evidence("a", "Same fact", "source"),
            _evidence("b", "Fallback fact"),
        ])
        metrics = evaluate([query], documents, {"q": _ranking("q", ["copy"])})

        self.assertEqual(metrics["official"]["MAP@10"], 1)
        self.assertEqual(metrics["custom"]["Evidence Recall@4"], 1 / 2)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@4"], 0)

    def test_distinct_facts_from_one_document_count_individually(self):
        document = _document("source", "First fact and Second fact")
        query = _query("q", [
            _evidence("a", "First fact", "source"),
            _evidence("b", "Second fact", "source"),
            _evidence("c", "Unresolved fact"),
        ])
        metrics = evaluate([query], [document], {"q": _ranking("q", ["source"])})

        self.assertEqual(metrics["custom"]["Evidence Recall@4"], 2 / 3)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@10"], 0)

    def test_different_cutoffs_and_complete_early_evidence(self):
        documents = [_document(f"d{rank}", f"Fact {rank}") for rank in range(1, 6)]
        ids = [document.document_id for document in documents]
        early = _query("early", [_evidence("early", "Fact 4", "d4")])
        late = _query("late", [_evidence("late", "Fact 5", "d5")])
        metrics = evaluate([early, late], documents, {
            "early": _ranking("early", ids),
            "late": _ranking("late", ids),
        })

        self.assertEqual(metrics["official"]["Hits@4"], 1 / 2)
        self.assertEqual(metrics["official"]["Hits@10"], 1)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@4"], 1 / 2)
        self.assertEqual(metrics["custom"]["Complete Evidence Coverage@10"], 1)

    def test_only_null_queries_leave_answerable_metrics_undefined(self):
        query = _query("null", [], "null_query")
        metrics = evaluate([query], [], {"null": []})

        self.assertEqual(metrics["answerable_queries"], 0)
        self.assertTrue(all(value is None for value in metrics["official"].values()))
        self.assertTrue(all(value is None for value in metrics["custom"].values()))
        self.assertEqual(metrics["unanswerable"]["queries_without_results"], 1)

    def test_incomplete_or_duplicate_rankings_raise(self):
        document = _document("known", "One fact")
        query = _query("q", [_evidence("e", "One fact", "known")])

        with self.assertRaisesRegex(ValueError, "expected exactly one query and a ranking"):
            evaluate([query], [document], {})
        with self.assertRaisesRegex(ValueError, "duplicate retrieved document"):
            evaluate([query], [document], {"q": _ranking("q", ["known", "known"])})
        with self.assertRaisesRegex(ValueError, "unknown retrieved document"):
            evaluate([query], [document], {"q": _ranking("q", ["missing"])})


class StaticGraphRegressionTests(unittest.TestCase):
    def test_reproduces_known_defects_on_unmodified_example_corpus(self):
        example = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"
        report = run_graph_regressions(example)

        wallace = report["wallace_entity_switch"]
        self.assertTrue(wallace["top_path_wrong"])
        self.assertIsNone(wallace["expected_evidence_chain"][1]["document_id"])
        edge = report["missing_key_edge"]
        self.assertTrue(edge["complete_chain_present_original"])
        self.assertFalse(edge["complete_chain_present_ablated"])
        self.assertTrue(edge["defect_reproduced"])
        self.assertTrue(edge["wrong_path_present_ablated"])


if __name__ == "__main__":
    unittest.main()
