import json
import tempfile
import unittest
from pathlib import Path

from hoprag.dataset import load_dataset, stratified_sample
from hoprag.models import Evidence, Query


class DatasetTests(unittest.TestCase):
    def write_release(self, directory, corpus, queries):
        directory = Path(directory)
        (directory / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
        (directory / "MultiHopRAG.json").write_text(json.dumps(queries), encoding="utf-8")
        return directory

    def test_parses_real_release_fields_and_maps_evidence_url(self):
        document = {
            "title": "Title", "body": "A fact.", "url": "https://example.test/a",
            "author": None, "source": "Source", "published_at": "2023-01-01",
            "category": "science", "additional": "preserve",
        }
        query = {
            "query": "Which fact?", "answer": "A fact", "question_type": "inference_query",
            "evidence_list": [
                {"fact": "A fact.", "url": document["url"], "author": None},
                {"fact": "Unresolved", "url": "https://example.test/missing"},
            ],
        }
        null = {
            "query": "Unknown?", "answer": "Insufficient information.",
            "question_type": "null_query", "evidence_list": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            data = load_dataset(self.write_release(directory, [document, document], [query, null]))
        self.assertEqual(len(data.documents), 1)
        article = data.documents[0]
        self.assertEqual((article.document_id, article.text), ("doc-0001", "A fact."))
        self.assertIsNone(article.metadata["author"])
        self.assertEqual(article.metadata["additional"], "preserve")
        first, second = data.queries[0].evidence_list
        self.assertEqual((first.evidence_id, first.source_document_id), ("q-0001-e01", "doc-0001"))
        self.assertEqual(first.metadata["url"], document["url"])
        self.assertIsNone(second.source_document_id)
        self.assertFalse(data.queries[1].is_answerable)
        self.assertEqual(data.audit["duplicates"]["document_rows"], 1)
        self.assertEqual(data.audit["evidence_document_mapping"]["unmatched_url"], 1)

    def test_ambiguous_url_does_not_choose_arbitrary_source(self):
        documents = [
            {"title": title, "body": title, "url": "https://example.test/shared"}
            for title in ("First", "Second")
        ]
        queries = [{
            "query": "What?", "answer": "First", "question_type": "inference_query",
            "evidence_list": [{"fact": "First", "url": documents[0]["url"]}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            data = load_dataset(self.write_release(directory, documents, queries))
        self.assertEqual(len(data.documents), 2)
        self.assertIsNone(data.queries[0].evidence_list[0].source_document_id)
        self.assertEqual(data.audit["evidence_document_mapping"]["ambiguous_url"], 1)

    def test_rejects_invalid_evidence_and_missing_required_text(self):
        documents = [{"title": "Title", "body": "Body"}]
        query = {
            "query": "Question?", "answer": "Answer", "question_type": "comparison_query",
            "evidence_list": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_dataset(self.write_release(directory, documents, [query]))
            with self.assertRaisesRegex(ValueError, "row 1.body"):
                load_dataset(self.write_release(directory, [{"title": "Title"}], [query]))

    def test_stratified_sample_is_reproducible_and_includes_rare_strata(self):
        queries = [
            Query(str(i), "Question?", "Answer", "inference_query", (Evidence(str(i), "Fact"),))
            for i in range(12)
        ]
        queries.extend([
            Query("rare", "Other?", "Answer", "inference_query", (Evidence("a", "A"), Evidence("b", "B"))),
            Query("null", "Unknown?", "Insufficient information.", "null_query"),
        ])
        first = stratified_sample(queries, 5, seed=42)
        self.assertEqual(first, stratified_sample(queries, 5, seed=42))
        self.assertEqual(len({query.query_id for query in first}), 5)
        self.assertEqual({(query.question_type, len(query.evidence_list)) for query in first}, {
            ("inference_query", 1), ("inference_query", 2), ("null_query", 0),
        })
        self.assertEqual(stratified_sample(queries, len(queries)), tuple(queries))
        with self.assertRaises(ValueError):
            stratified_sample(queries, len(queries) + 1)
