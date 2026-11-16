import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from multihop import Document, Retriever, load_corpus


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        # Synthetic documents keep query terms separate from the bridge.
        self.documents = (
            Document("atlas", "Atlas survey", "Atlas survey field records.", ("oak", "carter")),
            Document("oak", "Oak ridge", "Pine forest restoration records."),
            Document("carter", "Carter notebook", "Maritime archives.", ("coastal", "atlas")),
            Document("coastal", "Coastal observatory", "Stargazing site beside the harbor."),
        )
        self.retriever = Retriever(self.documents)
        self.question = "Which observatory did the Atlas survey identify?"

    def test_two_hops_can_cross_a_bridge_without_query_terms(self):
        one_hop = self.retriever.search(self.question, max_hops=1, top_k=10)
        two_hops = self.retriever.search(self.question, max_hops=2, top_k=10)

        self.assertNotIn(("atlas", "carter", "coastal"), self.paths(one_hop))
        self.assertIn(("atlas", "carter", "coastal"), self.paths(two_hops))

    def test_relevant_endpoint_ranks_above_an_unrelated_branch(self):
        documents = (
            Document("atlas", "Atlas survey", "Atlas survey field records.", ("oak", "coastal")),
            self.documents[1],
            self.documents[3],
        )
        results = Retriever(documents).search(self.question, max_hops=1, top_k=1)

        self.assertEqual(self.paths(results), [("atlas", "coastal")])

    def test_cycles_do_not_repeat_documents_in_a_path(self):
        results = self.retriever.search(self.question, max_hops=4, top_k=20)
        paths = self.paths(results)

        self.assertIn(("atlas", "carter", "coastal"), paths)
        self.assertTrue(all(len(path) == len(set(path)) for path in paths))

    def test_zero_hops_returns_only_seed_documents(self):
        results = self.retriever.search(self.question, max_hops=0)
        paths = self.paths(results)

        self.assertIn(("atlas",), paths)
        self.assertTrue(all(len(path) == 1 for path in paths))

    def test_question_without_matching_terms_returns_no_results(self):
        self.assertEqual(self.retriever.search("zeppelin"), [])

    @staticmethod
    def paths(results):
        return [tuple(document.id for document in result.documents) for result in results]


class CorpusTests(unittest.TestCase):
    def test_dangling_link_is_rejected(self):
        document = {
            "id": "atlas",
            "title": "Atlas survey",
            "text": "Field records.",
            "links": ["missing"],
        }
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_corpus(corpus)

    def test_cli_returns_linked_chinese_evidence_as_json(self):
        documents = (
            {
                "id": "atlas",
                "title": "北方巡查",
                "text": "北方巡查的记录涉及若干地点。",
                "links": ["coastal"],
                "source": "https://example.com/atlas",
            },
            {
                "id": "coastal",
                "title": "海岸观测站",
                "text": "观测站位于港口附近。",
                "links": [],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(
                "\n".join(json.dumps(document, ensure_ascii=False) for document in documents),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable, "-m", "multihop", str(corpus), "北方巡查涉及哪座观测站？",
                    "--max-hops", "1", "--top-k", "1", "--json",
                ],
                capture_output=True,
                check=True,
                encoding="utf-8",
                timeout=10,
            )

        results = json.loads(completed.stdout)
        evidence = results[0]["documents"]
        self.assertEqual([document["id"] for document in evidence], ["atlas", "coastal"])
        self.assertEqual(evidence[1]["title"], "海岸观测站")
        self.assertEqual(evidence[0]["source"], "https://example.com/atlas")


if __name__ == "__main__":
    unittest.main()
