import json
from dataclasses import replace
import unittest

from hoprag.models import Document
from hoprag.retrievers import Chunk
from analysis.ege_v0.expansion import (
    ExpansionConfig, LexicalIndex, build_query, expand, extract_terms, select_seeds, through_hop,
)


QUESTION = "Which harbor permit was not approved before November 3, 2023?"


def corpus():
    titles_and_bodies = [
        ("Harbor permit", "The harbor permit was denied by Meridian Trust in 2023. "
                          "Last Tuesday the permit had an amendment. Harbor decisions and actions."),
        ("Port news", "The harbor permit was noted during the news. Council members took action."),
        ("Port calendar", "Routine harbor events happened last Tuesday. Entry permitted."),
        ("Local update", "Councillors published meeting minutes. Harbor permit information."),
        ("Town schedule", "Some local events are planned. Local music and tourism."),
        ("Archive record", "The harbor permit denied by Meridian Trust involved Oriel Council. "
                           "The harbor permit was not approved before November 3, 2023."),
        ("External record", "The action described by Oriel Council involved Arcadia Agency."),
        ("External report", "Arcadia Agency noted an external proceeding."),
        ("Summary for council", "The harbor permit was denied. Meridian Trust issued a statement."),
    ] + [(f"Filler {number}", "Routine harbor reporting and local news.") for number in range(30)]
    documents = []
    chunks = {}
    for sequence, (title, text) in enumerate(titles_and_bodies, 1):
        doc_id = f"doc-{sequence:03}"
        documents.append(Document(doc_id, title, text))
        texts = [f"{title}\n{text}"]
        if sequence == 1:
            breakpoint = texts[0].index("Last Tuesday")
            texts = [texts[0][:breakpoint], texts[0][breakpoint:]]
        chunks[doc_id] = [Chunk(f"{doc_id}::{part}", doc_id, value) for part, value in enumerate(texts)]
    return LexicalIndex(documents, chunks)


def stable(result):
    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items() if not key.endswith("_seconds")}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value
    return scrub(result)


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = corpus()
        cls.baseline = ["doc-001", "doc-002", "doc-003", "doc-004", "doc-005"]
        cls.jina = [{"document_id": "doc-001", "score": 3.5, "winning_chunk_id": "doc-001::0"},
                    {"document_id": "doc-002", "score": -2.0, "winning_chunk_id": "doc-001::1"}]

    def test_select_top_seeds_stably_from_baseline_and_verify_archive_winners(self):
        top_three = select_seeds(self.index, QUESTION, self.baseline, self.jina, 3)
        top_five = select_seeds(self.index, QUESTION, self.baseline, self.jina, 5)
        self.assertEqual(top_three, top_five[:3])
        self.assertEqual([row["document_id"] for row in top_five], self.baseline)
        self.assertEqual([row["original_rank"] for row in top_five], list(range(1, 6)))
        self.assertEqual(top_three[0]["jina_score"], 3.5)
        self.assertEqual(top_three[0]["chunk_id"], "doc-001::0")
        self.assertEqual(self.index.chunk_map[top_three[1]["chunk_id"]].document_id, "doc-002")

    def test_extract_low_frequency_name_and_retain_exact_original_witness(self):
        seed = select_seeds(self.index, QUESTION, self.baseline, self.jina, 3)[0]
        terms = extract_terms(self.index, QUESTION, seed, visited=self.baseline)
        match = next(row for row in terms if row["term"] == "Meridian Trust")
        self.assertEqual(match["term_kind"], "name")
        self.assertEqual(match["term_document_frequency"], 3)
        self.assertGreater(match["idf"], 0)
        self.assertIn("Meridian Trust", match["excerpt"])
        chunk = self.index.chunk_map[match["seed_chunk_id"]]
        document = self.index.documents[match["seed_document_id"]]
        offset = match["position"]
        original = f"{document.title}\n{document.text}"
        self.assertEqual(original[offset["start"]:offset["end"]], match["term"])
        self.assertEqual(chunk.text[offset["chunk_start"]:offset["chunk_end"]], match["term"])
        self.assertNotIn("harbor", {row["term"].casefold() for row in terms})
        self.assertNotIn("2023", {row["term"] for row in terms})

    def test_query_variants_use_original_constraints_and_exact_clue(self):
        term = {"term": "Meridian Trust", "term_tokens": ["meridian", "trust"]}
        self.assertEqual(build_query(self.index, QUESTION, term, "q1"), f"{QUESTION}\nMeridian Trust")
        compressed = build_query(self.index, QUESTION, term, "q2").split()
        self.assertIn("meridian", compressed)
        self.assertIn("trust", compressed)
        self.assertTrue({"not", "before", "november", "3", "2023"}.issubset(compressed))
        self.assertEqual(build_query(self.index, QUESTION, term, "q3"), "Meridian Trust")
        hits = self.index.search(f"{QUESTION}\nMeridian Trust", ("meridian", "trust"), 7)
        self.assertEqual({hit.document_id for hit in hits}, {"doc-001", "doc-006", "doc-009"})

    def test_second_hop_has_real_prefix_path_and_attenuation(self):
        config = ExpansionConfig(first_hop_queries=1, max_queries=3, second_hop_min_score=0.2)
        twice = expand(self.index, "one", QUESTION, self.baseline, self.jina, config=config)
        once = expand(self.index, "one", QUESTION, self.baseline, self.jina,
                      config=replace(config, max_hops=1))
        before = through_hop(twice, 1)
        self.assertEqual(stable(once), stable(before))
        first = {item["document_id"]: item for item in before["new_candidates"]}
        self.assertIn("doc-006", first)
        second = next(row for row in twice["new_candidates"] if row["document_id"] == "doc-007")
        self.assertEqual(second["path"], ["doc-001", "doc-006", "doc-007"])
        self.assertEqual(second["hop"], 2)
        self.assertLess(second["score"], first["doc-006"]["score"])
        self.assertLessEqual(second["score"], config.path_decay * first["doc-006"]["score"])
        self.assertEqual(second["seed_document_id"], "doc-006")
        self.assertEqual(self.index.chunk_map[second["chunk_id"]].document_id, "doc-007")
        self.assertEqual(twice["max_hop_reached"], 2)

    def test_safeguards_deduplicate_parents_and_terms_and_stop_at_all_caps(self):
        config = ExpansionConfig(first_hop_queries=1, max_queries=2, max_new_documents=3)
        result = expand(self.index, "one", QUESTION, self.baseline, self.jina, config=config)
        queries, candidates = result["queries"], result["candidate_document_ids"]
        self.assertLessEqual(len(queries), 2)
        self.assertLessEqual(len(candidates), len(self.baseline) + 3)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(candidates[:len(self.baseline)], self.baseline)
        terms = [tuple(query["term"]["term_tokens"]) for query in queries]
        self.assertEqual(len(set(terms)), len(terms))
        self.assertTrue(all(1 <= len(query["hits"]) <= config.hits_per_query for query in queries))
        for item in result["new_candidates"]:
            self.assertEqual(len(set(item["path"])), len(item["path"]))
        with self.assertRaises(ValueError):
            ExpansionConfig(max_hops=3)
        with self.assertRaises(ValueError):
            ExpansionConfig(max_queries=6)

    def test_disconnected_question_preserves_complete_baseline(self):
        result = expand(self.index, "q", "Zaphod explained a kraken's journey.", self.baseline, [])
        self.assertEqual(result["candidate_document_ids"], self.baseline)
        self.assertEqual(result["queries"], [])
        self.assertEqual(result["max_hop_reached"], 0)
        self.assertEqual(result["fallback_reason"], "no_valid_expansion_queries")

    def test_question_only_and_seed_expansion_are_gold_free_and_reproducible(self):
        class Unlabelled:
            query_id = "unlabelled"
            question = QUESTION

            def __getattr__(self, attribute):
                raise AssertionError(f"Unexpected question field lookup {attribute}")

        runtime = Unlabelled()
        config = ExpansionConfig(first_hop_queries=1, max_queries=3)
        for question_only in (False, True):
            a = expand(self.index, runtime.query_id, runtime.question, self.baseline, self.jina,
                       config=config, question_only=question_only)
            b = expand(self.index, runtime.query_id, runtime.question, self.baseline, self.jina,
                       config=config, question_only=question_only)
            self.assertEqual(json.dumps(stable(a), sort_keys=True), json.dumps(stable(b), sort_keys=True))
            self.assertEqual(a["scoring_model_pairs"], 0)
            self.assertEqual(a["embedding_requests"], 0)
            if question_only:
                self.assertEqual(a["seeds"], [])
                self.assertTrue(all(q["term"]["seed_document_id"] is None and q["hop"] == 1
                                    for q in a["queries"]))


if __name__ == "__main__":
    unittest.main()

