from dataclasses import dataclass
import unittest

from analysis.besa_v0.selector import select


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str


@dataclass
class Doc:
    document_id: str
    chunks: tuple
    article_source: str


def doc(doc_id, text="a witness discussed the merger in this week's filings", source="News Desk"):
    return Doc(doc_id, (Chunk(f"{doc_id}::0", doc_id, text),), source)


class GuardedPool:
    def __init__(self, *candidates):
        self.candidates = candidates

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected access to {name}")


def cell(doc_id, strength):
    return {"support": strength, "chunk_id": f"{doc_id}::0"}


class SelectorTests(unittest.TestCase):
    def test_marginal_shrinks_after_same_need_is_covered(self):
        pool = GuardedPool(doc("a"), doc("b"), doc("c"))
        supports = {"n1": {"a": cell("a", 0.8), "b": cell("b", 0.7)}, "n2": {}}
        outcome = select(pool, ["a", "b", "c"], supports=supports)
        self.assertEqual(outcome["document_ids"], ["a", "b", "c"])
        self.assertAlmostEqual(outcome["choices"][0]["marginal_coverage"], 0.4)
        self.assertEqual(outcome["choices"][1]["need_gains"], {"n1": 0.0})

    def test_complement_beats_lower_prior_duplicate_without_gold(self):
        pool = GuardedPool(doc("a"), doc("b"), doc("c"))
        supports = {"n1": {"a": cell("a", 0.8), "b": cell("b", 0.7)}, "n2": {"c": cell("c", 0.9)}}
        outcome = select(pool, ["a", "b", "c"], supports=supports, limit=2)
        self.assertEqual(outcome["document_ids"], ["a", "c"])
        self.assertEqual(outcome["full_document_ids"], ["a", "c", "b"])
        self.assertEqual(outcome["lambda_bridge"], 0)

    def test_same_need_penalty_requires_similar_text_and_same_source(self):
        base = "a witness discussed the merger in this week's filings in federal court"
        pool = GuardedPool(
            doc("d", "the directors voted against the merger when market funding unexpectedly collapsed"),
            doc("a", base), doc("b", base), doc("c", base, "Independent News"),
        )
        supports = {"n1": {doc_id: cell(doc_id, 0.8) for doc_id in "abcd"}}
        vectors = {f"{doc_id}::0": [1, 0] for doc_id in "abcd"}
        outcome = select(pool, list("abcd"), mode="D", supports=supports,
                         chunk_vectors=vectors, lambda_repeat=0.20)
        self.assertEqual(outcome["document_ids"], list("acdb"))
        per_doc = {row["document_id"]: row for row in outcome["choices"]}
        self.assertEqual(per_doc["d"]["redundancy"], 0)
        self.assertEqual(per_doc["c"]["redundancy"], 0)
        self.assertEqual(per_doc["b"]["redundancy"], 1)
        self.assertEqual(per_doc["b"]["duplicate_evidence"]["need_id"], "n1")

    def test_same_event_new_constraint_or_need_protects_a_parent(self):
        base = "the same news report discusses the market transaction, the same publishers, and the closing schedule with buyers"
        pool = GuardedPool(doc("a", base + " 2024"), doc("b", base + " 2025"), doc("c", base + " 2024"))
        supports = {"n1": {"a": cell("a", 1), "b": cell("b", 0.7), "c": cell("c", 0.7)},
                    "n2": {"c": cell("c", 0.25)}}
        vectors = {f"{doc_id}::0": [1, 0] for doc_id in "abc"}
        outcome = select(pool, list("abc"), mode="D", supports=supports, chunk_vectors=vectors)
        self.assertTrue(all(not row["redundancy"] for row in outcome["choices"]))
        self.assertEqual(outcome["choices"][0]["document_id"], "a")

    def test_shared_chunk_text_with_different_negation_is_not_penalized(self):
        shared = "the same news report discusses the market transaction the same publishers news about the closing schedule with buyers this week"
        pool = GuardedPool(doc("a", shared + " they will accept"), doc("b", shared + " they will not accept"))
        supports = {"n1": {"a": cell("a", 1), "b": cell("b", 0.7)}}
        outcome = select(pool, list("ab"), mode="D", supports=supports,
                         chunk_vectors={"a::0": [1, 0], "b::0": [1, 0]})
        self.assertEqual(outcome["choices"][1]["redundancy"], 0)

    def test_mmr_stable_tie_and_unique_ten(self):
        ids = [f"d{i:02d}" for i in range(12)]
        vectors = {doc_id: [1.0, 0.0] for doc_id in ids}
        first = select(GuardedPool(*(doc(x) for x in ids)), ids, mode="B",
                       parent_vectors=vectors, mmr_lambda=0.0)
        second = select(GuardedPool(*(doc(x) for x in reversed(ids))), ids, mode="B",
                        parent_vectors=vectors, mmr_lambda=0.0)
        self.assertEqual(first, second)
        self.assertEqual(first["document_ids"], ids[:10])
        self.assertEqual(len(set(first["full_document_ids"])), 12)

    def test_fallback_retains_full_history_and_unscored_is_neutral(self):
        ids = [f"d{i:02d}" for i in range(12)]
        pool = GuardedPool(*(doc(x) for x in ids))
        fallback = select(pool, ids, mode="D", fallback_reason="scoring_error")
        self.assertEqual(fallback["full_document_ids"], ids)
        self.assertEqual(fallback["document_ids"], ids[:10])
        self.assertEqual(fallback["fallback_reason"], "scoring_error")
        sparse = select(pool, ids, supports={"n1": {ids[0]: cell(ids[0], 0.7)}})
        self.assertEqual(sparse["document_ids"], ids[:10])
        self.assertEqual(sparse["choices"][1]["supports"], {})
        self.assertEqual(sparse["choices"][1]["marginal_coverage"], 0)
        invalid = select(pool, ids, supports={"n1": {ids[0]: cell(ids[1], 0.7)}})
        self.assertEqual(invalid["fallback_reason"], "invalid_or_missing_need_support")
        self.assertEqual(invalid["full_document_ids"], ids)

    def test_anchor_retains_first_two_even_if_other_parents_have_higher_gain(self):
        pool = GuardedPool(doc("a"), doc("b"), doc("c"))
        supports = {"n1": {"c": cell("c", 1.0)}}
        anchored = select(pool, list("abc"), mode="E", supports=supports, anchor_count=2)
        self.assertEqual(anchored["document_ids"], list("abc"))
        self.assertEqual([item["policy"] for item in anchored["choices"][:2]], ["anchor", "anchor"])


if __name__ == "__main__":
    unittest.main()
