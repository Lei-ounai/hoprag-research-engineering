from dataclasses import replace
import unittest

from analysis.reranking.pool import Candidate, QueryPool
from analysis.reranking_optimization.policies import (
    DEV_SUPERSET,
    POLICIES,
    POLICY_UNION,
    candidate_record,
    select,
    weighted_fuse,
)
from hoprag.retrievers import Chunk


def candidate(doc_id, rrf, bm25=None, dense=None):
    return Candidate(
        document_id=doc_id,
        rrf_rank=rrf,
        bm25_rank=bm25,
        dense_rank=dense,
        bm25_score=0.5 if bm25 else None,
        dense_score=0.6 if dense else None,
        rrf_score=0.03,
        source=tuple(lane for lane, rank in (("bm25", bm25), ("dense", dense)) if rank),
        title="Original title",
        text="Original body",
        chunks=(Chunk(f"{doc_id}::0", doc_id, "Original title\nOriginal body"),
                Chunk(f"{doc_id}::1", doc_id, "Retained second chunk")),
        article_source="Original source",
    )


class PoolPolicyTests(unittest.TestCase):
    def test_all_six_candidate_sets_and_development_superset(self):
        original = QueryPool("q", "Original question", (
            candidate("a", 20, 10, 11),
            candidate("b", 21, 11),
            candidate("c", 30, dense=15),
            candidate("d", 31, 16),
            candidate("e", 50, 51, 52),
            candidate("f", 51, 25, 26),
            candidate("g", 52, dense=50),
            candidate("h", 53, 51, 51),
        ))
        expected = {
            "P1": ("a",), "P2": ("a", "b", "c"), "P3": ("a", "b", "c", "d", "e"),
            "P4": ("a",), "P5": ("a", "b", "c"), "P6": ("a", "b", "c", "d", "f"),
            DEV_SUPERSET: ("a", "b", "c", "d", "e", "f", "g"),
            POLICY_UNION: ("a", "b", "c", "d", "e", "f"),
        }
        for name in (*POLICIES, DEV_SUPERSET, POLICY_UNION):
            with self.subTest(name=name):
                self.assertEqual(tuple(row.document_id for row in select(original, name).candidates), expected[name])
        with self.assertRaises(ValueError):
            select(original, "UNKNOWN")

    def test_union_deduplicates_parents_and_rejects_conflicts(self):
        both = candidate("a", 1, 1, 2)
        original = QueryPool("q", "Original question", (both, both, candidate("b", 2, dense=1)))
        selected = select(original, "P4")
        self.assertEqual(tuple(row.document_id for row in selected.candidates), ("a", "b"))
        self.assertEqual(candidate_record(selected, "P4")["candidates"][0]["admitted_by"], ["bm25", "dense"])
        with self.assertRaises(ValueError):
            select(QueryPool("q", "Question", (both, replace(both, title="Conflict"))), "P4")

    def test_parent_selection_preserves_original_chunk_objects_and_ranks(self):
        original = candidate("a", 22, 3, 7)
        selected = select(QueryPool("q", "Question", (original,)), "P4")
        self.assertIs(selected.candidates[0], original)
        row = candidate_record(selected, "P4")["candidates"][0]
        self.assertEqual((row["bm25_rank"], row["dense_rank"], row["rrf_rank"]), (3, 7, 22))
        self.assertEqual(row["parent_document_id"], original.document_id)
        self.assertEqual(row["chunk_ids"], [chunk.chunk_id for chunk in original.chunks])
        self.assertTrue(all(chunk.document_id == original.document_id for chunk in original.chunks))

    def test_weighted_fusion_compacts_rrf_positions_and_keeps_unique_top_ten(self):
        pool = QueryPool("q", "Question", tuple(
            candidate(f"d{i:02}", i * 2, i) for i in range(1, 14)
        ))
        ce = [{"document_id": f"d{i:02}", "rank": 14 - i, "score": -1e6} for i in range(1, 14)]
        ranked = weighted_fuse(pool, ce, 4)
        self.assertEqual(ranked[0]["document_id"], "d01")
        self.assertAlmostEqual(ranked[0]["score"], 4 / (60 + 1) + 1 / (60 + 13))
        self.assertEqual(len({row["document_id"] for row in ranked[:10]}), 10)
        self.assertEqual([row["rank"] for row in ranked[:10]], list(range(1, 11)))

    def test_missing_rank_contributes_zero_and_invalid_rank_is_rejected(self):
        pool = QueryPool("q", "Question", (candidate("a", 7, 1), candidate("b", None, dense=2)))
        ranked = weighted_fuse(pool, [{"document_id": "b", "rank": 1}], 2)
        by_id = {row["document_id"]: row["score"] for row in ranked}
        self.assertAlmostEqual(by_id["a"], 2 / 61)
        self.assertAlmostEqual(by_id["b"], 1 / 61)
        for invalid in (
            [{"document_id": "b", "rank": 0}],
            [{"document_id": "c", "rank": 1}],
            [{"document_id": "b", "rank": 1}, {"document_id": "b", "rank": 2}],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                weighted_fuse(pool, invalid, 2)


if __name__ == "__main__":
    unittest.main()
