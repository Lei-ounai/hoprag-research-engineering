from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.reranking import pool
from hoprag.models import Evidence, Query, RetrievalResult


class CandidatePoolTests(unittest.TestCase):
    def setUp(self):
        self.rankings = {
            "bm25": [RetrievalResult("q", "a", "bm25", 12.0, 1), RetrievalResult("q", "b", "bm25", 9.0, 2)],
            "dense": [RetrievalResult("q", "b", "dense", 0.8, 1), RetrievalResult("q", "c", "dense", 0.7, 2)],
            "rrf": [
                RetrievalResult("q", "b", "hybrid", 1 / 61 + 1 / 62, 1),
                RetrievalResult("q", "a", "hybrid", 1 / 61, 2),
                RetrievalResult("q", "c", "hybrid", 1 / 62, 3),
            ],
        }

    def test_candidate_union_deduplicates_parent_and_retains_branch_positions(self):
        record = pool.build_candidate_record("q", "question", self.rankings)
        rows = record["candidates"]
        self.assertEqual(["b", "a", "c"], [row["document_id"] for row in rows])
        self.assertEqual((2, 1, 1, ["bm25", "dense"]), tuple(
            rows[0][key] for key in ("bm25_rank", "dense_rank", "rrf_rank", "source")
        ))
        self.assertEqual((9.0, 0.8), (rows[0]["bm25_score"], rows[0]["dense_score"]))
        self.assertEqual((None, None), (rows[1]["dense_rank"], rows[2]["bm25_rank"]))

    def test_candidate_union_rejects_incomplete_fused_rank(self):
        self.rankings["rrf"] = self.rankings["rrf"][:2]
        with self.assertRaisesRegex(ValueError, "did not preserve"):
            pool.build_candidate_record("q", "question", self.rankings)

    def test_frozen_input_preserves_every_chunk_and_exposes_no_gold(self):
        with tempfile.TemporaryDirectory(dir=pool.OUTPUT) as folder:
            output = Path(folder)
            record = pool.build_candidate_record("q", "actual question", self.rankings)
            self.assertEqual({"query_id", "question", "candidates"}, set(record))
            (output / pool.POOLS_FILE).write_text(pool._line(record), encoding="utf-8")
            with (output / pool.DOCUMENTS_FILE).open("w", encoding="utf-8") as docs:
                for doc_id in ("a", "b", "c"):
                    docs.write(pool._line({
                        "document_id": doc_id, "title": f"title {doc_id}", "text": f"whole {doc_id}",
                        "article_source": "publication", "chunks": [
                            {"chunk_id": f"{doc_id}::0000", "document_id": doc_id, "text": f"lead {doc_id}"},
                            {"chunk_id": f"{doc_id}::0001", "document_id": doc_id, "text": f"tail {doc_id}"},
                        ],
                    }))
            (output / pool.SPLITS_FILE).write_text("{}", encoding="utf-8")
            pool._write_json(output / pool.MANIFEST_FILE, {
                "version": 1,
                "artifacts": {name: pool._fingerprint(output / name) for name in pool.ARTIFACTS},
            })
            with patch.object(pool, "load_dataset", side_effect=AssertionError("scorer accessed annotations")):
                item, = pool.iter_pools(["q"], output=output)
            self.assertEqual("actual question", item.question)
            self.assertFalse(hasattr(item, "answer"))
            self.assertFalse(hasattr(item, "evidence_list"))
            self.assertEqual(["b::0000", "b::0001"], [chunk.chunk_id for chunk in item.candidates[0].chunks])
            self.assertEqual(["lead b", "tail b"], [chunk.text for chunk in item.candidates[0].chunks])
            self.assertEqual("whole b", item.candidates[0].text)
            (output / pool.POOLS_FILE).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                list(pool.iter_pools(["q"], output=output))

    def test_dev_holdout_are_disjoint_and_experiments_are_nested(self):
        queries = [
            Query(f"a-{i:04d}", "question", "gold answer", f"type-{i % 2}",
                  tuple(Evidence(f"e-{j}", "gold", "doc") for j in range(2 + i % 3)))
            for i in range(505)
        ]
        queries += [Query(f"n-{i:04d}", "null", "gold answer", "null_query") for i in range(47)]
        split = pool.build_splits(queries)
        self.assertEqual(split, pool.build_splits(queries))
        self.assertEqual((101, 404, 9, 38), tuple(
            len(split[name]) for name in ("answerable_dev", "answerable_heldout", "null_dev", "null_heldout")
        ))
        for prefix in ("answerable", "null"):
            self.assertFalse(set(split[f"{prefix}_dev"]) & set(split[f"{prefix}_heldout"]))
        self.assertTrue(set(split["dev_5"]) < set(split["dev_20"]) < set(split["dev_100"]) < set(split["answerable_dev"]))
        self.assertEqual([5, 20, 100], [len(split[key]) for key in ("dev_5", "dev_20", "dev_100")])
        self.assertEqual(6, len(split["strata"]["answerable_dev"]))


if __name__ == "__main__":
    unittest.main()
