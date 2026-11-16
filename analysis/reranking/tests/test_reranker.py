from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

from analysis.reranking.cache import RerankCache, Score, score_pool
from analysis.reranking.core import FINAL_K, aggregate, fuse, make_pairs


def candidate(number, *chunks):
    doc_id = f"d{number:02}"
    return NS(
        document_id=doc_id, title=f"Title {number}", rrf_rank=number,
        chunks=[NS(document_id=doc_id, chunk_id=f"{doc_id}:{i}", text=text) for i, text in enumerate(chunks)],
    )


def pool(*candidates):
    return NS(query_id="q", question="The original question", candidates=candidates)


def metadata(mode="unified", revision="revision-1"):
    return {
        "model_id": "model", "revision": revision, "model_sha256": "digest",
        "effective_max_tokens": 256 if mode == "unified" else 8192,
        "context_mode": mode, "truncation_strategy": "longest_first",
    }


class RerankerTests(unittest.TestCase):
    def test_all_chunks_are_retained_with_one_consistent_title(self):
        item = pool(candidate(1, "Title 1\noriginal first", "original last"))
        pairs = make_pairs(item)
        self.assertEqual([p.text for p in pairs], ["Title 1\noriginal first", "Title 1\noriginal last"])
        self.assertEqual([p.chunk_id for p in pairs], ["d01:0", "d01:1"])

    def test_parent_max_raw_score_and_top_ten(self):
        item = pool(candidate(1, "weak", "strong"), *(candidate(n, "passage") for n in range(2, 13)))
        pairs = make_pairs(item)
        ranked = aggregate(item, pairs, [-15, 2, *range(-10, 1)])
        self.assertEqual(ranked[0], {"document_id": "d01", "score": 2, "rank": 1, "winning_chunk_id": "d01:1"})
        final = ranked[:FINAL_K]
        self.assertEqual(len({row["document_id"] for row in final}), 10)
        self.assertEqual([row["rank"] for row in final], list(range(1, 11)))

    def test_rank_fusion_uses_full_fixed_pool_and_rank_only(self):
        item = pool(candidate(1, "one"), candidate(2, "two"), candidate(3, "three"))
        ce = aggregate(item, make_pairs(item), [100, 1000, 3])
        fused = fuse(item, ce)
        self.assertAlmostEqual(fused[0]["score"], 1 / 61 + 1 / 62)
        self.assertEqual([row["document_id"] for row in fused], ["d01", "d02", "d03"])
        with self.assertRaises(ValueError):
            fuse(item, ce[:2])

    def test_gold_cannot_be_read_by_pair_construction_or_scoring(self):
        class GuardedPool:
            query_id = "q"
            question = "Original question only"
            candidates = (candidate(1, "Content only"),)
            def __getattr__(self, name):
                raise AssertionError(f"Attempted to read {name}")

        class GuardedBackend:
            def score(self, question, texts):
                if question != "Original question only" or texts != ["Title 1\nContent only"]:
                    raise AssertionError("Unexpected input")
                return [Score(0.3, 4, 4, False, 0.01)]

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            with RerankCache(Path(temp) / "cache.db", metadata()) as cache:
                scores, actual, _ = score_pool(GuardedBackend(), cache, GuardedPool(), make_pairs(GuardedPool()), 2)
                self.assertEqual((scores[0].score, actual), (0.3, 1))

    def test_cache_resumes_and_separates_context_revision_and_text(self):
        class CountingBackend:
            def __init__(self):
                self.inputs = []
            def score(self, question, texts):
                self.inputs.extend(texts)
                return [Score(0.2, 256, 420, True, 0.03) for _ in texts]

        backend = CountingBackend()
        item = pool(candidate(1, "chunk"), candidate(2, "other"))
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "cache.db"
            with RerankCache(path, metadata()) as cache:
                rows, actual, _ = score_pool(backend, cache, item, make_pairs(item), 1)
                self.assertEqual(actual, 2)
                self.assertEqual([row.truncated for row in rows], [True, True])
            with RerankCache(path, metadata()) as cache:
                warm, actual, model_seconds = score_pool(backend, cache, item, make_pairs(item), 1)
                self.assertEqual((warm, actual, model_seconds), (rows, 0, 0))
                changed = pool(candidate(1, "chunk"), candidate(2, "changed"))
                self.assertEqual(score_pool(backend, cache, changed, make_pairs(changed), 1)[1], 1)
            for config in (metadata("native"), metadata(revision="revision-2")):
                with RerankCache(path, config) as cache:
                    self.assertEqual(score_pool(backend, cache, item, make_pairs(item), 2)[1], 2)

    def test_cache_rejects_false_truncation_metadata_and_wrong_parent(self):
        item = pool(candidate(1, "chunk"))
        pairs = make_pairs(item)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            with RerankCache(Path(temp) / "cache.db", metadata()) as cache:
                with self.assertRaises(ValueError):
                    cache.put_batch("q", "query-hash", [("text-hash", pairs[0])], [Score(0.2, 256, 400, False, 0.01)])
                self.assertEqual(cache.get("query-hash"), {})
        item.candidates[0].chunks[0].document_id = "wrong"
        with self.assertRaises(ValueError):
            make_pairs(item)


if __name__ == "__main__":
    unittest.main()
