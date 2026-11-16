from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from analysis.reranking.cache import RerankCache, Score, digest, score_pool
from analysis.reranking.core import Pair
from analysis.reranking_optimization.scoring import LayeredCache, PairCounts


METADATA = {
    "model_id": "model", "revision": "fixed", "model_sha256": "digest",
    "effective_max_tokens": 256, "context_mode": "unified", "truncation_strategy": "longest_first",
}
QUERY = SimpleNamespace(query_id="q1", question="An original question")
PAIRS = [Pair("d1", "c1", "Text A"), Pair("d2", "c2", "Text B"), Pair("d3", "c3", "Text C")]
VALUE = Score(0.5, 10, 10, False, 0.01)


class Backend:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def score(self, query, texts):
        self.calls.extend(texts)
        if self.fail_on in texts:
            raise RuntimeError("interrupted")
        return [VALUE for _ in texts]


class LayeredCacheTests(unittest.TestCase):
    def test_immutable_history_and_exact_identity(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            history, local = Path(temp) / "previous.sqlite3", Path(temp) / "new.sqlite3"
            key = digest(QUERY.question)
            row = [(digest(PAIRS[0].text), PAIRS[0])]
            with RerankCache(history, METADATA) as cache:
                cache.put_batch(QUERY.query_id, key, row, [VALUE])
            old_sha = sha256(history.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                LayeredCache(history, history, METADATA)
            with LayeredCache(local, history, METADATA) as cache:
                self.assertEqual(cache.inspect(QUERY, PAIRS), PairCounts(3, 3, 1, 0, 0, 2))
                self.assertEqual(cache.inspect(QUERY, [PAIRS[0], PAIRS[0]]), PairCounts(2, 1, 1, 0, 1, 0))
                cache.put_batch(QUERY.query_id, key, row, [Score(7.0, 10, 10, False, 0.01)])
                backend = Backend()
                results, actual, _ = score_pool(backend, cache, QUERY, PAIRS, 2)
                self.assertEqual(results, [VALUE] * 3)
                self.assertEqual(actual, 2)
                self.assertEqual(backend.calls, ["Text B", "Text C"])
                self.assertEqual(cache.inspect(QUERY, PAIRS), PairCounts(3, 3, 1, 2, 0, 0))
            with LayeredCache(None, history, dict(METADATA, effective_max_tokens=512)) as cache:
                self.assertEqual(cache.inspect(QUERY, PAIRS), PairCounts(3, 3, 0, 0, 0, 3))
            self.assertEqual(sha256(history.read_bytes()).hexdigest(), old_sha)

    def test_resumes_committed_pairs_and_completed_queries(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            history, local = Path(temp) / "previous.sqlite3", Path(temp) / "new.sqlite3"
            with RerankCache(history, METADATA):
                pass
            with LayeredCache(local, history, METADATA) as cache:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    score_pool(Backend("Text B"), cache, QUERY, PAIRS, 1)
                self.assertIsNone(cache.checkpoint_get("frozen-config/dev", QUERY.query_id))
            with LayeredCache(local, history, METADATA) as cache:
                backend = Backend()
                self.assertEqual(score_pool(backend, cache, QUERY, PAIRS, 1)[1], 2)
                self.assertEqual(backend.calls, ["Text B", "Text C"])
                checkpoint = {"actual_pairs": 2, "rankings": ["d1", "d3", "d2"]}
                cache.checkpoint_put("frozen-config/dev", QUERY.query_id, checkpoint)
                cache.checkpoint_put("frozen-config/dev", QUERY.query_id, checkpoint)
                with self.assertRaises(ValueError):
                    cache.checkpoint_put("frozen-config/dev", QUERY.query_id, {"actual_pairs": 0})
            with LayeredCache(local, history, METADATA) as cache:
                self.assertEqual(cache.checkpoint_get("frozen-config/dev", QUERY.query_id), checkpoint)
                self.assertEqual(list(cache.checkpoint_items("frozen-config/dev")), [("q1", checkpoint)])
                self.assertIsNone(cache.checkpoint_get("other-config/dev", QUERY.query_id))
                self.assertEqual(score_pool(Backend("Text A"), cache, QUERY, PAIRS, 1)[1], 0)


if __name__ == "__main__":
    unittest.main()
