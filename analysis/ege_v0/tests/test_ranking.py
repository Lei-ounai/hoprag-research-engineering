from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from types import SimpleNamespace
import unittest

from analysis.ege_v0.ranking import DocumentScore, MAX_NEW_PAIRS, ScoreBatch, Scorer, fuse
from analysis.reranking.backend import PairScore
from analysis.reranking.cache import RerankCache, digest
from analysis.reranking.core import Pair
from hoprag.retrievers import Chunk


ROOT = Path(__file__).resolve().parents[1]
META = {
    "model_id": "offline-fixture",
    "revision": "test-revision",
    "model_sha256": "1" * 64,
    "context_mode": "unified",
    "effective_max_tokens": 256,
    "truncation_strategy": "longest_first",
}
QUESTION = "Where did the Mars probe land?"
BASE = [f"original-{i:02d}" for i in range(1, 16)]
BASE_WITNESSES = {doc: doc + "::0000" for doc in BASE}


def article(parent, *paragraphs):
    blocks = tuple(Chunk(f"{parent}::{i:04d}", parent, value) for i, value in enumerate(paragraphs))
    return SimpleNamespace(document_id=parent, title=f"Title {parent}", chunks=blocks)


class StubBackend:
    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def score(self, question, texts):
        self.calls.append((question, texts))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("fixture model failure")
        return [PairScore(len(text) / 100.0, 15, 16, True, 0.01) for text in texts]


class RankingTests(unittest.TestCase):
    def test_rrf_stable_parent_deduplication_and_baseline_lane(self):
        first = fuse(BASE, ["new-2", BASE[1], "new-1", "new-2"], expansion_weight=1.0)
        second = fuse(BASE, ["new-2", BASE[1], "new-1", "new-2"], expansion_weight=1.0)
        self.assertEqual(first, second)
        self.assertEqual(first.document_ids[:4], [BASE[0], "new-2", BASE[1], "new-1"])
        self.assertEqual([doc for doc in first.ranking if doc in BASE], BASE)
        self.assertEqual(len(first.document_ids), 10)
        self.assertEqual(len(first.ranking), len(set(first.ranking)))

    def test_anchor_protects_top_absent_and_small_pools(self):
        ranked = fuse(BASE, ["new"], mode="anchor", expansion_weight=10, anchor=8)
        self.assertEqual(ranked.document_ids, BASE[:8] + ["new", BASE[8]])
        self.assertEqual(fuse(BASE[:1], [], mode="anchor").document_ids, BASE[:1])
        self.assertEqual(
            fuse(BASE, ["new"], mode="baseline", baseline_witnesses=BASE_WITNESSES).ranking, BASE,
        )

    def test_jina_reuses_protected_history_and_unknown_is_neutral(self):
        baseline_scores = {doc: float(i) for i, doc in enumerate(BASE)}
        expansion = ("scored-low", "unknown", "scored-high")
        chunk_map = {"scored-low": "path-low", "unknown": "path-unknown", "scored-high": "path-high"}
        batch = ScoreBatch({
            "scored-low": DocumentScore(-1, "jina-low"),
            "scored-high": DocumentScore(100, "jina-high"),
        })
        kwargs = dict(expansion_weight=0.96, anchor=5, baseline_scores=baseline_scores,
                      baseline_witnesses=BASE_WITNESSES, expansion_witnesses=chunk_map)
        result = fuse(BASE, expansion, mode="jina", score_batch=batch, **kwargs)
        self.assertEqual(result.ranking[:5], BASE[:5])
        self.assertEqual([doc for doc in result.ranking if doc in expansion], ["scored-high", "unknown", "scored-low"])
        self.assertEqual(result.witnesses["unknown"], "path-unknown")
        self.assertEqual(result.witnesses["scored-high"], "jina-high")
        self.assertEqual(result.witnesses[BASE[0]], BASE_WITNESSES[BASE[0]])
        lexical = fuse(BASE, expansion, mode="anchor", **kwargs)
        neutral = fuse(BASE, expansion, mode="jina", score_batch=ScoreBatch(), **kwargs)
        zero = fuse(BASE, expansion, mode="jina", score_batch=batch, jina_strength=0, **kwargs)
        self.assertEqual(neutral, lexical)
        self.assertEqual(zero, lexical)

    def test_scoring_or_cache_failure_preserves_complete_p5(self):
        for failure in (None, ScoreBatch(fallback_reason="jina_cache_unavailable:DatabaseError")):
            with self.subTest(failure=failure):
                ranked = fuse(BASE, ["new"], mode="jina", expansion_weight=2, score_batch=failure,
                             baseline_witnesses=BASE_WITNESSES)
                self.assertEqual(ranked.ranking, BASE)
                self.assertEqual(ranked.document_ids, BASE[:10])
                self.assertEqual(ranked.witnesses, BASE_WITNESSES)
                self.assertIsNotNone(ranked.fallback_reason)


class ScorerTests(unittest.TestCase):
    def test_old_cache_reuse_exactly_and_independent_resume(self):
        docs = {
            "old-hit": article("old-hit", "Mars probe"),
            "fresh": article("fresh", "other unconnected prose", "landing site on Mars"),
        }
        witness = {"fresh": "fresh::0001"}
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            old, local = root / "old.sqlite3", root / "ege.sqlite3"
            pair = Pair("old-hit", "old-hit::0000", "Title old-hit\nMars probe")
            with RerankCache(old, META) as store:
                store.put_batch("old-query", digest(QUESTION), [(digest(pair.text), pair)],
                                [PairScore(7, 12, 12, False, 0.1)])
            before = sha256(old.read_bytes()).hexdigest()
            backend = StubBackend()
            with Scorer(local, old, META, backend) as scorer:
                cold = scorer.score("sample", QUESTION, list(docs), docs, expansion_witnesses=witness)
            self.assertIsNone(cold.fallback_reason)
            self.assertEqual((cold.actual_pairs, cold.new_pairs, cold.historical_hits, cold.local_hits), (2, 1, 1, 0))
            self.assertEqual(cold.parents["old-hit"], DocumentScore(7, "old-hit::0000"))
            self.assertEqual(cold.parents["fresh"].chunk_id, "fresh::0001")
            self.assertEqual(backend.calls, [(QUESTION, ["Title fresh\nlanding site on Mars"])])
            self.assertEqual(ScoreBatch.from_dict(asdict(cold)), cold)

            with Scorer(local, old, META, None) as reopened:
                warm = reopened.score("sample", QUESTION, list(docs), docs, expansion_witnesses=witness)
            self.assertIsNone(warm.fallback_reason)
            self.assertEqual(warm.parents, cold.parents)
            self.assertEqual((warm.actual_pairs, warm.new_pairs, warm.historical_hits, warm.local_hits), (2, 0, 1, 1))
            self.assertEqual(warm.inference_seconds, 0.0)
            self.assertEqual(sha256(old.read_bytes()).hexdigest(), before)
            with Scorer(old, old, META, None) as invalid:
                failed = invalid.score("sample", QUESTION, list(docs), docs)
            self.assertEqual(failed.fallback_reason, "jina_cache_unavailable:ValueError")
            self.assertEqual(sha256(old.read_bytes()).hexdigest(), before)

    def test_per_query_budget_and_existing_chunk_mapping(self):
        docs = {str(i): article(str(i), "probe Mars land on the selected block") for i in range(12)}
        backend = StubBackend()
        with TemporaryDirectory(dir=ROOT) as directory:
            with Scorer(Path(directory) / "ege.sqlite3", None, META, backend) as scorer:
                result = scorer.score("sample", QUESTION, list(docs), docs)
        self.assertEqual((result.actual_pairs, result.new_pairs), (MAX_NEW_PAIRS, MAX_NEW_PAIRS))
        self.assertEqual(set(result.parents), set(list(docs)[:MAX_NEW_PAIRS]))
        self.assertEqual({row.chunk_id for row in result.parents.values()},
                         {doc + "::0000" for doc in list(docs)[:MAX_NEW_PAIRS]})
        with self.assertRaises(ValueError):
            Scorer(None, None, META, backend, max_parents=MAX_NEW_PAIRS + 1)

    def test_partial_pair_commit_resumes_without_rescoring(self):
        docs = {"first": article("first", "A passage"), "second": article("second", "A later passage")}
        with TemporaryDirectory(dir=ROOT) as directory:
            local = Path(directory) / "ege.sqlite3"
            with Scorer(local, None, META, StubBackend(fail_on_call=2), batch_size=1) as scorer:
                failed = scorer.score("sample", QUESTION, list(docs), docs)
            self.assertEqual(failed.fallback_reason, "jina_scoring_error:RuntimeError")
            self.assertEqual(failed.parents, {})
            backend = StubBackend()
            with Scorer(local, None, META, backend, batch_size=1) as resumed:
                result = resumed.score("sample", QUESTION, list(docs), docs)
                stopped = resumed.score("sample", QUESTION, list(docs), docs, deadline=monotonic() - 1)
            self.assertEqual((result.local_hits, result.new_pairs, len(backend.calls)), (1, 1, 1))
            self.assertEqual(stopped.fallback_reason, "deadline_exhausted")
            self.assertEqual(stopped.new_pairs, 0)


if __name__ == "__main__":
    unittest.main()

