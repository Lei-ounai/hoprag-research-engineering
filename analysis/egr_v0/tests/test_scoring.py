from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from types import SimpleNamespace
import unittest

import numpy as np

from analysis.egr_v0.scoring import FairScorer, select_chunks
from analysis.reranking.backend import PairScore
from analysis.reranking.cache import RerankCache, digest
from analysis.reranking.core import Pair
from hoprag.models import Document
from hoprag.retrievers import Chunk


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Where did the Mars lander touch down?"
META = {
    "model_id": "offline-fixture", "revision": "test-revision", "model_sha256": "1" * 64,
    "context_mode": "unified", "effective_max_tokens": 256, "truncation_strategy": "longest_first",
    "truncation_side": "right",
}


def fixture():
    docs = {
        "old": Document("old", "Mars lander", ""),
        "new": Document("new", "Mission updates", ""),
    }
    texts = {
        "old": ["Mars lander\nMars lander touched down", "reflected telescopes in the highlands"],
        "new": ["mission status update", "mission status duplicate update", "Mars lander at Elysium Planitia",
                "Elysium Planitia described a dust cloud"],
    }
    chunks = {doc: tuple(Chunk(f"{doc}::{i:04d}", doc, text) for i, text in enumerate(values))
              for doc, values in texts.items()}
    return docs, chunks


class VectorStub:
    def __init__(self):
        self.chunk_vectors = dict(zip(
            ("old::0000", "old::0001", "new::0000", "new::0001", "new::0002", "new::0003"),
            tuple(np.array(x) for x in ((1, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
            strict=True,
        ))
        self.seen = []

    def cosines(self, question, blocks):
        self.seen.append(question)
        weights = {"old::0000": .8, "old::0001": .25, "new::0000": .92, "new::0001": .91,
                   "new::0002": .63, "new::0003": .5}
        return {row.chunk_id: weights[row.chunk_id] for row in blocks}

    def lexical(self, question, chunk_ids):
        weights = {"old::0000": .8, "old::0001": .25, "new::0000": .1, "new::0001": .19,
                   "new::0002": .75, "new::0003": .1}
        return {chunk_id: weights[chunk_id] for chunk_id in chunk_ids}


class BackendStub:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def score(self, question, texts):
        self.calls.append((question, texts))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("fixture failure")
        return [PairScore(len(text) / 100, 10, 10, False, 0.02) for text in texts]


class FairChunkTests(unittest.TestCase):
    def test_same_cap_multiroute_and_diversity_on_real_original_offsets(self):
        docs, chunks = fixture()
        vec = VectorStub()
        result = select_chunks(QUESTION, docs, chunks, vec, {"new": ["Elysium Planitia"]})
        self.assertEqual({doc: len(rs) for doc, rs in result.items()}, {"old": 2, "new": 3})
        self.assertEqual([r["chunk_id"] for r in result["new"]], ["new::0000", "new::0002", "new::0003"])
        self.assertEqual([r["selection_route"] for r in result["new"]],
                         ["cached_dense", "query_lexical_and_provenance", "diversity"])
        self.assertEqual(result["new"][1]["expansion_query_match"], 1)
        self.assertEqual(result["old"][0]["title_match"], 1)
        self.assertEqual(vec.seen, [QUESTION])
        self.assertEqual(result, select_chunks(QUESTION, docs, chunks, vec, {"new": ["Elysium Planitia"]}))
        original_ids = {row.chunk_id for values in chunks.values() for row in values}
        self.assertTrue({r["chunk_id"] for rs in result.values() for r in rs} <= original_ids)

    def test_per_parent_soft_budget_and_wrong_identity(self):
        docs, chunks = fixture()
        vec = VectorStub()
        self.assertEqual({doc: len(r) for doc, r in select_chunks(QUESTION, docs, chunks, vec, limit=1).items()},
                         {"old": 1, "new": 1})
        with self.assertRaises(ValueError):
            select_chunks(QUESTION, docs, chunks, vec, limit=4)
        displaced = {**chunks, "old": chunks["new"], "new": chunks["old"]}
        with self.assertRaisesRegex(ValueError, "Unresolved parent"):
            select_chunks(QUESTION, docs, displaced, vec)


class ScoreTests(unittest.TestCase):
    def test_exact_history_and_warm_cache_keep_identical_scores(self):
        docs, chunks = fixture()
        vec = VectorStub()
        backend = BackendStub()
        kwargs = dict(vectors=vec, candidates=docs, chunks=chunks)
        with TemporaryDirectory(dir=ROOT) as tmp:
            history = Path(tmp) / "history.sqlite3"
            target = Path(tmp) / "current.sqlite3"
            known = Pair("old", "old::0000", chunks["old"][0].text)
            with RerankCache(history, META) as seed:
                seed.put_batch("legacy", digest(QUESTION), [(digest(known.text), known)],
                               [PairScore(3, 10, 12, True, 0.02)])
            before = sha256(history.read_bytes()).hexdigest()
            with FairScorer(target, {"history": history}, META, backend, **kwargs) as scorer:
                first = scorer.score("q", QUESTION, docs, {"new": "Elysium Planitia"})
            self.assertIsNone(first["fallback_reason"])
            self.assertEqual((first["actual_pairs"], first["historical_hits"], first["new_pairs"]), (5, 1, 4))
            self.assertEqual(first["documents"]["old"][0]["jina_score"], 3)
            self.assertIsNone(first["documents"]["old"][0]["model_visible_chunk_chars"])
            self.assertEqual(first["documents"]["old"][1]["model_visible_chunk_chars"], len(chunks["old"][1].text))
            self.assertIn("Mission updates\nMars lander at Elysium Planitia", backend.calls[0][1])
            self.assertEqual(sha256(history.read_bytes()).hexdigest(), before)
            with FairScorer(target, {"history": history}, META, None, **kwargs) as scorer:
                warm = scorer.score("q", QUESTION, docs, {"new": "Elysium Planitia"})
            self.assertIsNone(warm["fallback_reason"])
            self.assertEqual((warm["new_pairs"], warm["local_hits"], warm["historical_hits"]), (0, 4, 1))
            for doc in docs:
                self.assertEqual([(row["chunk_id"], row["jina_score"]) for row in first["documents"][doc]],
                                 [(row["chunk_id"], row["jina_score"]) for row in warm["documents"][doc]])

    def test_model_range_maps_template_offsets_back_to_exact_original_chunk(self):
        docs, chunks = fixture()

        class TokenizerStub:
            def encode_batch(self, pairs):
                result = []
                for _, text in pairs:
                    body_start = text.index("\n") + 1
                    result.append(SimpleNamespace(
                        attention_mask=[1, 1, 1, 1, 0],
                        sequence_ids=[None, 0, 1, 1, None],
                        offsets=[(0, 0), (0, 5), (0, 3), (body_start, body_start + 2), (0, 0)],
                    ))
                return result

        class BackendWithTokenizer:
            _encoder = SimpleNamespace(model=SimpleNamespace(tokenizer=TokenizerStub()))

            def score(self, question, texts):
                return [PairScore(0.5, 4, 6, True, 0.02) for text in texts]

        with TemporaryDirectory(dir=ROOT) as tmp:
            with FairScorer(Path(tmp) / "current.sqlite3", None, META, BackendWithTokenizer(),
                            vectors=VectorStub(), candidates=docs, chunks=chunks) as scorer:
                result = scorer.score("q", QUESTION, ["old"])
        self.assertIsNone(result["fallback_reason"])
        first, second = result["documents"]["old"]
        self.assertEqual(first["model_visible_chunk_chars"], len("Mars lander\n") + 2)
        self.assertEqual(second["model_visible_chunk_chars"], 2)
        self.assertEqual(first["model_visible_query_chars"], 5)
        self.assertEqual(second["model_visible_query_chars"], 5)

    def test_scoring_failure_and_expired_deadline_keep_partial_cache(self):
        docs, chunks = fixture()
        kwargs = dict(vectors=VectorStub(), candidates=docs, chunks=chunks, batch_size=1)
        with TemporaryDirectory(dir=ROOT) as tmp:
            target = Path(tmp) / "current.sqlite3"
            with FairScorer(target, None, META, BackendStub(fail_at=2), **kwargs) as scorer:
                failed = scorer.score("q", QUESTION, docs)
            self.assertEqual(failed["fallback_reason"], "jina_scoring_error:RuntimeError")
            self.assertEqual(failed["documents"], {})
            with FairScorer(target, None, META, BackendStub(), **kwargs) as scorer:
                expired = scorer.score("q", QUESTION, docs, deadline=monotonic() - 10)
                resumed = scorer.score("q", QUESTION, docs)
            self.assertEqual(expired["fallback_reason"], "deadline_exhausted")
            self.assertEqual(expired["new_pairs"], 0)
            self.assertEqual((resumed["local_hits"], resumed["new_pairs"]), (1, 4))


if __name__ == "__main__":
    unittest.main()

