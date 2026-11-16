import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import numpy as np

from hoprag.config import BenchmarkConfig
from hoprag.embeddings import EmbeddingError, OllamaEmbedder
from hoprag.models import Document, Query, RetrievalResult
from hoprag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever


def document(document_id, title, text):
    return Document(document_id=document_id, title=title, text=text, metadata={})


def query(question):
    return Query(
        query_id="q",
        question=question,
        answer="",
        question_type="inference",
        evidence_list=(),
    )


def response(vectors):
    return io.BytesIO(json.dumps({"embeddings": vectors, "prompt_eval_count": 4}).encode("utf-8"))


class LexicalAndHybridTests(unittest.TestCase):
    def test_bm25_matches_title_and_body_and_deduplicates_ids(self):
        docs = [
            document("a", "Lighthouse", "The port opened in 1912."),
            document("b", "Other news", "A market opened."),
            document("c", "Festival", "An arts gathering."),
            document("d", "Lake", "This is a reservoir."),
        ]
        bm25 = BM25Retriever([*docs, docs[0]])
        hits = bm25.search(query("LIGHTHOUSE 1912"), 4)
        self.assertEqual([(hit.document_id, hit.rank, hit.retrieval_method) for hit in hits],
                         [("a", 1, "bm25")])
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(bm25.search(query("unseen"), 4), [])

    def test_conflicting_duplicate_documents_fail(self):
        docs = [document("a", "First", "An article."), document("a", "Second", "A different article.")]
        with self.assertRaisesRegex(ValueError, "Conflicting records"):
            BM25Retriever(docs)

    def test_rrf_uses_ranks_only_and_deduplicates_within_each_method(self):
        bm25 = [
            RetrievalResult("q", "a", "bm25", 1000.0, 1),
            RetrievalResult("q", "b", "bm25", -1.0, 2),
            RetrievalResult("q", "a", "bm25", 900.0, 3),
        ]
        dense = [
            RetrievalResult("q", "b", "dense", 0.25, 1),
            RetrievalResult("q", "c", "dense", 0.20, 2),
        ]
        hits = HybridRetriever(None, None).fuse(query("port"), bm25, dense, 3)
        self.assertEqual([(hit.document_id, hit.rank) for hit in hits], [("b", 1), ("a", 2), ("c", 3)])
        self.assertAlmostEqual(hits[0].score, 1 / 62 + 1 / 61)
        self.assertAlmostEqual(hits[1].score, 1 / 61)
        self.assertAlmostEqual(hits[2].score, 1 / 62)


    def test_hybrid_search_ranks_beyond_output_limit(self):
        class Branch:
            def __init__(self, method, document_ids):
                self.method = method
                self.document_ids = document_ids
                self.requested_depths = []

            def search(self, query, top_k):
                self.requested_depths.append(top_k)
                return [
                    RetrievalResult(query.query_id, doc_id, self.method, 0.0, rank)
                    for rank, doc_id in enumerate(self.document_ids[:top_k], start=1)
                ]

        bm25 = Branch("bm25", ("a", "b"))
        dense = Branch("dense", ("c", "b"))
        retriever = HybridRetriever(bm25, dense)
        first = retriever.search(query("port"), 1)
        self.assertEqual([hit.document_id for hit in first], ["b"])
        self.assertEqual(bm25.requested_depths, [BenchmarkConfig.fusion_depth])
        self.assertEqual(dense.requested_depths, [BenchmarkConfig.fusion_depth])

        larger_limit = BenchmarkConfig.fusion_depth + 1
        retriever.search(query("port"), larger_limit)
        self.assertEqual(bm25.requested_depths[-1], larger_limit)
        self.assertEqual(dense.requested_depths[-1], larger_limit)


class DenseTests(unittest.TestCase):
    def test_chunking_keeps_all_text_and_dense_uses_best_chunk_cosine(self):
        class FakeEmbedder:
            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(tuple(texts))
                return np.array([
                    [10, 0] if "CAT" in text else [1, 1] if text == "mixed" else [0, 5]
                    for text in texts
                ], dtype=np.float32)

        first = document("a", "猫咪", "🙂 go to the market CAT")
        second = document("b", "Dogs", "A news story.")
        embedder = FakeEmbedder()
        dense = DenseRetriever([second, first, first], embedder, max_chunk_bytes=12)
        chunks_a = [chunk for chunk in dense.chunks if chunk.document_id == "a"]
        self.assertGreater(len(chunks_a), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks_a), f"{first.title}\n{first.text}")
        self.assertEqual([chunk.chunk_id for chunk in chunks_a],
                         [f"a::{index:04d}" for index in range(len(chunks_a))])
        self.assertTrue(all(len(chunk.text.encode("utf-8")) <= 12 for chunk in dense.chunks))

        main_query = query("CAT")
        dense.prepare_queries((main_query, query("mixed"), main_query))
        results = dense.search(main_query, 4)
        self.assertEqual(embedder.calls[1:], [("CAT", "mixed")])
        self.assertEqual([(hit.document_id, hit.rank) for hit in results], [("a", 1), ("b", 2)])
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 0.0)


class EmbeddingCacheTests(unittest.TestCase):
    def test_batching_cache_model_and_input_order(self):
        payloads = []

        def http(request, *, timeout):
            payload = json.loads(request.data)
            payloads.append(payload)
            return response([[float(text), 1.0] for text in payload["input"]])

        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "vectors.sqlite"
            progress = []
            with patch("hoprag.embeddings.urlopen", side_effect=http):
                with OllamaEmbedder(
                    cache_path, dimensions=2, batch_size=2,
                    progress=lambda done, total: progress.append((done, total)),
                ) as embedder:
                    vectors = embedder.embed(["2", "1", "3", "2"])
                    self.assertEqual(embedder.stats.api_requests, 2)
                    self.assertEqual(embedder.stats.prompt_eval_count, 8)
                    self.assertEqual(embedder.stats.cache_misses, 3)
                with OllamaEmbedder(cache_path, dimensions=2) as embedder:
                    cached = embedder.embed(["3", "1", "3"])
                    self.assertEqual(embedder.stats.cache_hits, 2)
                    self.assertEqual(embedder.stats.api_requests, 0)
                with OllamaEmbedder(cache_path, model="other", dimensions=2) as embedder:
                    embedder.embed(["3"])
                    self.assertEqual(embedder.stats.cache_misses, 1)

        self.assertEqual(progress, [(0, 3), (2, 3), (3, 3)])
        self.assertEqual([payload["input"] for payload in payloads], [["2", "1"], ["3"], ["3"]])
        self.assertTrue(all(payload["truncate"] is False for payload in payloads))
        np.testing.assert_allclose(vectors, [[2, 1], [1, 1], [3, 1], [2, 1]])
        np.testing.assert_allclose(cached, [[3, 1], [1, 1], [3, 1]])

    def test_failed_batch_preserves_previous_commits_for_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "vectors.sqlite"
            with OllamaEmbedder(cache_path, dimensions=2, batch_size=1) as embedder:
                with patch("hoprag.embeddings.urlopen",
                           side_effect=[response([[1, 2]]), URLError("offline")]):
                    with self.assertRaisesRegex(EmbeddingError, "missing input 2"):
                        embedder.embed(["first", "second"])
            with OllamaEmbedder(cache_path, dimensions=2) as embedder:
                with patch("hoprag.embeddings.urlopen", return_value=response([[2, 3]])) as http:
                    vectors = embedder.embed(["first", "second"])
                self.assertEqual(json.loads(http.call_args.args[0].data)["input"], ["second"])
                np.testing.assert_allclose(vectors, [[1, 2], [2, 3]])

    def test_rejects_wrong_dimension_and_nonfinite_or_zero_values(self):
        invalid = [
            ([[1.0]], "shape"),
            ([[float("nan"), 0.0]], "NaN"),
            ([[float("inf"), 0.0]], "infinite"),
            ([[0.0, 0.0]], "zero"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            with OllamaEmbedder(Path(folder) / "vectors.sqlite", dimensions=2) as embedder:
                for vectors, error in invalid:
                    with self.subTest(error=error):
                        with patch("hoprag.embeddings.urlopen", return_value=response(vectors)):
                            with self.assertRaisesRegex(EmbeddingError, error):
                                embedder.embed(["example"])
                self.assertEqual(embedder.stats.cache_hits, 0)


@unittest.skipUnless(os.environ.get("HOPRAG_OLLAMA_TEST") == "1", "opt-in local Ollama check")
class OllamaIntegrationTests(unittest.TestCase):
    def test_local_bge_m3_endpoint_returns_two_1024_dimension_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            with OllamaEmbedder(Path(folder) / "vectors.sqlite") as embedder:
                vectors = embedder.embed(["lighthouse", "观测站"])
                self.assertEqual(vectors.shape, (2, 1024))
                self.assertTrue(np.isfinite(vectors).all())
                self.assertEqual(embedder.stats.api_requests, 1)
