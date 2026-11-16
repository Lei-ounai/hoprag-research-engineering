from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np

from analysis.besa_v0.scoring import (
    BUDGETS, ROUTES, DenseVectors, NeedScoreCache, schedule, score_schedule, summarize_scores,
)
from analysis.reranking.cache import RerankCache, Score, digest


METADATA = {
    "model_id": "model", "revision": "fixed", "model_sha256": "digest",
    "effective_max_tokens": 256, "context_mode": "unified", "truncation_strategy": "longest_first",
}


class GuardedPool:
    def __init__(self, candidates):
        self.query_id = "q-testing"
        self.question = "Where was the first Mars probe launched?"
        self.candidates = candidates

    def __getattr__(self, name):
        raise AssertionError(f"Ranker accessed unexpected input: {name}")


class FakeBackend:
    def __init__(self):
        self.calls = []

    def score(self, need_text, texts):
        self.calls.append((need_text, list(texts)))
        return [Score(float(len(text) % 7) - 2.0, 18, 20, True, 0.03) for text in texts]


def fixture(parent_count=18):
    parents = {}
    vectors = {}
    candidates = []
    for number in range(parent_count):
        doc_id = f"p{number:02}"
        title = f"Astronomy report {number}"
        rows = tuple(
            NS(
                document_id=doc_id, chunk_id=f"{doc_id}::{chunk_number}",
                text=f"{title}\nfirst dense" if chunk_number == 0 else f"lexical Mars probe {chunk_number} launch {number}",
            )
            for chunk_number in range(4)
        )
        parents[doc_id] = rows
        candidates.append(NS(document_id=doc_id, title=title, chunks=rows))
        for index, row in enumerate(rows):
            vectors[digest(row.text)] = np.array([4.0 if index == 0 else 1.0, 4.0 if index else 1.0])
    pool = GuardedPool(candidates)
    vectors[digest(pool.question)] = np.array([1.0, 0.0])
    return pool, DenseVectors(vectors, parents), {doc.document_id: i for i, doc in enumerate(candidates, 1)}


class BudgetedScoringTests(unittest.TestCase):
    def test_strictly_nested_budgets_and_fair_multi_route_screening(self):
        pool, vectors, prior = fixture()
        needs = [NS(need_id=f"n{i}", need_text=f"Mars probe condition {i}") for i in range(3)]
        by_budget = {budget: schedule(pool, needs, prior, vectors, budget) for budget in BUDGETS}
        for budget, pairs in by_budget.items():
            self.assertEqual(pairs, by_budget[90][:budget])
            self.assertEqual(len({(p.need_id, p.chunk_id) for p in pairs}), budget)
            self.assertEqual({p.route for p in pairs}, set(ROUTES))
            per_need = {n.need_id: [p for p in pairs if p.need_id == n.need_id] for n in needs}
            for need_pairs in per_need.values():
                self.assertEqual(len(need_pairs), budget // 3)
                first_visit = need_pairs[:min(len(need_pairs), len(prior))]
                self.assertEqual(len({p.document_id for p in first_visit}), len(first_visit))
            self.assertTrue(any(not p.chunk_id.endswith("::0") for p in pairs))
        one = by_budget[30][0]
        self.assertTrue(one.text.startswith(f"Astronomy report {int(one.document_id[1:])}\n"))
        self.assertFalse(one.chunk_id.endswith("::0"))
        self.assertEqual(set(vectors.parent_vectors(pool)), set(prior))
        with self.assertRaises(ValueError):
            schedule(pool, needs, prior, vectors, 91)

    def test_actual_model_calls_share_maximum_prefix_and_warm_reproduce_scores(self):
        pool, vectors, prior = fixture()
        needs = [NS(need_id=f"n{i}", need_text=f"Mars probe condition {i}") for i in range(3)]
        pairs = schedule(pool, needs, prior, vectors)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            sqlite = Path(tmp) / "local.sqlite3"
            cold_backend, warm_backend = FakeBackend(), FakeBackend()
            with NeedScoreCache(sqlite, {}, METADATA) as cold:
                cold_rows = score_schedule(cold_backend, cold, pool.query_id, pairs)
            with NeedScoreCache(sqlite, {}, METADATA) as warm:
                warm_rows = score_schedule(warm_backend, warm, pool.query_id, pairs)
            self.assertEqual(sum(len(texts) for _, texts in cold_backend.calls), 90)
            self.assertTrue(all(len(texts) <= 16 for _, texts in cold_backend.calls))
            self.assertEqual(warm_backend.calls, [])
            self.assertEqual([p.score for p in cold_rows], [p.score for p in warm_rows])
            for budget in BUDGETS:
                old = summarize_scores(cold_rows, budget)
                new = summarize_scores(warm_rows, budget)
                self.assertEqual(old["supports"], new["supports"])
                self.assertEqual(old["new_pairs"], budget)
                self.assertEqual(new["local_cache_hits"], budget)

    def test_historical_exact_pair_is_reused_read_only(self):
        pool, vectors, prior = fixture()
        needs = [NS(need_id="n1", need_text="Probe landing location")]
        pairs = schedule(pool, needs, prior, vectors)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            history = Path(tmp) / "history.sqlite3"
            local = Path(tmp) / "new.sqlite3"
            with RerankCache(history, METADATA) as historical:
                historical.put_batch(
                    pool.query_id, digest(needs[0].need_text),
                    [(digest(pairs[0].text), pairs[0])], [Score(2, 16, 16, False, 0.02)],
                )
            before = sha256(history.read_bytes()).hexdigest()
            backend = FakeBackend()
            with NeedScoreCache(local, {"audit": history}, METADATA) as cache:
                rows = score_schedule(backend, cache, pool.query_id, pairs[:30])
            self.assertEqual((rows[0].score, rows[0].cache_source), (2, "audit"))
            self.assertEqual(sum(len(texts) for _, texts in backend.calls), 29)
            self.assertEqual(sha256(history.read_bytes()).hexdigest(), before)
            with self.assertRaises(ValueError):
                NeedScoreCache(history, {"audit": history}, METADATA)

    def test_absolute_support_gate_and_unscored_docs_are_distinct(self):
        pool, vectors, prior = fixture()
        needs = [NS(need_id="n1", need_text="Probe landing location")]
        pairs = schedule(pool, needs, prior, vectors)[:3]
        with NeedScoreCache(None, {}, METADATA) as cache:
            rows = score_schedule(FakeBackend(), cache, pool.query_id, pairs)
        rows = tuple(replace(row, score=score) for row, score in zip(rows, (-0.001, -2, -3), strict=True))
        matrix = summarize_scores(rows, 30)["supports"]["n1"]
        self.assertEqual({value["support"] for value in matrix.values()}, {0.0})
        self.assertEqual(set(matrix), {row.document_id for row in rows})
        self.assertLess(len(matrix), len(pool.candidates))
        near = replace(rows[0], score=0.001)
        strong = replace(rows[1], score=3.0)
        actual = summarize_scores((near, strong, rows[2]), 30)["supports"]["n1"]
        self.assertLess(actual[near.document_id]["support"], 0.001)
        self.assertGreater(actual[strong.document_id]["support"], 0.9)

    def test_small_pool_exhausts_without_exceeding_available_slots(self):
        pool, vectors, prior = fixture(2)
        needs = [NS(need_id=f"n{i}", need_text=f"condition {i}") for i in range(4)]
        self.assertEqual(len(schedule(pool, needs, prior, vectors, 90)), 32)
        self.assertEqual(len(schedule(pool, needs, prior, vectors, 30)), 30)
