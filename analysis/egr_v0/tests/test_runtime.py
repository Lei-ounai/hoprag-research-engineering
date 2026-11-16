"""Integration checks for Gold-free competitions, fixed controls and full fallback."""

import unittest

from analysis.egr_v0.runtime import competitions, control, gate_config
from analysis.egr_v0.selector import choose


class GoldGuard(dict):
    def __getitem__(self, key):
        if key in {"answer", "question_type", "gold", "evidence_list", "is_answerable"}:
            raise AssertionError(f"Access to isolated field: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"answer", "question_type", "gold", "evidence_list", "is_answerable"}:
            raise AssertionError(f"Access to isolated field: {key}")
        return super().get(key, default)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.base = [f"doc-{i:02}" for i in range(1, 16)]
        self.novel = "doc-16"
        total = [*self.base, self.novel]
        self.record = GoldGuard(baseline=self.base, fixed=total, candidate_ids=total,
                               candidates={doc: {"rrf_rank": i} for i, doc in enumerate(total, 1)},
                               scoring={"documents": {doc: [{"jina_score": float(-i)}]
                                                      for i, doc in enumerate(total, 1)},
                                        "fallback_reason": None})
        self.record["scoring"]["documents"][self.novel] = [{"jina_score": 1.0}]
        self.witnesses = {self.novel: [GoldGuard(document_id=self.novel, chunk_id="doc-16::0001",
            status="supported", quality=0.99, jina_score=1.0, coverage_keys=("explicit relationship",),
            evidence_excerpt="An original statement addresses the requested relationship.",
            within_model_visible_text=True)]}

    def test_cohort_labels_never_change_the_runtime_ranking(self):
        config = gate_config(1, 0.75, 0.7, 0.8)
        results = []
        for label in ("null", "answerable"):
            record = GoldGuard(self.record, question_type=label, answer="offline only")
            outcome = choose(competitions(record, self.witnesses)["fixed"], config)
            self.assertIn(self.novel, outcome.document_ids)
            self.assertEqual(self.base[:7], outcome.document_ids[:7])
            results.append(outcome)
        self.assertEqual(*results)

    def test_control_protection_and_scoring_failure_full_p5b(self):
        self.assertEqual(control(self.record, "E0"), self.base[:10])
        self.assertEqual(control(self.record, "E1")[:7], self.base[:7])
        self.assertEqual(control(self.record, "E2")[:7], self.base[:7])
        self.record["scoring"]["fallback_reason"] = "jina_backend_unavailable"
        self.assertEqual(control(self.record, "E2"), self.base[:10])
        outcome = choose(competitions(self.record, {})["augmented"])
        self.assertEqual(outcome.ranking, self.base)
        self.assertEqual(outcome.fallback_reason, "jina_backend_unavailable")


if __name__ == "__main__":
    unittest.main()

