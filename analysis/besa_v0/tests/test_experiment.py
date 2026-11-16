from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sqlite3
import unittest
from unittest.mock import patch

from analysis.besa_v0.experiment import (
    OUT, Checkpoints, assemble_scoring, development_inputs, fallback_record, guarded_selection,
)
from analysis.besa_v0.planner import Need, Plan
from analysis.besa_v0 import run
from analysis.besa_v0.scoring import NeedPair


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.baseline = [f"doc-{i:02d}" for i in range(15)]
        self.pool = SimpleNamespace(query_id="test-query", question="A text question")
        self.plan = Plan(
            question=self.pool.question, accepted=True, needs=(
                Need("n1", "Evidence from article A", "evidence", (), 0.9),
                Need("n2", "Evidence from article B", "evidence", (), 0.9),
            ),
        )

    def test_rejected_plan_never_scores(self):
        def forbidden():
            raise AssertionError("Rejected plan must not score")

        p = Plan(question=self.pool.question, fallback_reason="ambiguous_reference")
        scored, reason = assemble_scoring(self.pool, self.baseline, p, forbidden)
        self.assertEqual(scored, [])
        result = fallback_record(self.baseline, reason)
        self.assertEqual(result["ranking"], self.baseline)
        self.assertEqual(result["document_ids"], self.baseline[:10])

    def test_model_cache_and_invalid_selector_fallback_keep_full_history(self):
        for exc in (RuntimeError("Jina response"), sqlite3.DatabaseError("cache")):
            def bad_score():
                raise exc
            _, reason = assemble_scoring(self.pool, self.baseline, self.plan, bad_score)
            self.assertIsNotNone(reason)
            self.assertEqual(fallback_record(self.baseline, reason)["ranking"], self.baseline)

        def invalid(*args):
            return {"document_ids": ["ghost"] * 10, "fallback_reason": None}

        result = guarded_selection(self.pool, self.baseline, invalid)
        self.assertEqual(result["ranking"], self.baseline)
        self.assertTrue(result["fallback_reason"].startswith("selector_invalid"))

    def test_expired_scoring_does_not_return_a_completed_record(self):
        def unexpected_inference(*args):
            self.fail("Model called after the scoring deadline")

        model = SimpleNamespace(score=unexpected_inference)
        backend = run.TimedBackend(model, datetime(2000, 1, 1, tzinfo=timezone.utc))
        cache = SimpleNamespace(lookup_pairs=lambda *args: {})
        pairs = tuple(
            NeedPair(need.need_id, need.need_text, self.baseline[i], f"chunk-{i}",
                     "text", "need_lexical", 0.0, 1.0)
            for i, need in enumerate(self.plan.needs)
        )
        with (
            patch.object(run, "_checked_plan", return_value=self.plan),
            patch.object(run, "schedule", return_value=pairs),
        ):
            with self.assertRaisesRegex(TimeoutError, "deadline reached"):
                run._scored_record(self.pool, self.baseline, None, backend, cache)

    def test_restart_keeps_completed_question_records(self):
        with TemporaryDirectory(dir=OUT) as directory:
            file = Path(directory) / "checkpoints.sqlite3"
            stage = Checkpoints(file)
            stage.put("scope", {"query_id": "q", "baseline": self.baseline})
            stage.close()

            stage = Checkpoints(file)
            try:
                self.assertEqual(stage.get("scope", "q")["baseline"], self.baseline)
                self.assertIsNone(stage.get("other", "q"))
                self.assertEqual(len(stage.export("scope")), 1)
            finally:
                stage.close()

    def test_development_boundary_excludes_non_development_ids(self):
        with patch("analysis.besa_v0.experiment.development_ids", return_value=("development-id",)):
            with self.assertRaisesRegex(ValueError, "only accepts"):
                list(development_inputs(["unknown-heldout-id"]))


if __name__ == "__main__":
    unittest.main()
