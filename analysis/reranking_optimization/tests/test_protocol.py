from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.reranking_optimization import runner


class ProtocolTests(unittest.TestCase):
    def test_pilot_is_deterministic_and_excludes_cached_and_heldout(self):
        splits = {
            "answerable_dev": [f"d{i}" for i in range(60)],
            "dev_100": [f"d{i}" for i in range(30)],
            "answerable_heldout": [f"h{i}" for i in range(100)],
        }
        ids = runner.pilot_ids(splits)
        self.assertEqual(len(ids), 20)
        self.assertEqual(ids, runner.pilot_ids(splits))
        self.assertTrue(set(ids) <= set(splits["answerable_dev"]) - set(splits["dev_100"]))

    def test_frozen_config_validation_and_resume_lock(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            (root / "splits.json").write_text("split")
            (root / "candidate_pool_manifest.json").write_text("pool")
            frozen_path = root / "frozen.json"
            model = {key: f"value-{key}" for key in runner.IDENTITY_FIELDS}
            frozen = {
                "state": "frozen", "policy": "P1", "weight_rrf": 2, "weight_jina": 1, "rrf_k": 60,
                "template": runner.TEMPLATE, "model": model,
                "split_sha256": runner.sha(root / "splits.json"),
                "candidate_manifest_sha256": runner.sha(root / "candidate_pool_manifest.json"),
            }
            with patch.multiple(runner, OUTPUT=root, PARENT=root, FROZEN=frozen_path):
                runner.write(frozen_path, frozen)
                self.assertEqual(
                    runner._stage_spec("heldout", {"answerable_heldout": ["h"],"null_dev":[],"null_heldout":[]}, model),
                    ("P1", ["h"]),
                )
                runner.lock_heldout()
                runner.lock_heldout()
                runner.write(frozen_path, {**frozen, "weight_rrf": 3})
                with self.assertRaisesRegex(ValueError, "changing"):
                    runner.lock_heldout()
                runner.write(frozen_path, {**frozen, "weight_rrf": 99})
                with self.assertRaisesRegex(ValueError, "does not match"):
                    runner._stage_spec("heldout", {}, model)

    def test_expired_deadline_never_calls_backend(self):
        class NoInference:
            def score(self, query, texts):
                raise AssertionError("inference must not start")
        with patch.object(runner, "remaining_seconds", return_value=0):
            with self.assertRaises(TimeoutError):
                runner.BudgetBackend(NoInference()).score("question", ["chunk"])


if __name__ == "__main__":
    unittest.main()
