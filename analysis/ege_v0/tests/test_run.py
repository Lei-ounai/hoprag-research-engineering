from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.besa_v0.experiment import rows
from analysis.ege_v0 import run
from analysis.reranking.pool import ARTIFACTS, MANIFEST_FILE


REPLAY_INPUTS = (run.SPLIT, run.DEV_RANKS, run.NULL_RANKS, *(run.R5 / name for name in (*ARTIFACTS, MANIFEST_FILE)))


class ExpansionRuntimeTests(unittest.TestCase):
    def test_checkpoint_resume_discards_only_incomplete_line(self):
        with TemporaryDirectory(dir=run.OUT) as scratch:
            path = Path(scratch) / "checkpoint.jsonl"
            first = {"query_id": "q-01", "question": "测试"}
            second = {"query_id": "q-02", "question": "Another question"}
            log = run.CheckpointLog(path, "scope")
            log.put(first)
            first_bytes = path.read_bytes()
            with path.open("ab") as stream:
                stream.write(b'{"scope":"scope","record":{"query_id":"q-02"')
            resumed = run.CheckpointLog(path, "scope")
            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertEqual(resumed.completed, {"q-01": first})
            resumed.put(second)
            self.assertEqual(run.CheckpointLog(path, "scope").completed, {"q-01": first, "q-02": second})
            with self.assertRaisesRegex(ValueError, "already checkpointed"):
                resumed.put(first)
            with self.assertRaisesRegex(ValueError, "input signature changed"):
                run.CheckpointLog(path, "changed-scope")

    def test_answerable_heldout_is_rejected_before_candidate_access(self):
        with (
            patch.object(run, "cohorts", return_value=({"tune"}, {"validation"}, {"null"})),
            patch.object(run, "rows", side_effect=AssertionError("Open read outside the experiment cohort")),
            patch.object(run, "iter_pools", side_effect=AssertionError("Opened candidate data")),
        ):
            with self.assertRaisesRegex(ValueError, "frozen development and null cohorts"):
                next(run.inputs(["heldout"]))

    @unittest.skipUnless(all(path.is_file() for path in REPLAY_INPUTS), "frozen local P5-B replay artifacts are unavailable")
    def test_frozen_pilot_and_null_replay_without_gold_or_models(self):
        tune, _, null_ids = run.cohorts()
        pilot = sorted(tune, key=lambda qid: sha256(f"ege-pilot-42:{qid}".encode()).digest())[:run.PILOT_LIMIT]
        null = min(null_ids)
        requested = set(pilot) | {null}
        development_ranks = {
            row["query_id"]: row["final_top10_by_rrf_weight"]["2"]
            for row in rows(run.DEV_RANKS) if row["query_id"] in pilot and row["pool_id"] == "P5"
        }
        null_rank = next(
            row["optimized_ranking"] for row in rows(run.NULL_RANKS)
            if row["query_id"] == null and row["pool_id"] == "P5" and row["scored"]
        )
        expected = {**development_ranks, null: null_rank}
        read_archive = run.rows

        def frozen_ranks_only(path):
            self.assertIn(path, (run.DEV_RANKS, run.NULL_RANKS))
            return read_archive(path)

        with (
            patch.object(run, "rows", side_effect=frozen_ranks_only),
            patch.object(run, "local_jina_backend", side_effect=AssertionError("Started a model")),
            patch("hoprag.dataset.load_dataset", side_effect=AssertionError("Loaded dataset labels")),
        ):
            replay = list(run.inputs(requested))
        self.assertEqual({row[0] for row in replay}, requested)
        for qid, question, baseline, jina, fixed, _ in replay:
            with self.subTest(query_id=qid):
                original = [doc["document_id"] for doc in expected[qid]]
                self.assertEqual(baseline[:len(original)], original)
                self.assertTrue(question)
                self.assertEqual(len(baseline), len(set(baseline)))
                self.assertEqual(set(baseline), {row["document_id"] for row in jina})
                self.assertLessEqual(set(baseline), set(fixed["30"]))
                self.assertLessEqual(set(fixed["30"]), set(fixed["50"]))
                self.assertLessEqual(set(fixed["50"]), set(fixed["100"]))


if __name__ == "__main__":
    unittest.main()
