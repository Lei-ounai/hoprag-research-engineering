"""Checks for the restricted offline sampling frame and design weights."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.evidence_verification_benchmark.sampling import (
    _gold_sources,
    _top_quintile,
    assign_tags,
    choose_stratified,
    read_gate_frames,
    read_witness_diagnostics,
)


def candidate(cohort, *tags):
    return {"cohort": cohort, "overlapping_tags": list(tags)}


class SamplingTests(unittest.TestCase):
    def test_stratified_draw_reproduces_when_input_order_changes(self):
        units = {(f"q-{i:03}", "doc-001"): candidate("development", "ordinary")
                 for i in range(30)}
        units.update({(f"q-{100+i:03}", "doc-002"): candidate("null", "ordinary")
                      for i in range(10)})
        targets = {"development": 8, "null": 4}
        first = choose_stratified(units, seed=27, targets=targets)
        second = choose_stratified(dict(reversed(list(units.items()))), seed=27, targets=targets)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], choose_stratified(units, seed=28, targets=targets)[0])
        self.assertEqual(len({(r["query_id"], r["document_id"]) for r in first[0]}), 12)
        self.assertEqual(len({r["case_id"] for r in first[0]}), 12)
        self.assertEqual(Counter(r["cohort"] for r in first[0]), targets)

    def test_overlap_precedence_and_positive_weights_reconstruct_each_population(self):
        units = {("q-certainty", "doc-1"): candidate("development", "e3_control_accepted", "margin_rejected")}
        units.update({(f"q-{i:03}", "doc-1"): candidate("development", "ordinary")
                      for i in range(20)})
        units.update({(f"q-{100+i:03}", "doc-1"): candidate("null", "ordinary")
                      for i in range(30)})
        selected, stats = choose_stratified(units, targets={"development": 8, "null": 9})
        certainty = next(r for r in selected if r["query_id"] == "q-certainty")
        self.assertEqual(certainty["stratum"], "e3_control_accepted")
        self.assertEqual(certainty["inclusion_probability"], 1.0)
        self.assertEqual(certainty["sample_weight"], 1.0)
        margin = next(r for r in stats if r["cohort"] == "development" and r["stratum"] == "margin_rejected")
        self.assertEqual((margin["population"], margin["selected"]), (0, 0))
        self.assertEqual(margin["tagged_population_including_overlap"], 1)
        self.assertEqual(margin["tagged_selected_including_overlap"], 1)
        for cohort, expected in (("development", 21), ("null", 30)):
            total = sum(r["sample_weight"] for r in selected if r["cohort"] == cohort)
            self.assertAlmostEqual(total, expected)
        for row in selected:
            self.assertGreater(row["inclusion_probability"], 0.0)
            self.assertAlmostEqual(row["inclusion_probability"] * row["sample_weight"], 1.0)

    def test_dev_population_changes_do_not_change_null_selection(self):
        null = {(f"q-{100+i:03}", "doc-1"): candidate("null", "ordinary")
                for i in range(20)}
        dev = {(f"q-{i:03}", "doc-1"): candidate("development", "ordinary")
               for i in range(30)}
        targets = {"development": 6, "null": 7}
        first, _ = choose_stratified({**dev, **null}, targets=targets)
        altered_dev = {key: value for key, value in dev.items() if key[0] < "q-015"}
        second, _ = choose_stratified({**altered_dev, **null}, targets=targets)
        self.assertEqual(
            {row["query_id"] for row in first if row["cohort"] == "null"},
            {row["query_id"] for row in second if row["cohort"] == "null"},
        )

    def test_shadow_novelty_is_separate_from_actual_novelty_rejection(self):
        frame = {("q-1", "new")}
        usable = [{"query_id": "q-1", "document_id": doc, "status": "supported",
                   "coverage_keys": ["claim"], "reasons": [], "features": {}}
                  for doc in ("new", "original")]
        with patch("analysis.evidence_verification_benchmark.sampling._json_rows", return_value=iter(usable)):
            flags = read_witness_diagnostics(Path("unused"), frame, {"q-1": ("original",)})
        self.assertEqual(flags["q-1", "new"], {"novelty_shadow_review"})
        score = {"status": "rejected", "reason": "below_quality_gate", "quality_pass": False,
                 "novelty_pass": None, "split": "tune"}
        tags = assign_tags(score, control=score, witness_flags=flags["q-1", "new"],
                           gold_required=False, high_score=False, e2_promoted=False)
        self.assertIn("novelty_shadow_review", tags)
        self.assertNotIn("novelty_rejected", tags)

    def test_duplicate_archived_pair_in_same_method_is_rejected(self):
        row = {"method": "E4-B3", "split": "tune", "query_id": "q-1", "document_id": "d-1"}
        with patch("analysis.evidence_verification_benchmark.sampling._ledger_rows", return_value=iter([row, row])):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_gate_frames(Path("unused"))

    def test_high_scores_are_relative_and_ties_are_deterministic(self):
        frame = {("q-1", f"doc-{i}"): {"candidate_parent_score": -8.0} for i in range(5)}
        frame.update({("q-2", f"doc-{i}"): {"candidate_parent_score": 2.0} for i in range(6)})
        self.assertEqual(_top_quintile(frame), {("q-1", "doc-0"), ("q-2", "doc-0"), ("q-2", "doc-1")})

    def test_gold_view_reads_only_ids_and_archived_decision_is_unchanged(self):
        class Source:
            source_document_id = "doc-gold"

            @property
            def evidence_text(self):
                raise AssertionError("The sampling rule only needs source IDs")

        class Question:
            query_id = "q-target"
            evidence_list = (Source(),)

            @property
            def answer(self):
                raise AssertionError("Sampling must never use answers")

        dataset = SimpleNamespace(queries=[Question(), SimpleNamespace(query_id="q-heldout")])
        with patch("analysis.evidence_verification_benchmark.sampling.load_dataset", return_value=dataset):
            refs = _gold_sources(Path("unused"), {"q-target"})
        self.assertEqual(refs, {"q-target": {"doc-gold"}})
        row = {"status": "rejected", "reason": "no_question_link", "quality_pass": False,
               "novelty_pass": None, "split": "validation"}
        unchanged = row.copy()
        for gold in (True, False):
            observed = assign_tags(row, control=row, witness_flags=(), gold_required=gold,
                                   high_score=True, e2_promoted=False)
            self.assertEqual("gold_required_quality_reject" in observed, gold)
            self.assertEqual(row, unchanged)


if __name__ == "__main__":
    unittest.main()

