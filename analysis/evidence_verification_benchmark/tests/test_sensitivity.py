"""Offline regression tests for threshold interpretation and label isolation."""

from dataclasses import replace
import unittest

from analysis.egr_v0.selector import GateConfig, choose
from analysis.evidence_verification_benchmark.sensitivity import competition, observe, points, sets


class GuardedDict(dict):
    denied = {"answer", "gold", "evidence_list", "gold_source", "question_type", "is_answerable"}

    def __getitem__(self, item):
        if item in self.denied:
            raise AssertionError(f"Forbidden answer annotation accessed: {item}")
        return super().__getitem__(item)

    def get(self, item, default=None):
        if item in self.denied:
            raise AssertionError(f"Forbidden answer annotation accessed: {item}")
        return super().get(item, default)


BASELINE = [f"p{i:02d}" for i in range(1, 11)]
CANDIDATE = "p30"
FIRST_KEY = "clause-01:source:predicate-original"
NEW_KEY = "clause-01:source:predicate-new"
CURRENT = GateConfig(max_replacements=1, min_quality=0.90, min_margin=0.70, novelty_overlap=0.80)
ORIGINAL_TEXT = "The statement confirmed the company appointed the chief officer to begin work immediately in Boston last Monday."
NEW_TEXT = "The statement confirmed the company appointed the chief officer to begin work immediately in Chicago last Monday."


def witness(doc, key, text=NEW_TEXT, *, quality=0.90, status="supported", score=0.95, suffix="0"):
    return GuardedDict(document_id=doc, chunk_id=f"{doc}::{suffix}", status=status, quality=quality,
                       coverage_keys=[key] if key else [], evidence_excerpt=text, jina_score=score,
                       source_metadata={"source": "Example"}, within_model_visible_text=True, reasons=[])


def fixture(chunks, reference=None, score=0.95):
    visited = [*BASELINE, CANDIDATE]
    scored = {d: [{"jina_score": 0.40}] for d in BASELINE}
    scored[CANDIDATE] = [{"jina_score": score}]
    record = GuardedDict(baseline=BASELINE, fixed=visited, scoring=GuardedDict(documents=scored, fallback_reason=None),
                         answer="unused", gold={"unused": "unused"}, is_answerable=False)
    raw = {CANDIDATE: chunks}
    if reference:
        raw[BASELINE[0]] = [reference]
    return competition(record, raw)


class SensitivityTests(unittest.TestCase):
    def test_all_frozen_thresholds_are_present_and_novelty_direction_is_reversed(self):
        grid = list(points(CURRENT))
        floors = {p.stage: p.threshold for p in grid
                  if p.context == "frozen_one_factor" and p.stage_direction == "current"}
        self.assertEqual(floors, {"quality": 0.90, "novelty": 0.80, "margin": 0.70})
        point = next(p for p in grid if p.context == "frozen_one_factor" and p.stage == "novelty"
                     and p.threshold == 0.50)
        self.assertEqual(point.stage_direction, "stricter")
        self.assertEqual(point.config.min_quality, CURRENT.min_quality)
        self.assertEqual(point.config.min_margin, CURRENT.min_margin)

    def test_lower_quality_keeps_rule_support_requirement_and_reports_signed_delta(self):
        comp = fixture([witness(CANDIDATE, NEW_KEY, quality=0.65)])
        baseline = choose(comp, CURRENT)
        looser = choose(comp, replace(CURRENT, min_quality=0.60))
        totals = observe(looser, sets(baseline), "quality", {CANDIDATE}, {CANDIDATE}, set())
        self.assertEqual(totals["stage_new_pass_vs_reference"], 1)
        self.assertEqual(totals["stage_pass"], 1)
        self.assertEqual(totals["margin_pass"], 0)
        self.assertEqual(totals["replacement_pairs"], 0)
        self.assertEqual(totals["not_eligible_with_rule_support_proxy"], 1)
        uncertain = fixture([witness(CANDIDATE, None, status="uncertain", quality=1)])
        self.assertEqual(sets(choose(uncertain, replace(CURRENT, min_quality=0)))["quality"], set())

    def test_lower_novelty_overlap_can_veto_lexical_overlap_but_never_bypasses_coverage_key(self):
        old = witness(BASELINE[0], FIRST_KEY, ORIGINAL_TEXT, score=0.40)
        doc = fixture([witness(CANDIDATE, NEW_KEY)], reference=old)
        self.assertEqual(sets(choose(doc, replace(CURRENT, novelty_overlap=.95)))["novelty"], {CANDIDATE})
        self.assertEqual(choose(doc, replace(CURRENT, novelty_overlap=.50)).decisions[0].reason,
                         "near_duplicate_witness")
        covered = fixture([witness(CANDIDATE, FIRST_KEY)], reference=old)
        high = choose(covered, replace(CURRENT, novelty_overlap=1.0)).decisions[0]
        self.assertEqual(high.reason, "already_covered")
        self.assertFalse(high.novelty_pass)

    def test_zero_margin_still_requires_positive_and_proxies_are_positional(self):
        equal = fixture([witness(CANDIDATE, NEW_KEY, score=0.4)], score=0.4)
        zero = replace(CURRENT, min_margin=0.0)
        self.assertFalse(choose(equal, zero).decisions[0].margin_pass)
        comp = fixture([witness(CANDIDATE, NEW_KEY),
                        witness(CANDIDATE, None, status="conflict", score=0.2, suffix="1")])
        totals = observe(choose(comp, zero), sets(choose(comp, CURRENT)), "margin", {CANDIDATE},
                         {CANDIDATE}, {CANDIDATE})
        self.assertEqual(totals["new_replacement_pairs_vs_reference"], 1)
        self.assertEqual(totals["replacement_with_conflicting_block_proxy"], 1)
        self.assertEqual(totals["stage_not_reached"], 0)
        self.assertEqual(totals["not_quality_passed_with_rule_support_proxy"], 0)

    def test_truth_labels_are_not_consumed_for_null_or_development_inputs(self):
        # The fixture raises if the threshold logic ever reads any label field.
        comp = fixture([witness(CANDIDATE, NEW_KEY)])
        original = choose(comp, CURRENT)
        self.assertIsNone(original.fallback_reason)
        self.assertEqual(len(original.decisions), 1)
        self.assertEqual(original.decisions[0].reason, "insufficient_replacement_margin")


if __name__ == "__main__":
    unittest.main()

