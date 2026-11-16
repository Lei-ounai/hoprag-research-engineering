"""Offline tests for ranking and novelty decisions on explicit witness inputs."""

import unittest
from dataclasses import asdict

from analysis.egr_v0.selector import GateConfig, choose, prepare, replace


BASELINE = tuple(f"old-{pos:02d}" for pos in range(1, 16))
NOVEL = ("new-one", "new-two", "new-three")
ORIGINAL_KEY = "qclause-01:relation:appointed"
OTHER_KEY = "qclause-02:relation:resigned"
SCORES = {
    **{doc: 0.5 for doc in BASELINE},
    "new-one": 3.0,
    "new-two": 2.5,
    "new-three": 2.0,
}


def witness(document, key, score=None, *, text=None, status="supported", quality=0.85, **extra):
    return {
        "document_id": document,
        "chunk_id": f"{document}::0000",
        "coverage_keys": (key,) if key else (),
        "status": status,
        "quality": quality,
        "jina_score": SCORES[document] if score is None else score,
        "evidence_excerpt": text or f"Article {document} directly states the relationship specified in the question.",
        "source_metadata": {"source": "Wire"},
        **extra,
    }


class GuardedSource(dict):
    def get(self, key, default=None):
        if key in {"gold", "gold_sources", "answer", "is_answerable", "question_type", "evidence_list"}:
            raise AssertionError(f"Unexpected annotation access: {key}")
        return super().get(key, default)


class SelectorTests(unittest.TestCase):
    def test_all_three_limits_protect_first_seven_and_keep_unique_parents(self):
        witnesses = {d: [witness(d, f"qclause-{i+1:02d}:relation:fact")]
                     for i, d in enumerate(NOVEL)}
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                out = replace(BASELINE, SCORES, witnesses, challengers=(*NOVEL, "new-one"),
                              config=GateConfig(max_replacements=limit))
                self.assertEqual(len(out.replacements), limit)
                self.assertEqual(out.document_ids[:7], list(BASELINE[:7]))
                self.assertEqual(len(out.document_ids), 10)
                self.assertEqual(len(out.ranking), len(set(out.ranking)))
                self.assertEqual(set(out.ranking), set(BASELINE) | set(NOVEL[:limit]))
                self.assertTrue(all(r.position in {8, 9, 10} for r in out.replacements))
                self.assertEqual(out.replacements[0].candidate_document_id, "new-one")
                self.assertEqual(out.replacements[0].position, 10)

    def test_bad_constraint_or_missing_witness_preserves_full_baseline(self):
        values = {NOVEL[0]: [witness(NOVEL[0], OTHER_KEY, status="conflict", reasons=("date_conflict",))],
                  NOVEL[1]: [witness(NOVEL[1], None, status="uncertain", reasons=("relation_unverified",))]}
        out = replace(BASELINE, SCORES, values, challengers=NOVEL)
        self.assertEqual(out.ranking, list(BASELINE))
        self.assertFalse(out.replacements)
        reasons = {decision.document_id: decision.reason for decision in out.decisions}
        self.assertEqual(reasons[NOVEL[0]], "date_conflict")
        self.assertEqual(reasons[NOVEL[1]], "relation_unverified")
        self.assertEqual(reasons[NOVEL[2]], "no_verified_witness")

    def test_novelty_includes_entire_current_set(self):
        witnesses = {BASELINE[0]: [witness(BASELINE[0], ORIGINAL_KEY)],
                     NOVEL[0]: [witness(NOVEL[0], ORIGINAL_KEY)],
                     NOVEL[1]: [witness(NOVEL[1], OTHER_KEY)],
                     NOVEL[2]: [witness(NOVEL[2], OTHER_KEY)]}
        out = replace(BASELINE, SCORES, witnesses, challengers=NOVEL,
                      config=GateConfig(max_replacements=3))
        self.assertEqual([r.candidate_document_id for r in out.replacements], [NOVEL[1]])
        reasons = {decision.document_id: decision.reason for decision in out.decisions}
        self.assertEqual(reasons[NOVEL[0]], "already_covered")
        self.assertEqual(reasons[NOVEL[2]], "already_covered")

    def test_fair_margin_uses_supporting_witness_and_actual_slot(self):
        scores = {**SCORES, **{BASELINE[index]: 1.5 for index in (7, 8, 9)}}
        challenger = witness(NOVEL[0], OTHER_KEY, score=1.625)
        stronger_unsupported = {
            **witness(NOVEL[0], None, status="uncertain"),
            "chunk_id": f"{NOVEL[0]}::0001",
        }
        witnesses = {NOVEL[0]: [challenger, stronger_unsupported]}
        rejected = replace(BASELINE, scores, witnesses, challengers=NOVEL[:1],
                           config=GateConfig(min_margin=0.25))
        self.assertEqual(rejected.ranking, list(BASELINE))
        decision = rejected.decisions[0]
        self.assertEqual(decision.reason, "insufficient_replacement_margin")
        self.assertEqual(decision.score_margin, 0.125)
        self.assertEqual(decision.incumbent_rank, 10)
        self.assertEqual(decision.candidate_parent_score, 3.0)
        self.assertEqual(decision.candidate_witness_score, 1.625)
        accepted = replace(BASELINE, scores, witnesses, challengers=NOVEL[:1],
                           config=GateConfig(min_margin=0.125))
        self.assertEqual(accepted.replacements[0].champion_chunk_id, challenger["chunk_id"])
        self.assertEqual(accepted.replacements[0].incumbent_document_id, BASELINE[9])

    def test_actual_loss_protects_only_existing_verified_claim(self):
        witnesses = {BASELINE[9]: [witness(BASELINE[9], ORIGINAL_KEY)],
                     NOVEL[0]: [witness(NOVEL[0], OTHER_KEY)]}
        scores = {**SCORES, BASELINE[7]: 1.5, BASELINE[8]: 1.0}
        out = replace(BASELINE, scores, witnesses, challengers=NOVEL[:1])
        self.assertEqual(out.replacements[0].position, 9)
        self.assertEqual(out.document_ids[9], BASELINE[9])
        scores.update({BASELINE[7]: 4.0, BASELINE[8]: 4.0})
        out = replace(BASELINE, scores, witnesses, challengers=NOVEL[:1])
        self.assertFalse(out.replacements)
        self.assertEqual(out.decisions[0].reason, "protected_sole_evidence")
        self.assertEqual(out.decisions[0].protected_coverage_keys, (ORIGINAL_KEY,))

    def test_near_copy_requires_comparable_context(self):
        body = ("The filing confirmed the committee officers presented the statement to the "
                "industry association during the recorded annual meeting on 18 June.")
        exact = {BASELINE[0]: [witness(BASELINE[0], ORIGINAL_KEY, text=body)],
                 NOVEL[0]: [witness(NOVEL[0], "qclause-01:relation:chair", text=body)]}
        out = replace(BASELINE, SCORES, exact, challengers=NOVEL[:1])
        self.assertEqual(out.decisions[0].reason, "near_duplicate_witness")
        exact[NOVEL[0]][0]["evidence_excerpt"] = body.replace("18 June", "24 July")
        out = replace(BASELINE, SCORES, exact, challengers=NOVEL[:1])
        self.assertEqual(out.replacements[0].candidate_document_id, NOVEL[0])

    def test_uncomparable_scores_do_not_count_as_negative_evidence(self):
        scores = dict(SCORES)
        del scores[BASELINE[9]]
        values = {NOVEL[0]: [witness(NOVEL[0], OTHER_KEY)]}
        values[NOVEL[0]][0]["jina_score"] = None
        out = replace(BASELINE, scores, values, challengers=NOVEL[:1])
        self.assertFalse(out.replacements)
        self.assertEqual(out.decisions[0].reason, "no_comparable_witness_score")

    def test_failure_returns_complete_original_order(self):
        for reason, scores in (("model_error", SCORES), (None, {})):
            with self.subTest(reason=reason):
                out = replace(BASELINE, scores, {}, challengers=NOVEL, failure_reason=reason)
                self.assertEqual(out.ranking, list(BASELINE))
                self.assertEqual(out.fallback_reason, reason or "no_comparable_incumbent_scores")
                self.assertEqual(len(out.decisions), len(NOVEL))

    def test_malformed_lookup_preserves_baseline(self):
        malformed = replace(BASELINE, SCORES, {}, challengers=[["invalid"]])
        self.assertEqual(malformed.ranking, list(BASELINE))
        self.assertEqual(malformed.fallback_reason, "invalid_competition_inputs")
        row = witness(NOVEL[0], OTHER_KEY)
        duplicates = replace(BASELINE, SCORES, {NOVEL[0]: [row, row]}, challengers=NOVEL[:1])
        self.assertEqual(duplicates.ranking, list(BASELINE))
        self.assertEqual(duplicates.fallback_reason, "invalid_competition_inputs")

    def test_question_labels_are_inaccessible_and_selection_is_stable(self):
        guarded = GuardedSource({d: [GuardedSource(witness(d, OTHER_KEY))]
                                 for d in NOVEL[:2]})
        scores = {**SCORES, NOVEL[0]: SCORES[NOVEL[1]]}
        guarded[NOVEL[0]][0]["jina_score"] = scores[NOVEL[0]]
        a = choose(prepare(BASELINE, scores, guarded, challengers=NOVEL[:2]))
        b = choose(prepare(BASELINE, dict(reversed(tuple(scores.items()))), guarded,
                           challengers=reversed(NOVEL[:2])))
        self.assertEqual(asdict(a), asdict(b))
        self.assertEqual(a.replacements[0].candidate_document_id, NOVEL[0])

    def test_support_beyond_model_input_cannot_get_score_credit(self):
        values = {NOVEL[0]: [witness(NOVEL[0], OTHER_KEY, within_model_visible_text=False)]}
        out = replace(BASELINE, SCORES, values, challengers=NOVEL[:1])
        self.assertFalse(out.replacements)
        self.assertEqual(out.ranking, list(BASELINE))


if __name__ == "__main__":
    unittest.main()

