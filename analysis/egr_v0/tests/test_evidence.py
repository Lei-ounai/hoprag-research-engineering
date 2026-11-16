from dataclasses import asdict
import unittest

from analysis.egr_v0.evidence import profile_question, verify


class EvidenceVerificationTests(unittest.TestCase):
    def check(self, question, text, *, title="", metadata=None, expansions=(), visibility=True):
        return verify(profile_question(question), "id", "doc-a", "doc-a::00", text,
                      title=title, metadata=metadata, expansion_queries=expansions, jina_score=1.0,
                      model_visible_chunk_chars=len(text) if visibility is True else visibility)

    def test_founder_has_attested_actor_and_original_excerpt(self):
        text = "Background about an unrelated business.\nJane Doe founded Acme in 2001."
        result = self.check("Who founded Acme?", text)
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.entity_roles, ("founder",))
        self.assertEqual(result.matched_entities, ("acme",))
        self.assertEqual(text[result.evidence_start:result.evidence_end], result.evidence_excerpt)
        self.assertTrue(result.within_model_visible_text)
        self.assertEqual(asdict(result)["query_id"], "id")

    def test_ceo_conflicts_with_requested_founder_role(self):
        result = self.check("Who founded Acme?", "Jane Doe is the CEO of Acme.")
        self.assertEqual(result.status, "conflict")
        self.assertIn("entity_role_conflict", result.reasons)
        self.assertEqual(result.coverage_keys, ())

    def test_role_attested_for_different_company(self):
        result = self.check("Who founded Acme?", "Jane Doe founded Globex and then advised Acme.")
        self.assertEqual(result.status, "uncertain")
        self.assertIn("entity_role_unverified", result.reasons)

    def test_passive_role_attestation(self):
        result = self.check("Who founded Acme?", "Acme was founded by Jane Doe in 2001.")
        self.assertEqual(result.status, "supported")

    def test_location_and_birth_are_different_relations(self):
        result = self.check("Where was Jane Doe born?", "Jane Doe is based in Germany.")
        self.assertEqual(result.status, "conflict")
        self.assertIn("relation_mismatch", result.reasons)
        self.assertEqual(result.quality, 0)

    def test_proper_names_and_birth_word_are_insufficient(self):
        result = self.check("Where was Jane Doe born?", "Jane Doe lives in Calgary, but Sarah Trent was born in Paris.")
        self.assertNotEqual(result.status, "supported")
        positive = self.check("Where was Jane Doe born?", "Jane Doe was born in Paris.")
        self.assertEqual(positive.status, "supported")
        self.assertEqual(positive.relation_terms, ("birth",))

    def test_acquisition_direction_must_match_the_question(self):
        question = "Which firm acquired Acme?"
        wrong = self.check(question, "Acme acquired Meridian.")
        right = self.check(question, "Acme was acquired by Meridian.")
        self.assertNotEqual(wrong.status, "supported")
        self.assertEqual(right.status, "supported")

    def test_event_date_does_not_copy_publication_date(self):
        question = "Who founded Acme in 2001?"
        wrong = self.check(question, "Jane Doe founded Acme in 2002.",
                           metadata={"published_at": "2001-03-02T00:00:00+00:00"})
        self.assertEqual(wrong.status, "conflict")
        self.assertIn("date_conflict", wrong.reasons)
        right = self.check(question, "Jane Doe founded Acme in 2001.",
                           metadata={"published_at": "2023-03-02T00:00:00+00:00"})
        self.assertEqual(right.status, "supported")
        self.assertEqual(right.features["date_kinds"], ("event",))

    def test_explicit_publication_date_uses_iso_metadata(self):
        question = "Which Fortune article on Orchid Labs was published before November 9, 2023?"
        text = "Orchid Labs expanded into Berlin during 2020."
        late = self.check(question, text, metadata={"source": "Fortune", "published_at": "2023-11-10T00:00:00Z"})
        self.assertEqual(late.status, "conflict")
        self.assertIn("date_conflict", late.reasons)
        early = self.check(question, text, metadata={"source": "Fortune", "published_at": "2023-11-08T00:00:00Z"})
        self.assertEqual(early.status, "supported")
        self.assertEqual(early.features["date_kinds"], ("publication",))

    def test_metadata_publisher_mismatch_is_not_overruled_by_text_mention(self):
        question = "Which firm acquired Acme, according to Fortune?"
        text = "Fortune was among news organizations. Meridian acquired Acme."
        metadata = {"source": "Reuters", "url": "https://example.test/story", "published_at": "2023-01-01"}
        result = self.check(question, text, metadata=metadata)
        self.assertEqual(result.source_metadata, metadata)
        self.assertIn("source_mismatch", result.reasons)
        self.assertEqual(result.status, "conflict")

    def test_independent_named_media_clauses_kept_separate(self):
        question = ("Did the Polygon report on the Steam Deck OLED improvements, published before November 9, 2023, "
                    "and the Engadget review of the Steam Deck OLED published shortly after, disagree on the "
                    "enhancements made to the new iteration of the Steam Deck by Valve?")
        clauses = profile_question(question).clauses
        self.assertEqual([item.source_names for item in clauses], [("polygon",), ("engadget",)])
        self.assertEqual([item.dates[0].time_kind for item in clauses], ["publication", "relative_publication"])

    def test_unrecognized_semantic_relation_remains_uncertain(self):
        text = "Acme is mentioned next to Miami and the famous playwright in a publication."
        result = self.check("Why did Acme make the playwright more optimistic?", text)
        self.assertEqual(result.status, "uncertain")
        self.assertIn("relation_unverified", result.reasons)

    def test_model_window_does_not_credit_missing_or_partial_relation(self):
        text = "Published during a busy week.\nJane Doe founded Acme as reported that week."
        for bound in (21, text.index(" as reported"), None):
            with self.subTest(bound=bound):
                result = self.check("Who founded Acme?", text, visibility=bound)
                self.assertEqual(result.status, "uncertain")
                self.assertEqual(result.coverage_keys, ())

    def test_expansion_match_is_diagnostic_only_and_gold_absent(self):
        result = self.check("Where was Jane Doe born?",
                            "Jane Doe is based in Germany and the patent finally cleared.",
                            expansions=("patent cleared",))
        self.assertGreater(result.expansion_query_match, 0)
        self.assertNotEqual(result.status, "supported")
        self.assertFalse(set(asdict(result)) & {"evidence_list", "gold_sources", "answer", "question_type"})

