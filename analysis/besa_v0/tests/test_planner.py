import json
from pathlib import Path
import unittest

from analysis.besa_v0.planner import plan


ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "data/raw/multihop_rag/MultiHopRAG.json"


class QueryNeedPlannerTests(unittest.TestCase):
    def test_simple_subject_intersection_generates_two_needs(self):
        question = "Which company launched a new satellite in 2022 and received a defense contract in 2023?"
        result = plan(question)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.needs), 2)
        self.assertEqual(result.needs[1].need_text, "received a defense contract in 2023")
        self.assertEqual(result.operations[0].kind, "same_answer_entity")
        self.assertEqual(result.operations[0].depends_on, ("n1", "n2"))

    def test_coordinated_article_claims_are_separate_needs(self):
        question = ("Did the Sporting News article describe increases in ticket prices, while the TechCrunch "
                    "article reported new investments in the ticketing platform?")
        result = plan(question)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.needs), 2)
        self.assertIn("Sporting News", result.needs[0].need_text)
        self.assertIn("TechCrunch", result.needs[1].need_text)
        self.assertEqual(result.operations[0].kind, "compare_report_claims")

    def test_sources_dates_and_publication_comparison_are_retained(self):
        question = ("Between the TechCrunch report on Apple published on October 7, 2023, and the Engadget "
                    "report on Apple published on January 3, 2024, was there consistency in the description?")
        result = plan(question)
        self.assertTrue(result.accepted)
        self.assertIn("TechCrunch", result.needs[0].need_text)
        self.assertIn("October 7, 2023", result.needs[0].need_text)
        self.assertIn("Engadget", result.needs[1].need_text)
        self.assertIn("January 3, 2024", result.needs[1].need_text)
        self.assertEqual(result.operations[0].text, question)

    def test_negation_and_numeric_scope_are_kept_verbatim(self):
        question = ("Did the TechCrunch article report that Apple did not sell the device before 2024, "
                    "while the Engadget article indicated the preorder offered no more than 500 units?")
        result = plan(question)
        self.assertTrue(result.accepted)
        self.assertIn("did not sell the device before 2024", result.needs[0].need_text)
        self.assertIn("no more than 500 units", result.needs[1].need_text)

    def test_unresolved_pronoun_falls_back(self):
        question = ("Did the Wired article name the software architect, while the TechCrunch article "
                    "suggested that she had invested in a competing company?")
        result = plan(question)
        self.assertFalse(result.accepted)
        self.assertEqual(result.fallback_reason, "ambiguous_reference")
        self.assertEqual(result.needs, ())
        self.assertEqual(result.question, question)

    @unittest.skipUnless(QUESTIONS.is_file(), "original MultiHop-RAG questions are a local fixture")
    def test_known_unsafe_split_patterns_on_preserved_questions(self):
        questions = json.loads(QUESTIONS.read_bytes())
        q1799 = plan(questions[1798]["query"])
        self.assertTrue(q1799.accepted)
        self.assertEqual(len(q1799.needs), 2)
        self.assertIn("NBA betting sites and apps published on October 2, 2023", q1799.needs[0].need_text)
        self.assertIn("November 6, 2023", q1799.needs[1].need_text)
        self.assertEqual(q1799.operations[0].kind, "compare_report_claims")

        q2270 = plan(questions[2269]["query"])
        self.assertFalse(q2270.accepted)
        self.assertEqual(q2270.fallback_reason, "ambiguous_reference")
        self.assertFalse(q2270.operations)

    def test_source_only_fragment_after_an_auxiliary_is_rejected(self):
        question = ("Which Western Bulldogs player played for three AFLW clubs, has contended with kidney "
                    "disease during the season, and was reported on by The Age?")
        self.assertFalse(plan(question).accepted)

    def test_gold_objects_are_rejected_at_the_planner_boundary(self):
        class GoldForbidden:
            def __getattr__(self, attribute):
                self.fail(f"Planner tried to read {attribute}")

        with self.assertRaises(TypeError):
            plan(GoldForbidden())


if __name__ == "__main__":
    unittest.main()