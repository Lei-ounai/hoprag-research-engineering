import unittest

from analysis.reranking_optimization.evaluate import transition_metrics


class RankingTransitionTests(unittest.TestCase):
    def test_regression_rate_uses_only_originally_complete_queries(self):
        metrics = transition_metrics(
            [True, True, True, False, False],
            [True, False, True, True, True],
        )
        self.assertEqual(metrics["original_complete"], 3)
        self.assertEqual(metrics["regressed_queries"], 1)
        self.assertEqual(metrics["ranking_regression_rate"], 1 / 3)

    def test_recovery_rate_uses_all_originally_incomplete_queries(self):
        metrics = transition_metrics(
            [True, True, True, False, False],
            [False, True, True, False, True],
        )
        self.assertEqual(metrics["recovered_queries"], 1)
        self.assertEqual(metrics["recovery_rate"], 1 / 2)
        self.assertEqual(metrics["net_gain"], 0)
        self.assertIsNone(transition_metrics([True], [True])["recovery_rate"])


if __name__ == "__main__":
    unittest.main()
