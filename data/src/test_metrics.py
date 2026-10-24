"""Small, manually reproducible checks of the evaluation convention."""

import unittest

from metrics import average_precision_at_k, mean_average_precision_at_k


class MetricTests(unittest.TestCase):
    def test_manual_four_relevant(self):
        # Hits at ranks 1, 3, 5: (1 + 2/3 + 3/5) / 4 = 17/30.
        self.assertAlmostEqual(average_precision_at_k({11, 22, 33, 44}, [11, 99, 22, 88, 33]), 17 / 30)

    def test_denominator_capped_at_five(self):
        # Seven relevant items: (1 + 1 + 3/4 + 4/5) / 5 = 0.71.
        self.assertAlmostEqual(average_precision_at_k(set(range(1, 8)), [1, 2, 9, 4, 5]), 0.71)

    def test_duplicate_predictions_do_not_repeat_hits(self):
        self.assertAlmostEqual(average_precision_at_k({11, 22}, [11, 11, 22]), (1 + 2 / 3) / 2)

    def test_empty_relevance(self):
        self.assertEqual(average_precision_at_k(set(), [1, 2, 3, 4, 5]), 0.0)

    def test_cutoff_and_no_hits(self):
        self.assertEqual(average_precision_at_k({6}, [1, 2, 3, 4, 5, 6]), 0.0)

    def test_distinct_relevance(self):
        self.assertEqual(average_precision_at_k([1, 1, 2], [1, 2]), 1.0)

    def test_cohort_includes_empty_truth_users(self):
        self.assertEqual(mean_average_precision_at_k({"a": {1}}, {"a": [1], "b": [2]}, ["a", "b"]), 0.5)

    def test_missing_predictions_are_not_silently_excluded(self):
        with self.assertRaises(KeyError):
            mean_average_precision_at_k({"a": {1}}, {}, ["a"])


if __name__ == "__main__":
    unittest.main()
