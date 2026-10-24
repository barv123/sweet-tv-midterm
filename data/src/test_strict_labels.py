"""Guard strict airborne watch labels and separate train-signal threshold."""

import unittest

from strict_relevance_experiment import ground_truth_positive, training_signal


class StrictLabelTests(unittest.TestCase):
    def test_strictly_more_than_80_percent_is_relevant(self):
        self.assertTrue(ground_truth_positive(81, 100))
        self.assertFalse(ground_truth_positive(80, 100))
        self.assertFalse(ground_truth_positive(79, 100))

    def test_train_threshold_is_separate_and_strictly_more_than_50_percent(self):
        self.assertFalse(training_signal(50, 100))
        self.assertTrue(training_signal(51, 100))
        self.assertFalse(ground_truth_positive(50, 100))

    def test_unrelated_sessions_are_not_combined(self):
        # Two separate 51% sessions are each training signals but neither is a label;
        # no helper sums their durations into an 80% event.
        session_coverages = [(51, 100), (51, 100)]
        self.assertTrue(all(training_signal(overlap, airing) for overlap, airing in session_coverages))
        self.assertFalse(any(ground_truth_positive(overlap, airing) for overlap, airing in session_coverages))

    def test_airing_duration_is_the_denominator(self):
        # A long viewer session that intersects only 81/600 seconds is not relevant.
        session_seconds, overlap_seconds, airing_seconds = 3600, 81, 600
        self.assertGreater(session_seconds, airing_seconds)
        self.assertFalse(ground_truth_positive(overlap_seconds, airing_seconds))

    def test_nonpositive_airing_is_never_eligible(self):
        self.assertFalse(training_signal(0, 0))
        self.assertFalse(ground_truth_positive(10, 0))


if __name__ == "__main__":
    unittest.main()
