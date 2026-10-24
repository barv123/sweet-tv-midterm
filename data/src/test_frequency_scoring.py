"""Checks duplicate-airing handling and the planned decay variants."""

from datetime import date, timedelta
import unittest

from frequency_experiment import decayed_frequency, unique_airing_frequency


class FrequencyScoreTests(unittest.TestCase):
    def test_airing_frequency_deduplicates_airing_keys(self):
        self.assertEqual(unique_airing_frequency([12, 12, 13, 14, 14]), 3)

    def test_no_decay_is_event_frequency(self):
        self.assertEqual(unique_airing_frequency([1, 2, 3, 3]), 3)

    def test_half_life_weight_is_one_half_after_one_half_life(self):
        cutoff = date(2020, 4, 6)
        self.assertEqual(decayed_frequency([cutoff-timedelta(days=90)], cutoff, 90), 0.5)
        self.assertEqual(decayed_frequency([cutoff-timedelta(days=30)], cutoff, 30), 0.5)

    def test_same_day_distinct_airings_each_count(self):
        cutoff = date(2020, 4, 6)
        watched = cutoff-timedelta(days=90)
        self.assertEqual(decayed_frequency([watched, watched], cutoff, 90), 1.0)

    def test_invalid_half_life_rejected(self):
        with self.assertRaises(ValueError):
            decayed_frequency([date(2020, 4, 1)], date(2020, 4, 6), 0)


if __name__ == "__main__":
    unittest.main()
