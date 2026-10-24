"""Reproducible mathematical, ranking, eligibility, and fallback checks."""

import unittest

import duckdb
import numpy as np
from scipy import sparse

from recommendation_methods import (content_features, content_score_blocks, mine_association_rules,
                                    ranked_indices, reciprocal_rank_fusion, rule_score_blocks)


class RecommendationTests(unittest.TestCase):
    def test_rank_ties_and_exact_five(self):
        ids = np.array([9, 8, 7, 6, 5, 4])
        pop = np.array([3, 3, 1, 1, 1, 1])
        actual = ids[ranked_indices(np.zeros(6), ids, pop, 5)].tolist()
        self.assertEqual(actual, [8, 9, 4, 5, 6])
        self.assertEqual(len(set(actual)), 5)
        self.assertTrue(set(actual) <= set(ids))

    def test_candidate_score_precedes_popularity(self):
        self.assertEqual(ranked_indices([0.1, 0.9], np.array([1, 2]), np.array([100, 0]), 2).tolist(), [1, 0])

    def test_directional_rule_arithmetic_and_no_self_rules(self):
        binary = sparse.csr_matrix([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]])
        rules, info = mine_association_rules(binary, np.array([10, 20, 30]), np.array([0, 1, 2]),
            min_pair_count=1, min_lift=0, min_confidence=0, min_item_viewers=1)
        by_pair = {(row["antecedent"], row["consequent"]): row for row in info["sample_rules"]}
        self.assertAlmostEqual(by_pair[10, 20]["support"], 0.5)
        self.assertAlmostEqual(by_pair[10, 20]["confidence"], 2 / 3)
        self.assertAlmostEqual(by_pair[20, 10]["confidence"], 1)
        self.assertAlmostEqual(by_pair[10, 20]["lift"], 4 / 3)
        self.assertTrue(np.all(rules.diagonal() == 0))

    def test_zero_history_rule_fallback(self):
        history = sparse.csr_matrix((1, 6), dtype=float)
        rules = sparse.identity(6, format="csr")
        _, score = next(rule_score_blocks(history, rules, np.arange(6)))
        actual = ranked_indices(score[0], np.arange(1, 7), np.array([6, 5, 4, 3, 2, 1]), 5)
        self.assertEqual(actual.tolist(), [0, 1, 2, 3, 4])

    def test_content_uses_metadata_not_item_popularity(self):
        metadata = {1: {"category": "news", "metadata_period": 0},
                    2: {"category": "news", "metadata_period": 1},
                    3: {"category": "sports", "metadata_period": 0}}
        features, info = content_features(metadata, np.array([1, 2, 3]), False)
        _, score = next(content_score_blocks(sparse.csr_matrix([[1., 0, 0], [0, 0, 0]]), features, np.arange(3)))
        self.assertGreater(score[0, 1], score[0, 2])
        self.assertTrue(np.all(score[1] == 0))
        self.assertEqual(info["idf_fit_documents"], 2)

    def test_temporal_snapshot_excludes_future_and_crossing(self):
        con = duckdb.connect()
        result = con.execute("""SELECT id FROM (VALUES
            (1,TIMESTAMP '2020-05-17 22:00',TIMESTAMP '2020-05-18 00:00'),
            (2,TIMESTAMP '2020-05-17 23:00',TIMESTAMP '2020-05-18 00:01'),
            (3,TIMESTAMP '2020-05-18 01:00',TIMESTAMP '2020-05-18 02:00')) t(id,session_start,session_end)
            WHERE session_start>=TIMESTAMP '2020-03-09' AND session_end<=TIMESTAMP '2020-05-18'""").fetchall()
        self.assertEqual(result, [(1,)])
        con.close()

    def test_fusion_exact_five_and_deterministic(self):
        args = ([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], np.arange(1, 7), np.arange(6, 0, -1))
        self.assertEqual(reciprocal_rank_fusion(*args, limit=5), reciprocal_rank_fusion(*args, limit=5))
        self.assertEqual(len(set(reciprocal_rank_fusion(*args, limit=5))), 5)


if __name__ == "__main__":
    unittest.main()
